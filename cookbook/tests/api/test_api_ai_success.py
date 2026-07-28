import json
from io import BytesIO
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from django.core.files.uploadedfile import SimpleUploadedFile
from django.urls import reverse
from django_scopes import scopes_disabled
from PIL import Image

from cookbook.ai_serializers import AI_RECIPE_IMPORT_RESPONSE_SCHEMA
from cookbook.helper.ai_helper import AI_OUTPUT_TOKEN_BUDGETS, CLOUDFLARE_GEMMA_MODEL
from cookbook.models import AiLog, AiProvider, PropertyType
from cookbook.tests.factories import FoodFactory, RecipeFactory


def completion_response(content):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))], )


def schema_recipe():
    return {
        'name': 'AI Soup',
        'recipeIngredient': ['1 cup water'],
        'recipeInstructions': ['Boil the water.'],
    }


@pytest.fixture
def ai_provider(space_1):
    return AiProvider.objects.create(
        name='test_provider',
        space=space_1,
        api_key='test-key',
        model_name='openai/test-model',
    )


@pytest.fixture
def ai_space(space_1, ai_provider):
    space_1.ai_provider = ai_provider
    space_1.save()
    return space_1


@pytest.mark.django_db
@patch('cookbook.helper.ai_helper.completion')
def test_food_properties_success_uses_central_completion(mock_completion, ai_space, a1_s1):
    with scopes_disabled():
        food = FoodFactory.create(space=ai_space)
    PropertyType.objects.create(name='Calories', space=ai_space)
    mock_completion.return_value = completion_response({
        'name': food.name,
        'properties': [],
    })

    response = a1_s1.post(
        f'{reverse("api:food-aiproperties", kwargs={"pk": food.pk})}?provider={ai_space.ai_provider.pk}',
        json.dumps({
            'name': food.name,
        }),
        content_type='application/json',
    )

    assert response.status_code == 200
    assert json.loads(response.content)['properties'] == []
    assert mock_completion.call_args.kwargs['success_callback'][0].function == AiLog.F_FOOD_PROPERTIES
    assert mock_completion.call_args.kwargs['max_tokens'] == AI_OUTPUT_TOKEN_BUDGETS[AiLog.F_FOOD_PROPERTIES]


@pytest.mark.django_db
@patch('cookbook.helper.ai_helper.completion')
def test_recipe_properties_success_uses_central_completion(mock_completion, ai_space, a1_s1):
    with scopes_disabled():
        recipe = RecipeFactory.create(space=ai_space, keywords__count=0)
    PropertyType.objects.create(name='Calories', space=ai_space)
    mock_completion.return_value = completion_response({
        'name': recipe.name,
        'properties': [],
    })

    response = a1_s1.post(
        f'{reverse("api:recipe-aiproperties", kwargs={"pk": recipe.pk})}?provider={ai_space.ai_provider.pk}',
        json.dumps({
            'name': recipe.name,
        }),
        content_type='application/json',
    )

    assert response.status_code == 200
    assert json.loads(response.content)['properties'] == []
    assert mock_completion.call_args.kwargs['success_callback'][0].function == AiLog.F_RECIPE_PROPERTIES
    assert mock_completion.call_args.kwargs['max_tokens'] == AI_OUTPUT_TOKEN_BUDGETS[AiLog.F_RECIPE_PROPERTIES]


@pytest.mark.django_db
@pytest.mark.parametrize('model_content', [
    schema_recipe(),
    json.dumps(schema_recipe()),
])
@patch('cookbook.helper.ai_helper.completion')
def test_text_recipe_import_success_uses_central_completion(mock_completion, model_content, ai_space, a1_s1):
    mock_completion.return_value = completion_response(model_content)

    response = a1_s1.post(
        reverse('api_ai_import'),
        {
            'ai_provider_id': ai_space.ai_provider.pk,
            'text': 'AI Soup\n1 cup water\nBoil the water.',
            'recipe_id': '',
            'file': '',
        },
    )

    assert response.status_code == 200
    result = json.loads(response.content)
    assert result['recipe']['name'] == 'AI Soup'
    messages = mock_completion.call_args.kwargs['messages']
    assert messages[0]['content'][1]['text'].startswith('AI Soup')
    assert mock_completion.call_args.kwargs['success_callback'][0].function == AiLog.F_FILE_IMPORT
    assert mock_completion.call_args.kwargs['max_tokens'] == AI_OUTPUT_TOKEN_BUDGETS[AiLog.F_FILE_IMPORT]
    assert mock_completion.call_args.kwargs['response_format'] == {
        'type': 'json_object',
    }


@pytest.mark.django_db
@pytest.mark.parametrize('image_mode', ['RGB', 'RGBA', 'P'])
@patch('cookbook.helper.ai_helper.completion')
def test_image_recipe_import_preserves_multimodal_message(mock_completion, image_mode, ai_space, a1_s1):
    image = Image.new(image_mode, (1, 1))
    buffer = BytesIO()
    image.save(buffer, format='PNG')
    test_file = SimpleUploadedFile('recipe.png', buffer.getvalue(), content_type='image/png')
    mock_completion.return_value = completion_response(schema_recipe())

    response = a1_s1.post(
        reverse('api_ai_import'),
        {
            'ai_provider_id': ai_space.ai_provider.pk,
            'text': '',
            'recipe_id': '',
            'file': test_file,
        },
    )

    assert response.status_code == 200
    content = mock_completion.call_args.kwargs['messages'][0]['content']
    assert content[0]['type'] == 'text'
    assert content[1]['type'] == 'image_url'
    assert content[1]['image_url'].startswith('data:image/png;base64,')
    assert mock_completion.call_args.kwargs['max_tokens'] == AI_OUTPUT_TOKEN_BUDGETS[AiLog.F_FILE_IMPORT]


@pytest.mark.django_db
@patch('cookbook.helper.ai_helper.completion')
def test_cloudflare_recipe_import_receives_schema_and_gemma_settings(mock_completion, ai_space, a1_s1):
    ai_space.ai_provider.model_name = CLOUDFLARE_GEMMA_MODEL
    ai_space.ai_provider.save()
    mock_completion.return_value = completion_response(schema_recipe())

    response = a1_s1.post(
        reverse('api_ai_import'),
        {
            'ai_provider_id': ai_space.ai_provider.pk,
            'text': 'AI Soup\n1 cup water\nBoil the water.',
            'recipe_id': '',
            'file': '',
        },
    )

    assert response.status_code == 200
    request = mock_completion.call_args.kwargs
    assert request['response_format'] == {
        'type': 'json_schema',
        'json_schema': AI_RECIPE_IMPORT_RESPONSE_SCHEMA,
    }
    assert request['extra_body'] == {
        'chat_template_kwargs': {
            'enable_thinking': False,
        },
    }
    assert 'reasoning_effort' not in request


@pytest.mark.django_db
@pytest.mark.parametrize(
    'invalid_result', [
        {
            'recipeIngredient': ['1 cup water'],
            'recipeInstructions': ['Boil.'],
        },
        {
            'name': 'Soup',
            'recipeIngredient': '1 cup water',
            'recipeInstructions': ['Boil.'],
        },
        {
            'name': 'Soup',
            'recipeIngredient': [7],
            'recipeInstructions': ['Boil.'],
        },
        {
            'name': 'Soup',
            'recipeIngredient': ['1 cup water'],
            'recipeInstructions': ['Boil.'],
            'unexpected': 'private recipe contents',
        },
    ]
)
@patch('cookbook.helper.ai_helper.completion')
def test_recipe_import_rejects_invalid_contract_without_leaking_content(mock_completion, invalid_result, ai_space, a1_s1):
    mock_completion.return_value = completion_response(invalid_result)

    response = a1_s1.post(
        reverse('api_ai_import'),
        {
            'ai_provider_id': ai_space.ai_provider.pk,
            'text': 'private source recipe',
            'recipe_id': '',
            'file': '',
        },
    )

    assert response.status_code == 400
    response_text = response.content.decode()
    assert 'malformed structured output' in response_text
    assert 'private source recipe' not in response_text
    assert 'private recipe contents' not in response_text
    assert 'test-key' not in response_text
