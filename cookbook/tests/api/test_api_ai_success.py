import json
from io import BytesIO
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from django.core.files.uploadedfile import SimpleUploadedFile
from django.urls import reverse
from django_scopes import scopes_disabled
from PIL import Image

from cookbook.helper.ai_helper import AI_OUTPUT_TOKEN_BUDGETS
from cookbook.models import AiLog, AiProvider, PropertyType
from cookbook.tests.factories import FoodFactory, RecipeFactory


def completion_response(content):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))], )


def schema_recipe():
    return {
        '@context': 'https://schema.org',
        '@type': 'Recipe',
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
@patch('cookbook.helper.ai_helper.completion')
def test_text_recipe_import_success_uses_central_completion(mock_completion, ai_space, a1_s1):
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
    result = json.loads(response.content)
    assert result['recipe']['name'] == 'AI Soup'
    messages = mock_completion.call_args.kwargs['messages']
    assert messages[0]['content'][1]['text'].startswith('AI Soup')
    assert mock_completion.call_args.kwargs['success_callback'][0].function == AiLog.F_FILE_IMPORT
    assert mock_completion.call_args.kwargs['max_tokens'] == AI_OUTPUT_TOKEN_BUDGETS[AiLog.F_FILE_IMPORT]


@pytest.mark.django_db
@patch('cookbook.helper.ai_helper.completion')
def test_image_recipe_import_preserves_multimodal_message(mock_completion, ai_space, a1_s1):
    image = Image.new('RGB', (1, 1), color='red')
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
    assert content[1]['image_url'].startswith('data:image/PNG;base64,')
    assert mock_completion.call_args.kwargs['max_tokens'] == AI_OUTPUT_TOKEN_BUDGETS[AiLog.F_FILE_IMPORT]
