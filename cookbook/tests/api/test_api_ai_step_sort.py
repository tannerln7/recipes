import json
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from django.urls import reverse
from django_scopes import scopes_disabled

from cookbook.helper.ai_helper import AI_OUTPUT_TOKEN_BUDGETS
from cookbook.models import AiLog, AiProvider
from cookbook.tests.factories import IngredientFactory, RecipeFactory, StepFactory


def completion_response(content):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))], )


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


@pytest.fixture
def editor_recipe(ai_space, a1_s1):
    recipe = RecipeFactory.create(
        space=ai_space,
        name='Compact transformation',
        description='Metadata must stay unchanged',
        steps__count=0,
        keywords__count=0,
    )
    step = StepFactory.create(
        space=ai_space,
        name='Combined',
        instruction='Mix the flour. Cut in the butter.',
        time=15,
        order=7,
        show_as_header=False,
        show_ingredients_table=False,
        ingredients__count=0,
    )
    step.ingredients.add(
        IngredientFactory.create(
            space=ai_space,
            food__name='Flour',
            unit__name='cups',
            amount=2,
            note='dry',
            order=4,
        ),
        IngredientFactory.create(
            space=ai_space,
            food__name='Butter',
            unit=None,
            amount=1,
            note='cold',
            order=9,
        ),
    )
    recipe.steps.add(step)

    response = a1_s1.get(reverse('api:recipe-detail', kwargs={'pk': recipe.pk}))
    assert response.status_code == 200
    return json.loads(response.content)


@pytest.mark.django_db
@patch('cookbook.helper.ai_helper.completion')
def test_step_sort_sends_compact_plan_and_reconstructs_full_recipe(
    mock_completion,
    editor_recipe,
    ai_space,
    a1_s1,
):
    plan = {
        'steps': [
            {
                'source_step_key': 's0',
                'name': 'Mix',
                'instruction': 'Mix the flour.',
                'ingredient_keys': ['i0'],
            },
            {
                'source_step_key': 's0',
                'name': 'Finish',
                'instruction': 'Cut in the butter.',
                'ingredient_keys': ['i1'],
            },
        ],
    }
    mock_completion.return_value = completion_response(plan)

    response = a1_s1.post(
        f'{reverse("api_ai_step_sort")}?provider={ai_space.ai_provider.pk}',
        json.dumps(editor_recipe),
        content_type='application/json',
    )

    assert response.status_code == 200
    result = json.loads(response.content)
    compact_payload = json.loads(mock_completion.call_args.kwargs['messages'][0]['content'][1]['text'])
    assert mock_completion.call_args.kwargs['max_tokens'] == AI_OUTPUT_TOKEN_BUDGETS[AiLog.F_STEP_SORT]

    assert set(compact_payload) == {'steps'}
    assert set(compact_payload['steps'][0]) == {
        'step_key',
        'name',
        'instruction',
        'ingredients',
    }
    assert set(compact_payload['steps'][0]['ingredients'][0]) == {
        'ingredient_key',
        'display_text',
        'food_name',
        'amount',
        'unit_name',
        'note',
    }
    assert len(json.dumps(compact_payload)) < len(json.dumps(editor_recipe)) / 2
    for excluded in ['properties', 'shopping_lists', 'fdc_id', 'open_data_slug', 'substitute']:
        assert excluded not in json.dumps(compact_payload)

    expected_metadata = deepcopy(editor_recipe)
    expected_steps = expected_metadata.pop('steps')
    actual_metadata = deepcopy(result)
    actual_steps = actual_metadata.pop('steps')
    assert actual_metadata == expected_metadata

    assert actual_steps[0]['id'] == expected_steps[0]['id']
    assert 'id' not in actual_steps[1]
    assert [step['order'] for step in actual_steps] == [0, 1]
    assert actual_steps[0]['time'] == expected_steps[0]['time']
    assert actual_steps[1]['time'] == 0
    assert actual_steps[0]['show_as_header'] is False
    assert actual_steps[1]['show_as_header'] is False
    assert actual_steps[0]['show_ingredients_table'] is False
    assert actual_steps[1]['show_ingredients_table'] is False
    assert actual_steps[0]['ingredients'] == [expected_steps[0]['ingredients'][0]]
    assert actual_steps[1]['ingredients'] == [expected_steps[0]['ingredients'][1]]
    assert actual_steps[0]['instructions_markdown'] == '<p>Mix the flour.</p>'
    assert actual_steps[1]['instructions_markdown'] == '<p>Cut in the butter.</p>'
    assert actual_steps[1]['file'] is None
    assert actual_steps[1]['step_recipe'] is None
    assert actual_steps[1]['step_recipe_data'] is None
    assert actual_steps[1]['numrecipe'] == 0

    save_response = a1_s1.patch(
        reverse('api:recipe-detail', kwargs={'pk': result['id']}),
        json.dumps(result),
        content_type='application/json',
    )
    assert save_response.status_code == 200
    saved_recipe = json.loads(save_response.content)
    assert len(saved_recipe['steps']) == 2
    assert saved_recipe['steps'][0]['id'] == expected_steps[0]['id']
    assert saved_recipe['steps'][1]['id'] is not None
    expected_first_ingredient = {key: value for key, value in expected_steps[0]['ingredients'][0].items() if key != 'checked'}
    expected_second_ingredient = {key: value for key, value in expected_steps[0]['ingredients'][1].items() if key != 'checked'}
    assert saved_recipe['steps'][0]['ingredients'][0] == expected_first_ingredient
    assert saved_recipe['steps'][1]['ingredients'][0] == expected_second_ingredient


@pytest.mark.django_db
@patch('cookbook.helper.ai_helper.completion')
def test_step_sort_sanitizes_rendered_model_instruction(
    mock_completion,
    editor_recipe,
    ai_space,
    a1_s1,
):
    hostile_instruction = ('<script>alert("x")</script>'
                           '<img src="x" onerror="alert(1)">'
                           '{{ constructor.constructor("alert(2)")() }}')
    mock_completion.return_value = completion_response({
        'steps': [{
            'source_step_key': 's0',
            'name': 'Unsafe',
            'instruction': hostile_instruction,
            'ingredient_keys': ['i0', 'i1'],
        }],
    })

    response = a1_s1.post(
        f'{reverse("api_ai_step_sort")}?provider={ai_space.ai_provider.pk}',
        json.dumps(editor_recipe),
        content_type='application/json',
    )

    assert response.status_code == 200
    result = json.loads(response.content)
    assert result['steps'][0]['instruction'] == hostile_instruction
    rendered_instruction = result['steps'][0]['instructions_markdown'].lower()
    assert '<script' not in rendered_instruction
    assert 'onerror' not in rendered_instruction
    assert '{{' not in rendered_instruction
    assert 'constructor.constructor' not in rendered_instruction


@pytest.mark.django_db
@patch('cookbook.helper.ai_helper.completion')
def test_step_sort_moves_scraper_layout_ingredients_to_later_steps(
    mock_completion,
    ai_space,
    a1_s1,
):
    recipe = RecipeFactory.create(space=ai_space, steps__count=0, keywords__count=0)
    first_step = StepFactory.create(
        space=ai_space,
        instruction='Prepare the batter.',
        ingredients__count=0,
    )
    second_step = StepFactory.create(
        space=ai_space,
        instruction='Bake until golden.',
        ingredients__count=0,
    )
    with scopes_disabled():
        first_step.ingredients.add(
            IngredientFactory.create(space=ai_space, food__name='Flour', unit__name='cups', amount=2),
            IngredientFactory.create(space=ai_space, food__name='Butter', unit__name='tablespoons', amount=2),
        )
    recipe.steps.add(first_step, second_step)
    recipe_response = a1_s1.get(reverse('api:recipe-detail', kwargs={'pk': recipe.pk}))
    editor_recipe = json.loads(recipe_response.content)
    original_ingredients = deepcopy(editor_recipe['steps'][0]['ingredients'])
    mock_completion.return_value = completion_response({
        'steps': [
            {
                'source_step_key': 's0',
                'name': 'Prepare',
                'instruction': 'Prepare the batter with the flour.',
                'ingredient_keys': ['i0'],
            },
            {
                'source_step_key': 's1',
                'name': 'Bake',
                'instruction': 'Add the butter and bake until golden.',
                'ingredient_keys': ['i1'],
            },
        ],
    })

    response = a1_s1.post(
        f'{reverse("api_ai_step_sort")}?provider={ai_space.ai_provider.pk}',
        json.dumps(editor_recipe),
        content_type='application/json',
    )

    assert response.status_code == 200
    result = json.loads(response.content)
    assert result['steps'][0]['ingredients'] == [original_ingredients[0]]
    assert result['steps'][1]['ingredients'] == [original_ingredients[1]]
    model_prompt = mock_completion.call_args.kwargs['messages'][0]['content'][0]['text']
    assert 'may move to a step derived from any source step' in model_prompt

    save_response = a1_s1.patch(
        reverse('api:recipe-detail', kwargs={'pk': recipe.pk}),
        json.dumps(result),
        content_type='application/json',
    )
    assert save_response.status_code == 200
    saved_recipe = json.loads(save_response.content)
    assert saved_recipe['steps'][0]['ingredients'][0]['id'] == original_ingredients[0]['id']
    assert saved_recipe['steps'][1]['ingredients'][0]['id'] == original_ingredients[1]['id']


@pytest.mark.django_db
@pytest.mark.parametrize(
    'plan,error_fragment', [
        ({
            'steps': [{
                'source_step_key': 's0',
                'name': 'Invalid',
                'instruction': 'Invalid',
                'ingredient_keys': ['unknown'],
            }],
        }, 'unknown ingredient'),
        ({
            'steps': [{
                'source_step_key': 's0',
                'name': 'Invalid',
                'instruction': 'Invalid',
                'ingredient_keys': ['i0', 'i0', 'i1'],
            }],
        }, 'more than once'),
        ({
            'steps': [{
                'source_step_key': 's0',
                'name': 'Invalid',
                'instruction': 'Invalid',
                'ingredient_keys': ['i0'],
                'fabricated': 'field',
            }],
        }, 'required schema'),
    ]
)
@patch('cookbook.helper.ai_helper.completion')
def test_step_sort_rejects_invalid_plans(
    mock_completion,
    plan,
    error_fragment,
    editor_recipe,
    ai_space,
    a1_s1,
):
    mock_completion.return_value = completion_response(plan)

    response = a1_s1.post(
        f'{reverse("api_ai_step_sort")}?provider={ai_space.ai_provider.pk}',
        json.dumps(editor_recipe),
        content_type='application/json',
    )

    assert response.status_code == 400
    assert error_fragment in json.loads(response.content)['msg'].lower()
    assert 'test-key' not in response.content.decode()
    assert editor_recipe['description'] not in response.content.decode()


@pytest.mark.django_db
@patch('cookbook.helper.ai_helper.completion')
def test_shared_ingredient_id_gets_distinct_occurrence_keys(
    mock_completion,
    ai_space,
    a1_s1,
):
    recipe = RecipeFactory.create(space=ai_space, steps__count=0, keywords__count=0)
    with scopes_disabled():
        shared_ingredient = IngredientFactory.create(space=ai_space)
    first = StepFactory.create(space=ai_space, ingredients__count=0, instruction='Use it first.')
    second = StepFactory.create(space=ai_space, ingredients__count=0, instruction='Use it again.')
    first.ingredients.add(shared_ingredient)
    second.ingredients.add(shared_ingredient)
    recipe.steps.add(first, second)

    recipe_response = a1_s1.get(reverse('api:recipe-detail', kwargs={'pk': recipe.pk}))
    editor_recipe = json.loads(recipe_response.content)
    mock_completion.return_value = completion_response({
        'steps': [
            {
                'source_step_key': 's0',
                'name': '',
                'instruction': 'Use it first.',
                'ingredient_keys': ['i0'],
            },
            {
                'source_step_key': 's1',
                'name': '',
                'instruction': 'Use it again.',
                'ingredient_keys': ['i1'],
            },
        ],
    })

    response = a1_s1.post(
        f'{reverse("api_ai_step_sort")}?provider={ai_space.ai_provider.pk}',
        json.dumps(editor_recipe),
        content_type='application/json',
    )

    assert response.status_code == 200
    compact_payload = json.loads(mock_completion.call_args.kwargs['messages'][0]['content'][1]['text'])
    assert compact_payload['steps'][0]['ingredients'][0]['ingredient_key'] == 'i0'
    assert compact_payload['steps'][1]['ingredients'][0]['ingredient_key'] == 'i1'
    result = json.loads(response.content)
    assert result['steps'][0]['ingredients'][0]['id'] == shared_ingredient.id
    assert result['steps'][1]['ingredients'][0]['id'] == shared_ingredient.id
