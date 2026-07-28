import json
from copy import deepcopy

import pytest

from cookbook.helper.ai_helper import AI_OUTPUT_TOKEN_BUDGETS
from cookbook.helper.ai_step_sort import AiStepTransformationError, _validate_plan, build_step_sort_payload, reconstruct_step_sorted_recipe
from cookbook.models import AiLog


def recipe_data():
    return {
        'id': 41,
        'name': 'Compact test',
        'description': 'Must survive unchanged',
        'steps': [{
            'id': 11,
            'name': 'Combined',
            'instruction': 'Mix the flour. Add the butter.',
            'ingredients': [
                {
                    'id': 21,
                    'amount': '2.0000000000000000',
                    'unit': {
                        'id': 31,
                        'name': 'cups',
                    },
                    'food': {
                        'id': 41,
                        'name': 'Flour',
                        'properties': [{
                            'very': 'large',
                        }],
                        'shopping_lists': [{
                            'id': 51,
                        }],
                        'fdc_id': 123,
                    },
                    'note': '',
                    'order': 5,
                },
                {
                    'amount': '1.0000000000000000',
                    'unit': None,
                    'food': {
                        'name': 'Butter',
                    },
                    'note': 'cold',
                    'order': 9,
                },
            ],
            'instructions_markdown': '<p>old</p>',
            'time': 15,
            'order': 4,
            'show_as_header': False,
            'file': {
                'id': 70,
            },
            'step_recipe': 80,
            'step_recipe_data': {
                'id': 80,
            },
            'numrecipe': 1,
            'show_ingredients_table': False,
        }],
        'nutrition': {
            'calories': 100,
        },
    }


def split_plan():
    return {
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
                'instruction': 'Add the butter.',
                'ingredient_keys': ['i1'],
            },
        ],
    }


def test_compact_payload_excludes_nested_food_metadata():
    original = recipe_data()
    payload, _ = build_step_sort_payload(original)

    assert payload == {
        'steps': [{
            'step_key': 's0',
            'name': 'Combined',
            'instruction': 'Mix the flour. Add the butter.',
            'ingredients': [
                {
                    'ingredient_key': 'i0',
                    'display_text': '2.0000000000000000 cups Flour',
                    'food_name': 'Flour',
                    'amount': '2.0000000000000000',
                    'unit_name': 'cups',
                    'note': '',
                },
                {
                    'ingredient_key': 'i1',
                    'display_text': '1.0000000000000000 Butter cold',
                    'food_name': 'Butter',
                    'amount': '1.0000000000000000',
                    'unit_name': '',
                    'note': 'cold',
                },
            ],
        }],
    }
    assert 'properties' not in str(payload)
    assert 'shopping_lists' not in str(payload)
    assert 'fdc_id' not in str(payload)


def test_large_compact_plan_has_room_beyond_cloudflare_default():
    plan = {
        'steps': [{
            'source_step_key': f's{index}',
            'name': f'Step {index}',
            'instruction': f'Perform detailed recipe instruction {index}.',
            'ingredient_keys': [f'i{index}'],
        } for index in range(40)],
    }

    assert len(json.dumps(plan)) > 256 * 4
    assert AI_OUTPUT_TOKEN_BUDGETS[AiLog.F_STEP_SORT] == 4096
    assert AI_OUTPUT_TOKEN_BUDGETS[AiLog.F_STEP_SORT] > 256


def test_unsaved_and_repeated_ids_get_unique_occurrence_keys():
    original = recipe_data()
    shared_ingredient = deepcopy(original['steps'][0]['ingredients'][0])
    original['steps'].append({
        **deepcopy(original['steps'][0]),
        'id': None,
        'ingredients': [shared_ingredient, deepcopy(shared_ingredient)],
    })

    payload, _ = build_step_sort_payload(original)

    assert [ingredient['ingredient_key'] for step in payload['steps'] for ingredient in step['ingredients']] == [
        'i0',
        'i1',
        'i2',
        'i3',
    ]


def test_ingredients_can_move_between_source_steps():
    original = recipe_data()
    original['steps'].append({
        **deepcopy(original['steps'][0]),
        'id': 12,
        'instruction': 'Bake the mixture.',
        'ingredients': [],
    })
    _, context = build_step_sort_payload(original)
    plan = {
        'steps': [
            {
                'source_step_key': 's0',
                'name': 'Mix',
                'instruction': 'Mix the flour.',
                'ingredient_keys': ['i0'],
            },
            {
                'source_step_key': 's1',
                'name': 'Bake',
                'instruction': 'Add the butter and bake.',
                'ingredient_keys': ['i1'],
            },
        ],
    }

    assert _validate_plan(plan, context)['steps'][1]['ingredient_keys'] == ['i1']


def test_nonblank_source_instruction_cannot_be_erased():
    _, context = build_step_sort_payload(recipe_data())
    plan = split_plan()
    for step in plan['steps']:
        step['instruction'] = '   '

    with pytest.raises(AiStepTransformationError, match='erased'):
        _validate_plan(plan, context)


def test_blank_source_header_can_remain_blank():
    original = recipe_data()
    original['steps'][0]['instruction'] = '   '
    _, context = build_step_sort_payload(original)
    plan = split_plan()
    for step in plan['steps']:
        step['instruction'] = ''

    assert _validate_plan(plan, context)['steps'][0]['instruction'] == ''


@pytest.mark.parametrize(
    'mutate_plan', [
        lambda plan: plan['steps'][0].update(source_step_key='unknown'),
        lambda plan: plan['steps'][0]['ingredient_keys'].append('unknown'),
        lambda plan: plan['steps'][1]['ingredient_keys'].append('i0'),
        lambda plan: plan['steps'][1].update(ingredient_keys=[]),
        lambda plan: plan['steps'][0].update(fabricated=True),
    ]
)
def test_invalid_or_incomplete_plans_are_rejected(mutate_plan):
    _, context = build_step_sort_payload(recipe_data())
    plan = split_plan()
    mutate_plan(plan)

    with pytest.raises(AiStepTransformationError):
        # Validation occurs before the request-dependent recipe serializer.
        reconstruct_step_sorted_recipe(plan, context, request=None)
