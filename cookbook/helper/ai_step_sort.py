from copy import deepcopy
from dataclasses import dataclass

from cookbook.ai_serializers import AiStepSortPlanSerializer, AiStepSortRequestSerializer
from cookbook.helper.ai_helper import AiStructuredOutputError
from cookbook.helper.template_helper import IngredientObject, render_instruction_text
from cookbook.serializer import RecipeSerializer


class AiStepTransformationError(AiStructuredOutputError):
    client_message = 'The AI provider returned an invalid or incomplete step transformation.'


class AiRecipeReconstructionError(AiStructuredOutputError):
    client_message = 'The transformed recipe could not be validated.'


@dataclass
class StepSortContext:
    original_recipe: dict
    source_steps: dict
    ingredients: dict


def _string_value(value):
    return '' if value is None else str(value)


def _ingredient_display_text(ingredient):
    if ingredient.get('original_text'):
        return _string_value(ingredient['original_text'])

    food = ingredient.get('food') or {}
    unit = ingredient.get('unit') or {}
    return ' '.join(
        part for part in [
            _string_value(ingredient.get('amount')),
            _string_value(unit.get('name')),
            _string_value(food.get('name')),
            _string_value(ingredient.get('note')),
        ] if part
    )


def build_step_sort_payload(recipe_data):
    original_recipe = deepcopy(dict(recipe_data))
    source_steps = {}
    ingredients = {}
    compact_steps = []
    ingredient_index = 0

    for step_index, source_step in enumerate(original_recipe.get('steps', [])):
        step_key = f's{step_index}'
        source_steps[step_key] = source_step
        compact_ingredients = []

        for ingredient in source_step.get('ingredients', []):
            ingredient_key = f'i{ingredient_index}'
            ingredient_index += 1
            ingredients[ingredient_key] = ingredient

            food = ingredient.get('food') or {}
            unit = ingredient.get('unit') or {}
            compact_ingredients.append({
                'ingredient_key': ingredient_key,
                'display_text': _ingredient_display_text(ingredient),
                'food_name': _string_value(food.get('name')),
                'amount': _string_value(ingredient.get('amount')),
                'unit_name': _string_value(unit.get('name')),
                'note': _string_value(ingredient.get('note')),
            })

        compact_steps.append({
            'step_key': step_key,
            'name': _string_value(source_step.get('name')),
            'instruction': _string_value(source_step.get('instruction')),
            'ingredients': compact_ingredients,
        })

    payload = {'steps': compact_steps}
    serializer = AiStepSortRequestSerializer(data=payload)
    if not serializer.is_valid():
        raise AiStepTransformationError('The recipe could not be converted to the compact step-sort format.')

    return payload, StepSortContext(
        original_recipe=original_recipe,
        source_steps=source_steps,
        ingredients=ingredients,
    )


def _validate_plan(plan_data, context):
    serializer = AiStepSortPlanSerializer(data=plan_data)
    if not serializer.is_valid():
        raise AiStepTransformationError('The AI step transformation did not match the required schema.')
    plan = serializer.validated_data

    expected_source_order = list(context.source_steps)
    actual_source_order = []
    seen_sources = set()
    assigned_ingredients = set()
    source_has_nonblank_output = {step_key: False for step_key in expected_source_order}

    for output_step in plan['steps']:
        source_key = output_step['source_step_key']
        if source_key not in context.source_steps:
            raise AiStepTransformationError('The AI step transformation referenced an unknown source step.')

        if source_key not in seen_sources:
            actual_source_order.append(source_key)
            seen_sources.add(source_key)
        elif actual_source_order[-1] != source_key:
            raise AiStepTransformationError('The AI step transformation reordered or interleaved source steps.')

        if output_step['instruction'].strip():
            source_has_nonblank_output[source_key] = True

        for ingredient_key in output_step['ingredient_keys']:
            if ingredient_key not in context.ingredients:
                raise AiStepTransformationError('The AI step transformation referenced an unknown ingredient.')
            if ingredient_key in assigned_ingredients:
                raise AiStepTransformationError('The AI step transformation assigned an ingredient more than once.')
            assigned_ingredients.add(ingredient_key)

    if actual_source_order != expected_source_order:
        raise AiStepTransformationError('The AI step transformation omitted or reordered a source step.')

    if assigned_ingredients != set(context.ingredients):
        raise AiStepTransformationError('The AI step transformation omitted an ingredient.')

    for source_key, source_step in context.source_steps.items():
        if _string_value(source_step.get('instruction')).strip() and not source_has_nonblank_output[source_key]:
            raise AiStepTransformationError('The AI step transformation erased a nonblank source instruction.')

    return plan


def reconstruct_step_sorted_recipe(plan_data, context, request):
    plan = _validate_plan(plan_data, context)
    result = deepcopy(context.original_recipe)
    reconstructed_steps = []
    source_result_counts = {}

    for output_step in plan['steps']:
        source_key = output_step['source_step_key']
        source_result_index = source_result_counts.get(source_key, 0)
        source_result_counts[source_key] = source_result_index + 1
        reconstructed_step = deepcopy(context.source_steps[source_key])

        if source_result_index > 0:
            reconstructed_step.pop('id', None)
            reconstructed_step['time'] = 0
            reconstructed_step['file'] = None
            reconstructed_step['step_recipe'] = None
            reconstructed_step['step_recipe_data'] = None
            reconstructed_step['numrecipe'] = 0

        reconstructed_step['name'] = output_step['name']
        reconstructed_step['instruction'] = output_step['instruction']
        reconstructed_ingredients = [deepcopy(context.ingredients[ingredient_key]) for ingredient_key in output_step['ingredient_keys']]
        template_ingredients = [IngredientObject.from_dict(ingredient) for ingredient in reconstructed_ingredients]
        reconstructed_step['instructions_markdown'] = render_instruction_text(output_step['instruction'], template_ingredients)
        reconstructed_step['ingredients'] = reconstructed_ingredients
        reconstructed_step['order'] = len(reconstructed_steps)
        reconstructed_steps.append(reconstructed_step)

    result['steps'] = reconstructed_steps
    serializer = RecipeSerializer(
        data=deepcopy(result),
        partial=True,
        context={'request': request},
    )
    if not serializer.is_valid():
        raise AiRecipeReconstructionError()
    return result
