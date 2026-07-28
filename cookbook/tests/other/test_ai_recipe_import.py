from io import BytesIO

import pytest
from PIL import Image

from cookbook.ai_serializers import AI_RECIPE_IMPORT_REQUIRED_FIELDS, AI_RECIPE_IMPORT_RESPONSE_SCHEMA
from cookbook.helper.ai_helper import AiStructuredOutputError
from cookbook.helper.ai_recipe_import import build_recipe_import_prompt, canonicalize_recipe_import_result, normalize_recipe_import_image, recipe_import_contract_fields


def canonical_recipe():
    return {
        'name': 'Soup',
        'description': 'A simple soup.',
        'recipeYield': '2 servings',
        'prepTime': 'PT10M',
        'cookTime': 'PT20M',
        'totalTime': 'PT30M',
        'recipeIngredient': ['1 cup water'],
        'recipeInstructions': ['Boil the water.'],
    }


def test_recipe_import_prompt_and_schema_share_one_contract():
    prompt = build_recipe_import_prompt('text')

    assert set(AI_RECIPE_IMPORT_RESPONSE_SCHEMA['properties']) == recipe_import_contract_fields()
    assert set(AI_RECIPE_IMPORT_RESPONSE_SCHEMA['required']) == set(AI_RECIPE_IMPORT_REQUIRED_FIELDS)
    for field_name in recipe_import_contract_fields():
        assert field_name in prompt
    assert '@context' not in AI_RECIPE_IMPORT_RESPONSE_SCHEMA['properties']
    assert '@type' not in AI_RECIPE_IMPORT_RESPONSE_SCHEMA['properties']
    assert 'Do not return @context, @type' in prompt


def test_canonical_recipe_import_adds_schema_org_constants():
    result = canonicalize_recipe_import_result(canonical_recipe())

    assert result['@context'] == 'https://schema.org'
    assert result['@type'] == 'Recipe'
    assert result['name'] == 'Soup'


@pytest.mark.parametrize(
    'mutate', [
        lambda data: data.pop('name'),
        lambda data: data.update(name='   '),
        lambda data: data.update(recipeIngredient='1 cup water'),
        lambda data: data.update(recipeInstructions='Boil the water.'),
        lambda data: data.update(recipeIngredient=[17]),
        lambda data: data.update(recipeInstructions=[{
            'text': 'Boil.',
        }]),
        lambda data: data.update(unexpected='private content'),
        lambda data: data.update(prepTime='10 minutes'),
        lambda data: data.update(prepTime=10),
    ]
)
def test_invalid_recipe_import_contract_is_rejected(mutate):
    data = canonical_recipe()
    mutate(data)

    with pytest.raises(AiStructuredOutputError):
        canonicalize_recipe_import_result(data)


def test_recipe_import_supports_optional_nutrition():
    data = canonical_recipe()
    data['nutrition'] = {
        'calories': '120 kcal',
        'proteinContent': '3 g',
    }

    result = canonicalize_recipe_import_result(data)

    assert result['nutrition'] == {
        '@type': 'NutritionInformation',
        'calories': '120 kcal',
        'proteinContent': '3 g',
    }


def test_known_recipe_import_aliases_are_normalized_before_strict_validation():
    result = canonicalize_recipe_import_result({
        'name': 'Soup',
        'servings': '2 servings',
        'prep_time': 'PT10M',
        'ingredients': ['1 cup water'],
        'instructions': ['Boil the water.'],
    })

    assert result['recipeYield'] == '2 servings'
    assert result['prepTime'] == 'PT10M'
    assert result['recipeIngredient'] == ['1 cup water']
    assert result['recipeInstructions'] == ['Boil the water.']
    assert 'servings' not in result
    assert 'ingredients' not in result


@pytest.mark.parametrize('mode', ['RGB', 'RGBA', 'P'])
def test_recipe_import_png_images_are_normalized_to_rgb_with_lowercase_mime(mode):
    image = Image.new(mode, (2, 2))
    source = BytesIO()
    image.save(source, format='PNG')
    source.seek(0)

    mime_type, image_bytes = normalize_recipe_import_image(source)

    assert mime_type == 'image/png'
    with Image.open(BytesIO(image_bytes)) as normalized:
        assert normalized.format == 'PNG'
        assert normalized.mode == 'RGB'
