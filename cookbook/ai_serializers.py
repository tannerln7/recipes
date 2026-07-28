from rest_framework import serializers

AI_RECIPE_IMPORT_REQUIRED_FIELDS = (
    'name',
    'recipeIngredient',
    'recipeInstructions',
)
AI_RECIPE_IMPORT_OPTIONAL_FIELDS = (
    'description',
    'recipeYield',
    'prepTime',
    'cookTime',
    'totalTime',
    'author',
    'datePublished',
    'keywords',
    'nutrition',
)
AI_RECIPE_IMPORT_ALLOWED_FIELDS = AI_RECIPE_IMPORT_REQUIRED_FIELDS + AI_RECIPE_IMPORT_OPTIONAL_FIELDS
AI_RECIPE_IMPORT_DURATION_PATTERN = r'^(?:$|P(?=.+)(?:\d+Y)?(?:\d+M)?(?:\d+W)?(?:\d+D)?(?:T(?=\d)(?:\d+H)?(?:\d+M)?(?:\d+(?:\.\d+)?S)?)?)$'


class StrictSerializer(serializers.Serializer):

    def to_internal_value(self, data):
        if not isinstance(data, dict):
            raise serializers.ValidationError('Expected a JSON object.')

        unexpected_fields = sorted(set(data) - set(self.fields))
        if unexpected_fields:
            raise serializers.ValidationError({
                'unexpected_fields': f'Unexpected field(s): {", ".join(unexpected_fields)}',
            })
        return super().to_internal_value(data)


class StrictStringField(serializers.CharField):

    def to_internal_value(self, data):
        if not isinstance(data, str):
            self.fail('invalid')
        return super().to_internal_value(data)


class StrictRegexField(serializers.RegexField):

    def to_internal_value(self, data):
        if not isinstance(data, str):
            self.fail('invalid')
        return super().to_internal_value(data)


class AiStepSortIngredientSerializer(StrictSerializer):
    ingredient_key = StrictStringField()
    display_text = StrictStringField(allow_blank=True)
    food_name = StrictStringField(allow_blank=True)
    amount = StrictStringField(allow_blank=True)
    unit_name = StrictStringField(allow_blank=True)
    note = StrictStringField(allow_blank=True)


class AiStepSortSourceStepSerializer(StrictSerializer):
    source_step_key = StrictStringField()
    name = StrictStringField(allow_blank=True)
    instruction = StrictStringField(allow_blank=True)


class AiStepSortRequestSerializer(StrictSerializer):
    source_steps = AiStepSortSourceStepSerializer(many=True)
    ingredients = AiStepSortIngredientSerializer(many=True)


class AiStepSortPlanStepSerializer(StrictSerializer):
    source_step_key = StrictStringField()
    name = StrictStringField(allow_blank=True)
    instruction = StrictStringField(allow_blank=True)
    ingredient_keys = serializers.ListField(child=StrictStringField())


class AiStepSortPlanSerializer(StrictSerializer):
    steps = AiStepSortPlanStepSerializer(many=True)


class AiRecipeImportNutritionSerializer(StrictSerializer):
    servingSize = StrictStringField(required=False, allow_blank=True)
    calories = StrictStringField(required=False, allow_blank=True)
    carbohydrateContent = StrictStringField(required=False, allow_blank=True)
    cholesterolContent = StrictStringField(required=False, allow_blank=True)
    fatContent = StrictStringField(required=False, allow_blank=True)
    fiberContent = StrictStringField(required=False, allow_blank=True)
    proteinContent = StrictStringField(required=False, allow_blank=True)
    saturatedFatContent = StrictStringField(required=False, allow_blank=True)
    sodiumContent = StrictStringField(required=False, allow_blank=True)
    sugarContent = StrictStringField(required=False, allow_blank=True)
    transFatContent = StrictStringField(required=False, allow_blank=True)
    unsaturatedFatContent = StrictStringField(required=False, allow_blank=True)


class AiRecipeImportResponseSerializer(StrictSerializer):
    name = StrictStringField(allow_blank=False, trim_whitespace=True)
    description = StrictStringField(required=False, allow_blank=True)
    recipeYield = StrictStringField(required=False, allow_blank=True)
    prepTime = StrictRegexField(AI_RECIPE_IMPORT_DURATION_PATTERN, required=False, allow_blank=True)
    cookTime = StrictRegexField(AI_RECIPE_IMPORT_DURATION_PATTERN, required=False, allow_blank=True)
    totalTime = StrictRegexField(AI_RECIPE_IMPORT_DURATION_PATTERN, required=False, allow_blank=True)
    author = StrictStringField(required=False, allow_blank=True)
    datePublished = StrictStringField(required=False, allow_blank=True)
    keywords = StrictStringField(required=False, allow_blank=True)
    nutrition = AiRecipeImportNutritionSerializer(required=False)
    recipeIngredient = serializers.ListField(child=StrictStringField(allow_blank=True))
    recipeInstructions = serializers.ListField(child=StrictStringField(allow_blank=True))


AI_STEP_SORT_RESPONSE_SCHEMA = {
    'type': 'object',
    'additionalProperties': False,
    'properties': {
        'steps': {
            'type': 'array',
            'items': {
                'type': 'object',
                'additionalProperties': False,
                'properties': {
                    'source_step_key': {
                        'type': 'string',
                    },
                    'name': {
                        'type': 'string',
                    },
                    'instruction': {
                        'type': 'string',
                    },
                    'ingredient_keys': {
                        'type': 'array',
                        'items': {
                            'type': 'string',
                        },
                    },
                },
                'required': [
                    'source_step_key',
                    'name',
                    'instruction',
                    'ingredient_keys',
                ],
            },
        },
    },
    'required': ['steps'],
}

AI_RECIPE_IMPORT_NUTRITION_PROPERTIES = {
    'servingSize': {
        'type': 'string',
    },
    'calories': {
        'type': 'string',
    },
    'carbohydrateContent': {
        'type': 'string',
    },
    'cholesterolContent': {
        'type': 'string',
    },
    'fatContent': {
        'type': 'string',
    },
    'fiberContent': {
        'type': 'string',
    },
    'proteinContent': {
        'type': 'string',
    },
    'saturatedFatContent': {
        'type': 'string',
    },
    'sodiumContent': {
        'type': 'string',
    },
    'sugarContent': {
        'type': 'string',
    },
    'transFatContent': {
        'type': 'string',
    },
    'unsaturatedFatContent': {
        'type': 'string',
    },
}

AI_RECIPE_IMPORT_RESPONSE_SCHEMA = {
    'type': 'object',
    'additionalProperties': False,
    'properties': {
        'name': {
            'type': 'string',
            'minLength': 1,
        },
        'description': {
            'type': 'string',
        },
        'recipeYield': {
            'type': 'string',
        },
        'prepTime': {
            'type': 'string',
            'description': 'ISO 8601 duration or an empty string.',
        },
        'cookTime': {
            'type': 'string',
            'description': 'ISO 8601 duration or an empty string.',
        },
        'totalTime': {
            'type': 'string',
            'description': 'ISO 8601 duration or an empty string.',
        },
        'author': {
            'type': 'string',
        },
        'datePublished': {
            'type': 'string',
        },
        'keywords': {
            'type': 'string',
        },
        'nutrition': {
            'type': 'object',
            'additionalProperties': False,
            'properties': AI_RECIPE_IMPORT_NUTRITION_PROPERTIES,
        },
        'recipeIngredient': {
            'type': 'array',
            'items': {
                'type': 'string',
            },
        },
        'recipeInstructions': {
            'type': 'array',
            'items': {
                'type': 'string',
            },
        },
    },
    'required': list(AI_RECIPE_IMPORT_REQUIRED_FIELDS),
}
