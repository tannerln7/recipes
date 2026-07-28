from rest_framework import serializers


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


class AiStepSortIngredientSerializer(StrictSerializer):
    ingredient_key = serializers.CharField()
    display_text = serializers.CharField(allow_blank=True)
    food_name = serializers.CharField(allow_blank=True)
    amount = serializers.CharField(allow_blank=True)
    unit_name = serializers.CharField(allow_blank=True)
    note = serializers.CharField(allow_blank=True)


class AiStepSortSourceStepSerializer(StrictSerializer):
    step_key = serializers.CharField()
    name = serializers.CharField(allow_blank=True)
    instruction = serializers.CharField(allow_blank=True)
    ingredients = AiStepSortIngredientSerializer(many=True)


class AiStepSortRequestSerializer(StrictSerializer):
    steps = AiStepSortSourceStepSerializer(many=True)


class AiStepSortPlanStepSerializer(StrictSerializer):
    source_step_key = serializers.CharField()
    name = serializers.CharField(allow_blank=True)
    instruction = serializers.CharField(allow_blank=True)
    ingredient_keys = serializers.ListField(child=serializers.CharField())


class AiStepSortPlanSerializer(StrictSerializer):
    steps = AiStepSortPlanStepSerializer(many=True)


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
