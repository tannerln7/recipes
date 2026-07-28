import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import httpx
import pytest
from django.test import override_settings
from litellm.exceptions import AuthenticationError, BadRequestError, Timeout, UnsupportedParamsError

from cookbook.helper.ai_helper import (
    AI_OUTPUT_TOKEN_BUDGETS, CLOUDFLARE_GEMMA_MODEL, CLOUDFLARE_LLAMA_VISION_MODEL, AiAuthenticationError, AiProviderRequestError, AiProviderUrlNotAllowedError,
    AiRequestTimeoutError, AiStructuredOutputError, AiUnsupportedParameterError, build_ai_request, build_cloudflare_native_payload, complete_cloudflare_native,
    complete_structured, parse_structured_content, uses_native_cloudflare_transport
)
from cookbook.models import AiLog


def provider(model_name='openai/test-model', url=''):
    return SimpleNamespace(
        api_key='secret-key',
        model_name=model_name,
        url=url,
        log_credit_cost=False,
    )


def completion_response(content):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))], )


def test_build_default_provider_request():
    request = build_ai_request(provider(), [{'role': 'user', 'content': 'test'}])

    assert request == {
        'api_key': 'secret-key',
        'model': 'openai/test-model',
        'response_format': {
            'type': 'json_object',
        },
        'messages': [{
            'role': 'user',
            'content': 'test',
        }],
    }


def test_build_provider_request_includes_explicit_output_budget():
    request = build_ai_request(
        provider(),
        [{
            'role': 'user',
            'content': 'test',
        }],
        max_output_tokens=1536,
    )

    assert request['max_tokens'] == 1536


def test_cloudflare_request_gets_schema_and_allowed_parameter():
    schema = {
        'type': 'object',
    }
    request = build_ai_request(
        provider('cloudflare/@cf/meta/llama-3.2-11b-vision-instruct'),
        [],
        json_schema=schema,
    )

    assert request['allowed_openai_params'] == ['response_format']
    assert 'extra_body' not in request
    assert 'reasoning_effort' not in request
    assert request['response_format'] == {
        'type': 'json_schema',
        'json_schema': schema,
    }


def test_gemma_request_disables_thinking_through_exact_extra_body():
    request = build_ai_request(provider(CLOUDFLARE_GEMMA_MODEL), [])

    assert request['extra_body'] == {
        'chat_template_kwargs': {
            'enable_thinking': False,
        },
    }
    assert request['allowed_openai_params'] == ['response_format']
    assert 'reasoning_effort' not in request


@pytest.mark.parametrize(
    'ai_provider,expected_native', [
        (provider(CLOUDFLARE_LLAMA_VISION_MODEL), True),
        (provider(CLOUDFLARE_GEMMA_MODEL), False),
        (provider(CLOUDFLARE_LLAMA_VISION_MODEL, url='https://allowed.example/v1'), False),
        (provider('openai/test-model'), False),
    ]
)
def test_cloudflare_transport_policy_is_model_and_url_aware(ai_provider, expected_native):
    assert uses_native_cloudflare_transport(ai_provider) is expected_native


def test_non_cloudflare_provider_does_not_get_cloudflare_override():
    request = build_ai_request(provider('anthropic/claude-test'), [], json_schema={'type': 'object'})

    assert 'allowed_openai_params' not in request
    assert 'extra_body' not in request
    assert 'reasoning_effort' not in request
    assert request['response_format'] == {
        'type': 'json_object',
    }


@override_settings(AI_ALLOWED_URLS=['https://allowed.example/v1'])
def test_allowed_api_base_is_forwarded():
    request = build_ai_request(provider(url='https://allowed.example/v1'), [])

    assert request['api_base'] == 'https://allowed.example/v1'


@override_settings(AI_ALLOWED_URLS=['https://allowed.example/v1'])
def test_disallowed_api_base_is_rejected_without_leaking_url():
    with pytest.raises(AiProviderUrlNotAllowedError) as exc_info:
        build_ai_request(provider(url='https://private.example/v1'), [])

    assert 'private.example' not in exc_info.value.client_message


@pytest.mark.parametrize('content', [
    {
        'ok': True,
    },
    json.dumps({
        'ok': True,
    }),
    '```json\n{"ok": true}\n```',
])
def test_structured_parser_accepts_objects(content):
    assert parse_structured_content(content) == {
        'ok': True,
    }


def test_structured_parser_accepts_lists_only_when_enabled():
    assert parse_structured_content([1, 2], allow_list=True) == [1, 2]
    assert parse_structured_content('[1, 2]', allow_list=True) == [1, 2]

    with pytest.raises(AiStructuredOutputError):
        parse_structured_content([1, 2])


@pytest.mark.parametrize('content', [
    'Here is the result: {"ok": true}',
    '{"ok": true} trailing prose',
    'not json',
    '"a scalar"',
    None,
])
def test_structured_parser_rejects_non_structured_content(content):
    with pytest.raises(AiStructuredOutputError):
        parse_structured_content(content)


def test_cloudflare_native_payload_preserves_inline_image_content():
    payload = build_cloudflare_native_payload(
        [{
            'role': 'user',
            'content': [
                {
                    'type': 'text',
                    'text': 'Read this image.',
                },
                {
                    'type': 'image_url',
                    'image_url': 'data:image/png;base64,AAAA',
                },
            ],
        }],
        {
            'type': 'json_object',
        },
    )

    assert payload == {
        'messages': [{
            'role': 'user',
            'content': [
                {
                    'type': 'text',
                    'text': 'Read this image.',
                },
                {
                    'type': 'image_url',
                    'image_url': {
                        'url': 'data:image/png;base64,AAAA',
                    },
                },
            ],
        }],
        'response_format': {
            'type': 'json_object',
        },
    }
    assert 'image' not in payload
    assert 'max_tokens' not in payload


def test_cloudflare_native_payload_keeps_text_only_content_as_text():
    payload = build_cloudflare_native_payload(
        [{
            'role': 'user',
            'content': [
                {
                    'type': 'text',
                    'text': 'First.',
                },
                {
                    'type': 'text',
                    'text': 'Second.',
                },
            ],
        }],
        {
            'type': 'json_object',
        },
        max_output_tokens=512,
    )

    assert payload['messages'][0]['content'] == 'First.\nSecond.'
    assert payload['max_tokens'] == 512
    assert 'image' not in payload


@pytest.mark.parametrize(
    'content', [
        ['invalid'],
        [{
            'type': 'text',
            'text': None,
        }],
        [{
            'type': 'image_url',
            'image_url': '',
        }],
        [{
            'type': 'image_url',
            'image_url': {},
        }],
        [{
            'type': 'unsupported',
        }],
    ]
)
def test_cloudflare_native_payload_rejects_malformed_content_parts(content):
    with pytest.raises(AiUnsupportedParameterError):
        build_cloudflare_native_payload(
            [{
                'role': 'user',
                'content': content,
            }],
            {
                'type': 'json_object',
            },
        )


@pytest.mark.parametrize('native_content', [
    {
        'ok': True,
    },
    '{"ok": true}',
])
@patch('cookbook.helper.ai_helper.AiCallbackHandler')
@patch('cookbook.helper.ai_helper.httpx.post')
def test_complete_structured_normalizes_cloudflare_native_response(
    mock_post,
    mock_callback_handler,
    native_content,
    monkeypatch,
):
    monkeypatch.setenv('CLOUDFLARE_ACCOUNT_ID', 'account-id')
    mock_callback_handler.return_value = MagicMock()
    mock_post.return_value = httpx.Response(
        200,
        json={
            'result': {
                'response': native_content,
            },
        },
        request=httpx.Request('POST', 'https://api.cloudflare.com'),
    )

    result = complete_structured(
        provider(CLOUDFLARE_LLAMA_VISION_MODEL),
        [{
            'role': 'user',
            'content': 'Return JSON.',
        }],
        SimpleNamespace(),
        SimpleNamespace(),
        17,
        max_output_tokens=1024,
    )

    assert result == {
        'ok': True,
    }
    request = mock_post.call_args
    assert request.args[0].endswith('/ai/run/@cf/meta/llama-3.2-11b-vision-instruct')
    assert request.kwargs['json']['response_format'] == {
        'type': 'json_object',
    }
    assert request.kwargs['json']['max_tokens'] == 1024
    assert request.kwargs['headers']['Authorization'] == 'Bearer secret-key'


@patch('cookbook.helper.ai_helper.httpx.post')
def test_cloudflare_native_callback_receives_usage(mock_post, monkeypatch):
    monkeypatch.setenv('CLOUDFLARE_ACCOUNT_ID', 'account-id')
    callback = MagicMock()
    mock_post.return_value = httpx.Response(
        200,
        json={
            'result': {
                'response': {
                    'ok': True,
                },
                'usage': {
                    'prompt_tokens': 12,
                    'completion_tokens': 4,
                },
            },
        },
        request=httpx.Request('POST', 'https://api.cloudflare.com'),
    )

    content = complete_cloudflare_native(
        provider(CLOUDFLARE_LLAMA_VISION_MODEL),
        [],
        callback=callback,
        json_schema={
            'type': 'object',
        },
    )

    assert content == {
        'ok': True,
    }
    callback.log_success_event.assert_called_once()
    callback_response = callback.log_success_event.call_args.args[1]
    assert callback_response['usage'] == {
        'prompt_tokens': 12,
        'completion_tokens': 4,
    }
    assert mock_post.call_args.kwargs['json']['response_format'] == {
        'type': 'json_schema',
        'json_schema': {
            'type': 'object',
        },
    }


@patch('cookbook.helper.ai_helper.httpx.post')
def test_cloudflare_native_timeout_is_classified(mock_post, monkeypatch):
    monkeypatch.setenv('CLOUDFLARE_ACCOUNT_ID', 'account-id')
    mock_post.side_effect = httpx.ReadTimeout(
        'private timeout',
        request=httpx.Request('POST', 'https://api.cloudflare.com'),
    )

    with pytest.raises(AiRequestTimeoutError):
        complete_cloudflare_native(
            provider(CLOUDFLARE_LLAMA_VISION_MODEL),
            [],
            callback=MagicMock(),
        )


@patch('cookbook.helper.ai_helper.completion')
def test_complete_structured_uses_request_scoped_callbacks(mock_completion):
    mock_completion.return_value = completion_response({
        'ok': True,
    })

    result = complete_structured(
        provider(),
        [{
            'role': 'user',
            'content': 'private prompt',
        }],
        SimpleNamespace(),
        SimpleNamespace(),
        17,
        max_output_tokens=1536,
    )

    assert result == {
        'ok': True,
    }
    request = mock_completion.call_args.kwargs
    assert request['success_callback'][0] is request['failure_callback'][0]
    assert request['success_callback'][0].function == 17
    assert request['success_callback'][0].ai_provider.model_name == 'openai/test-model'
    assert request['max_tokens'] == 1536


@patch('cookbook.helper.ai_helper.completion')
def test_complete_structured_parses_openai_compatible_string(mock_completion):
    mock_completion.return_value = completion_response('{"ok": true}')

    result = complete_structured(
        provider(),
        [],
        SimpleNamespace(),
        SimpleNamespace(),
        17,
    )

    assert result == {
        'ok': True,
    }
    assert 'max_tokens' not in mock_completion.call_args.kwargs


def test_operation_output_budgets_are_bounded_and_exceed_cloudflare_default_where_needed():
    assert AI_OUTPUT_TOKEN_BUDGETS == {
        AiLog.F_FOOD_PROPERTIES: 2048,
        AiLog.F_RECIPE_PROPERTIES: 4096,
        AiLog.F_FILE_IMPORT: 8192,
        AiLog.F_STEP_SORT: 4096,
    }
    assert all(256 < budget <= 8192 for budget in AI_OUTPUT_TOKEN_BUDGETS.values())


@pytest.mark.parametrize(
    'provider_error,application_error,status_code', [
        (
            Timeout('private timeout details', 'test-model', 'test'),
            AiRequestTimeoutError,
            408,
        ),
        (
            AuthenticationError('secret-key was rejected', 'test', 'test-model'),
            AiAuthenticationError,
            400,
        ),
        (
            UnsupportedParamsError('private unsupported details', 'test', 'test-model'),
            AiUnsupportedParameterError,
            400,
        ),
        (
            BadRequestError('private recipe content', 'test-model', 'test'),
            AiProviderRequestError,
            400,
        ),
    ]
)
@patch('cookbook.helper.ai_helper.completion')
def test_provider_errors_are_classified_without_private_details(
    mock_completion,
    provider_error,
    application_error,
    status_code,
):
    mock_completion.side_effect = provider_error

    with pytest.raises(application_error) as exc_info:
        complete_structured(
            provider(),
            [{
                'role': 'user',
                'content': 'private recipe content',
            }],
            SimpleNamespace(),
            SimpleNamespace(),
            17,
        )

    assert exc_info.value.status_code == status_code
    assert 'secret-key' not in exc_info.value.client_message
    assert 'private recipe content' not in exc_info.value.client_message
