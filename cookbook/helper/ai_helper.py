import json
import logging
import os
import re
from decimal import Decimal
from urllib.parse import quote

import httpx
from django.conf import settings
from django.db.models import Q, Sum
from django.utils import timezone
from litellm import CustomLogger, completion, cost_per_token
from litellm.exceptions import AuthenticationError, BadRequestError
from litellm.exceptions import Timeout as LitellmTimeout
from litellm.exceptions import UnsupportedParamsError

from cookbook.models import AiLog, AiProvider

logger = logging.getLogger(__name__)


class AiIntegrationError(Exception):
    status_code = 500
    client_message = 'An unexpected error occurred while processing your AI request.'

    def __init__(self, client_message=None):
        if client_message:
            self.client_message = client_message
        super().__init__(self.client_message)

    def as_response(self):
        return {'error': True, 'msg': self.client_message}


class AiProviderUnavailableError(AiIntegrationError):
    status_code = 400
    client_message = 'The selected AI provider is unavailable or inaccessible.'


class AiProviderUrlNotAllowedError(AiIntegrationError):
    status_code = 400
    client_message = 'The selected AI provider URL is not allowed.'


class AiProviderConfigurationError(AiIntegrationError):
    status_code = 400
    client_message = 'The selected AI provider is missing required application configuration.'


class AiAuthenticationError(AiIntegrationError):
    status_code = 400
    client_message = 'The AI provider rejected its configured credentials.'


class AiUnsupportedParameterError(AiIntegrationError):
    status_code = 400
    client_message = 'The selected AI provider does not support the requested structured output.'


class AiRequestTimeoutError(AiIntegrationError):
    status_code = 408
    client_message = 'The AI request timed out. Please try again later.'


class AiProviderRequestError(AiIntegrationError):
    status_code = 400
    client_message = 'The AI provider could not process this request.'


class AiStructuredOutputError(AiIntegrationError):
    status_code = 400
    client_message = 'The AI provider returned malformed structured output.'


_JSON_FENCE_RE = re.compile(r'\A\s*```(?:json)?\s*(.*?)\s*```\s*\Z', re.IGNORECASE | re.DOTALL)

# Cloudflare's recommended vision model defaults to only 256 output tokens.
# These bounded, operation-specific budgets leave room for realistic structured
# results while preventing unexpectedly large generations across providers.
AI_OUTPUT_TOKEN_BUDGETS = {
    AiLog.F_FOOD_PROPERTIES: 2048,
    AiLog.F_RECIPE_PROPERTIES: 4096,
    AiLog.F_FILE_IMPORT: 8192,
    AiLog.F_STEP_SORT: 4096,
}


def is_cloudflare_provider(model_name):
    """Return whether a LiteLLM model name selects the Cloudflare provider."""
    return isinstance(model_name, str) and model_name.startswith('cloudflare/')


def uses_native_cloudflare_transport(ai_provider):
    return is_cloudflare_provider(ai_provider.model_name) and not ai_provider.url


def get_ai_provider(provider_id, space):
    provider = AiProvider.objects.filter(pk=provider_id).filter(Q(space=space) | Q(space__isnull=True)).first()
    if provider is None:
        raise AiProviderUnavailableError()
    return provider


def get_structured_response_format(ai_provider, json_schema=None):
    if json_schema is not None and is_cloudflare_provider(ai_provider.model_name):
        return {
            'type': 'json_schema',
            'json_schema': json_schema,
        }
    return {'type': 'json_object'}


def build_ai_request(ai_provider, messages, callback=None, json_schema=None, max_output_tokens=None):
    if ai_provider is None:
        raise AiProviderUnavailableError()

    ai_request = {
        'api_key': ai_provider.api_key,
        'model': ai_provider.model_name,
        'response_format': get_structured_response_format(ai_provider, json_schema=json_schema),
        'messages': messages,
    }
    if max_output_tokens is not None:
        ai_request['max_tokens'] = max_output_tokens

    if ai_provider.url:
        if ai_provider.url not in settings.AI_ALLOWED_URLS:
            raise AiProviderUrlNotAllowedError()
        ai_request['api_base'] = ai_provider.url

    if is_cloudflare_provider(ai_provider.model_name):
        # LiteLLM releases before its OpenAI-compatible Cloudflare adapter require
        # this explicit allowlist. Keeping it request-scoped is harmless on newer
        # releases and documents the provider capability Tandoor relies on.
        ai_request['allowed_openai_params'] = ['response_format']

    if callback is not None:
        # LiteLLM supports dynamic callbacks on individual completion calls. This
        # avoids replacing its process-wide callback list for concurrent requests.
        ai_request['success_callback'] = [callback]
        ai_request['failure_callback'] = [callback]

    return ai_request


def build_cloudflare_native_payload(messages, response_format, max_output_tokens=None):
    native_messages = []
    image = None

    for message in messages:
        content = message.get('content')
        if not isinstance(content, list):
            native_messages.append({
                **message,
                'content': content,
            })
            continue

        text_parts = []
        for part in content:
            if part.get('type') == 'text':
                text_parts.append(str(part.get('text', '')))
                continue
            if part.get('type') == 'image_url':
                image_url = part.get('image_url')
                if isinstance(image_url, dict):
                    image_url = image_url.get('url')
                if image is not None or not isinstance(image_url, str):
                    raise AiUnsupportedParameterError('Cloudflare native requests support one image per AI request.')
                image = image_url
                continue
            raise AiUnsupportedParameterError('Cloudflare native requests received an unsupported message content type.')

        native_messages.append({
            **message,
            'content': '\n'.join(text_parts),
        })

    payload = {
        'messages': native_messages,
        'response_format': response_format,
    }
    if max_output_tokens is not None:
        payload['max_tokens'] = max_output_tokens
    if image is not None:
        payload['image'] = image
    return payload


def _cloudflare_native_url(ai_provider):
    account_id = os.environ.get('CLOUDFLARE_ACCOUNT_ID', '').strip()
    if not account_id:
        raise AiProviderConfigurationError()

    model_name = ai_provider.model_name.removeprefix('cloudflare/')
    return ('https://api.cloudflare.com/client/v4/accounts/'
            f'{quote(account_id, safe="")}/ai/run/{quote(model_name, safe="@/")}')


def _cloudflare_usage(response_data):
    result = response_data.get('result')
    result_usage = result.get('usage') if isinstance(result, dict) else None
    usage = result_usage or response_data.get('usage') or {}

    def token_count(*names):
        for name in names:
            try:
                return int(usage.get(name, 0))
            except (TypeError, ValueError):
                continue
        return 0

    return {
        'prompt_tokens': token_count('prompt_tokens', 'input_tokens'),
        'completion_tokens': token_count('completion_tokens', 'output_tokens'),
    }


def _native_response_cost(model_name, usage):
    try:
        input_cost, output_cost = cost_per_token(
            model=model_name,
            prompt_tokens=usage['prompt_tokens'],
            completion_tokens=usage['completion_tokens'],
        )
        return input_cost + output_cost
    except Exception:
        return 0


def _notify_native_callback(callback, success, model_name, usage, start_time, end_time):
    if callback is None:
        return

    try:
        callback_method = callback.log_success_event if success else callback.log_failure_event
        callback_method(
            {
                'response_cost': _native_response_cost(model_name, usage),
            },
            {
                'usage': usage,
            },
            start_time,
            end_time,
        )
    except Exception as err:
        logger.error(
            'Native Cloudflare callback failed',
            extra={
                'ai_model': model_name,
                'exception_type': type(err).__name__,
            },
        )


def complete_cloudflare_native(ai_provider, messages, callback=None, json_schema=None, max_output_tokens=None):
    response_format = get_structured_response_format(ai_provider, json_schema=json_schema)
    payload = build_cloudflare_native_payload(messages, response_format, max_output_tokens=max_output_tokens)
    url = _cloudflare_native_url(ai_provider)
    start_time = timezone.now()
    empty_usage = {
        'prompt_tokens': 0,
        'completion_tokens': 0,
    }

    try:
        response = httpx.post(
            url,
            headers={
                'Authorization': f'Bearer {ai_provider.api_key}',
                'Content-Type': 'application/json',
            },
            json=payload,
            timeout=120,
        )
    except httpx.TimeoutException:
        end_time = timezone.now()
        _notify_native_callback(callback, False, ai_provider.model_name, empty_usage, start_time, end_time)
        raise AiRequestTimeoutError() from None
    except httpx.RequestError:
        end_time = timezone.now()
        _notify_native_callback(callback, False, ai_provider.model_name, empty_usage, start_time, end_time)
        raise AiProviderRequestError() from None

    end_time = timezone.now()
    if response.status_code in (401, 403):
        _notify_native_callback(callback, False, ai_provider.model_name, empty_usage, start_time, end_time)
        raise AiAuthenticationError()
    if response.is_error:
        _notify_native_callback(callback, False, ai_provider.model_name, empty_usage, start_time, end_time)
        raise AiProviderRequestError()

    try:
        response_data = response.json()
    except ValueError:
        _notify_native_callback(callback, False, ai_provider.model_name, empty_usage, start_time, end_time)
        raise AiStructuredOutputError() from None

    if not isinstance(response_data, dict) or response_data.get('success') is False:
        _notify_native_callback(callback, False, ai_provider.model_name, empty_usage, start_time, end_time)
        raise AiProviderRequestError()

    result = response_data.get('result')
    if not isinstance(result, dict) or 'response' not in result:
        _notify_native_callback(callback, False, ai_provider.model_name, empty_usage, start_time, end_time)
        raise AiStructuredOutputError()

    usage = _cloudflare_usage(response_data)
    _notify_native_callback(callback, True, ai_provider.model_name, usage, start_time, end_time)
    return result['response']


def parse_structured_content(content, allow_list=False):
    if isinstance(content, dict):
        return content
    if isinstance(content, list):
        if allow_list:
            return content
        raise AiStructuredOutputError('The AI provider returned a JSON list where an object was required.')
    if not isinstance(content, str):
        raise AiStructuredOutputError('The AI provider returned an unsupported structured-output value.')

    match = _JSON_FENCE_RE.fullmatch(content)
    if match:
        content = match.group(1)

    try:
        parsed = json.loads(content)
    except (json.JSONDecodeError, TypeError):
        raise AiStructuredOutputError() from None

    if isinstance(parsed, dict):
        return parsed
    if isinstance(parsed, list) and allow_list:
        return parsed
    if isinstance(parsed, list):
        raise AiStructuredOutputError('The AI provider returned a JSON list where an object was required.')
    raise AiStructuredOutputError('The AI provider returned JSON with an unsupported top-level type.')


def complete_structured(ai_provider, messages, space, user, function, json_schema=None, allow_list=False, max_output_tokens=None):
    callback = AiCallbackHandler(space, user, ai_provider, function)

    try:
        if uses_native_cloudflare_transport(ai_provider):
            content = complete_cloudflare_native(
                ai_provider,
                messages,
                callback=callback,
                json_schema=json_schema,
                max_output_tokens=max_output_tokens,
            )
        else:
            ai_request = build_ai_request(
                ai_provider,
                messages,
                callback=callback,
                json_schema=json_schema,
                max_output_tokens=max_output_tokens,
            )
            ai_response = completion(**ai_request)
            content = ai_response.choices[0].message.content
        return parse_structured_content(content, allow_list=allow_list)
    except LitellmTimeout:
        raise AiRequestTimeoutError() from None
    except AuthenticationError:
        raise AiAuthenticationError() from None
    except UnsupportedParamsError:
        raise AiUnsupportedParameterError() from None
    except BadRequestError:
        raise AiProviderRequestError() from None
    except AiIntegrationError:
        raise
    except Exception as err:
        logger.error(
            'Unexpected AI completion failure',
            extra={
                'ai_model': getattr(ai_provider, 'model_name', None),
                'ai_function': function,
                'exception_type': type(err).__name__,
            },
        )
        raise AiIntegrationError() from None


def get_monthly_token_usage(space):
    """
    returns the number of credits the space has used in the current month
    """
    token_usage = AiLog.objects.filter(space=space, credits_from_balance=False, created_at__month=timezone.now().month).aggregate(Sum('credit_cost'))['credit_cost__sum']
    if token_usage is None:
        token_usage = 0
    return token_usage


def has_monthly_token(space):
    """
    checks if the monthly credit limit has been exceeded
    """
    return get_monthly_token_usage(space) < space.ai_credits_monthly


def can_perform_ai_request(space):
    return (has_monthly_token(space) or space.ai_credits_balance > 0) and space.ai_enabled


class AiCallbackHandler(CustomLogger):
    space = None
    user = None
    ai_provider = None
    function = None

    def __init__(self, space, user, ai_provider, function):
        super().__init__()
        self.space = space
        self.user = user
        self.ai_provider = ai_provider
        self.function = function

    def log_pre_api_call(self, model, messages, kwargs):
        pass

    def log_post_api_call(self, kwargs, response_obj, start_time, end_time):
        pass

    def log_success_event(self, kwargs, response_obj, start_time, end_time):
        self.create_ai_log(kwargs, response_obj, start_time, end_time)

    def log_failure_event(self, kwargs, response_obj, start_time, end_time):
        self.create_ai_log(kwargs, response_obj, start_time, end_time)

    def create_ai_log(self, kwargs, response_obj, start_time, end_time):
        credit_cost = 0
        credits_from_balance = False
        if self.ai_provider.log_credit_cost:
            credit_cost = kwargs.get("response_cost", 0) * 100

        if (not has_monthly_token(self.space)) and self.space.ai_credits_balance > 0:
            remaining_balance = self.space.ai_credits_balance - Decimal(str(credit_cost))
            if remaining_balance < 0:
                remaining_balance = 0
                if settings.HOSTED and self.space.ai_credits_monthly == 0:
                    self.space.ai_enabled = False

            self.space.ai_credits_balance = remaining_balance
            credits_from_balance = True
            self.space.save()

        AiLog.objects.create(
            created_by=self.user,
            space=self.space,
            ai_provider=self.ai_provider,
            start_time=start_time,
            end_time=end_time,
            input_tokens=response_obj['usage']['prompt_tokens'],
            output_tokens=response_obj['usage']['completion_tokens'],
            function=self.function,
            credit_cost=credit_cost,
            credits_from_balance=credits_from_balance,
        )
