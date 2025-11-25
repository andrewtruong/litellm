from __future__ import annotations

import base64
import json
import os
from typing import TYPE_CHECKING, Any, Union

from litellm._logging import verbose_logger
from litellm.integrations.arize import _utils
from litellm.integrations.opentelemetry import OpenTelemetry
from litellm.integrations.weave.weave_otel_attributes import WeaveLLMObsOTELAttributes
from litellm.types.integrations.weave import WeaveOtelConfig, WeaveSpanAttributes
from litellm.types.utils import StandardCallbackDynamicParams

if TYPE_CHECKING:
    from opentelemetry.trace import Span as _Span

    from litellm.integrations.opentelemetry import (
        OpenTelemetryConfig as _OpenTelemetryConfig,
    )
    from litellm.types.integrations.arize import Protocol as _Protocol

    Protocol = _Protocol
    OpenTelemetryConfig = _OpenTelemetryConfig
    Span = Union[_Span, Any]
else:
    Protocol = Any
    OpenTelemetryConfig = Any
    Span = Any


# Weave OTEL endpoint
# Multi-tenant cloud: https://trace.wandb.ai/otel/v1/traces
# Self-managed: https://<your-subdomain>.wandb.io/traces/otel/v1/traces
WEAVE_CLOUD_ENDPOINT = "https://trace.wandb.ai/otel/v1/traces"


class WeaveOtelLogger(OpenTelemetry):
    """
    Weave (W&B) OpenTelemetry Logger for LiteLLM.

    Sends LLM traces to Weave via the OpenTelemetry Protocol (OTLP).

    Environment Variables:
        WANDB_API_KEY: Required. Weights & Biases API key for authentication.
        WEAVE_PROJECT_ID: Required. Project ID in format <entity>/<project_name>.
        WEAVE_HOST: Optional. Custom Weave host URL. Defaults to cloud endpoint.

    Usage:
        litellm.callbacks = ["weave_otel"]

    Reference:
        https://docs.wandb.ai/weave/guides/tracking/otel
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def set_weave_otel_attributes(span: Span, kwargs, response_obj):
        """
        Sets OpenTelemetry span attributes for Weave observability.
        Uses the same attribute setting logic as other OTEL integrations for consistency.
        """
        _utils.set_attributes(span, kwargs, response_obj, WeaveLLMObsOTELAttributes)

        # Set Weave-specific attributes
        WeaveOtelLogger._set_weave_specific_attributes(
            span=span, kwargs=kwargs, response_obj=response_obj
        )

    @staticmethod
    def _extract_weave_metadata(kwargs: dict) -> dict:
        """
        Extracts Weave metadata from the standard LiteLLM kwargs structure.

        Reads kwargs["litellm_params"]["metadata"] if present and is a dict.
        """
        litellm_params = kwargs.get("litellm_params", {}) or {}
        metadata = litellm_params.get("metadata") or {}
        if metadata is None or not isinstance(metadata, dict):
            metadata = {}
        return metadata

    @staticmethod
    def _set_metadata_attributes(span: Span, metadata: dict):
        """Helper to set metadata attributes from mapping."""
        from litellm.integrations.arize._utils import safe_set_attribute

        mapping = {
            "thread_id": WeaveSpanAttributes.THREAD_ID,
            "is_turn": WeaveSpanAttributes.IS_TURN,
            "trace_user_id": WeaveSpanAttributes.TRACE_USER_ID,
            "session_id": WeaveSpanAttributes.SESSION_ID,
            "trace_name": WeaveSpanAttributes.TRACE_NAME,
            "trace_id": WeaveSpanAttributes.TRACE_ID,
            "trace_metadata": WeaveSpanAttributes.TRACE_METADATA,
            "generation_name": WeaveSpanAttributes.GENERATION_NAME,
            "generation_id": WeaveSpanAttributes.GENERATION_ID,
        }

        for key, enum_attr in mapping.items():
            if key in metadata and metadata[key] is not None:
                value = metadata[key]
                if isinstance(value, (list, dict)):
                    try:
                        value = json.dumps(value)
                    except Exception:
                        value = str(value)
                safe_set_attribute(span, enum_attr.value, value)

    @staticmethod
    def _set_observation_output(span: Span, response_obj):
        """Helper to set observation output attributes."""
        from litellm.integrations.arize._utils import safe_set_attribute
        from litellm.litellm_core_utils.safe_json_dumps import safe_dumps

        if not response_obj or not hasattr(response_obj, "get"):
            return

        choices = response_obj.get("choices", [])
        if choices:
            first_choice = choices[0]
            message = first_choice.get("message", {})
            tool_calls = message.get("tool_calls")
            if tool_calls:
                transformed_tool_calls = []
                for tool_call in tool_calls:
                    function = tool_call.get("function", {})
                    arguments_str = function.get("arguments", "{}")
                    try:
                        arguments_obj = (
                            json.loads(arguments_str)
                            if isinstance(arguments_str, str)
                            else arguments_str
                        )
                    except json.JSONDecodeError:
                        arguments_obj = {}
                    weave_tool_call = {
                        "id": response_obj.get("id", ""),
                        "name": function.get("name", ""),
                        "call_id": tool_call.get("id", ""),
                        "type": "function_call",
                        "arguments": arguments_obj,
                    }
                    transformed_tool_calls.append(weave_tool_call)
                safe_set_attribute(
                    span,
                    WeaveSpanAttributes.OBSERVATION_OUTPUT.value,
                    safe_dumps(transformed_tool_calls),
                )
            else:
                output_data = {}
                if message.get("role"):
                    output_data["role"] = message.get("role")
                if message.get("content") is not None:
                    output_data["content"] = message.get("content")
                if output_data:
                    safe_set_attribute(
                        span,
                        WeaveSpanAttributes.OBSERVATION_OUTPUT.value,
                        safe_dumps(output_data),
                    )

        # Handle ResponsesAPI output format
        output = response_obj.get("output", [])
        if output:
            output_items_data: list[dict] = []
            for item in output:
                if hasattr(item, "type"):
                    item_type = item.type
                    if item_type == "reasoning" and hasattr(item, "summary"):
                        for summary in item.summary:
                            if hasattr(summary, "text"):
                                output_items_data.append(
                                    {"role": "reasoning_summary", "content": summary.text}
                                )
                    elif item_type == "message":
                        output_items_data.append(
                            {
                                "role": getattr(item, "role", "assistant"),
                                "content": getattr(
                                    getattr(item, "content", [{}])[0], "text", ""
                                ),
                            }
                        )
                    elif item_type == "function_call":
                        arguments_str = getattr(item, "arguments", "{}")
                        arguments_obj = (
                            json.loads(arguments_str)
                            if isinstance(arguments_str, str)
                            else arguments_str
                        )
                        weave_tool_call = {
                            "id": getattr(item, "id", ""),
                            "name": getattr(item, "name", ""),
                            "call_id": getattr(item, "call_id", ""),
                            "type": "function_call",
                            "arguments": arguments_obj,
                        }
                        output_items_data.append(weave_tool_call)
            if output_items_data:
                safe_set_attribute(
                    span,
                    WeaveSpanAttributes.OBSERVATION_OUTPUT.value,
                    safe_dumps(output_items_data),
                )

    @staticmethod
    def _set_weave_specific_attributes(span: Span, kwargs, response_obj):
        """
        Sets Weave-specific metadata attributes onto the OTEL span.

        Weave supports thread organization via wandb.thread_id and wandb.is_turn
        attributes, as well as standard trace metadata.
        """
        from litellm.integrations.arize._utils import safe_set_attribute
        from litellm.litellm_core_utils.safe_json_dumps import safe_dumps

        metadata = WeaveOtelLogger._extract_weave_metadata(kwargs)
        WeaveOtelLogger._set_metadata_attributes(span=span, metadata=metadata)

        messages = kwargs.get("messages")
        if messages:
            safe_set_attribute(
                span, WeaveSpanAttributes.OBSERVATION_INPUT.value, safe_dumps(messages)
            )

        WeaveOtelLogger._set_observation_output(span=span, response_obj=response_obj)

    @staticmethod
    def _get_weave_host() -> str | None:
        """
        Returns the Weave OTEL host based on environment variables.

        Returned in the following order of precedence:
        1. WEAVE_OTEL_HOST
        2. WEAVE_HOST
        """
        return os.environ.get("WEAVE_OTEL_HOST") or os.environ.get("WEAVE_HOST")

    @staticmethod
    def get_weave_otel_config() -> WeaveOtelConfig:
        """
        Retrieves the Weave OpenTelemetry configuration based on environment variables.

        Environment Variables:
            WANDB_API_KEY: Required. W&B API key for authentication.
            WEAVE_PROJECT_ID: Required. Project ID in format <entity>/<project_name>.
            WEAVE_HOST: Optional. Custom Weave host URL. Defaults to cloud endpoint.

        Returns:
            WeaveOtelConfig: A Pydantic model containing Weave OTEL configuration.

        Raises:
            ValueError: If required environment variables are missing.
        """
        api_key = os.environ.get("WANDB_API_KEY", None)
        project_id = os.environ.get("WEAVE_PROJECT_ID", None)

        if not api_key:
            raise ValueError(
                "WANDB_API_KEY must be set for Weave OpenTelemetry integration."
            )

        if not project_id:
            raise ValueError(
                "WEAVE_PROJECT_ID must be set for Weave OpenTelemetry integration. "
                "Format: <entity>/<project_name>"
            )

        # Determine endpoint
        weave_host = WeaveOtelLogger._get_weave_host()

        if weave_host:
            if not weave_host.startswith("http"):
                weave_host = "https://" + weave_host
            # Self-managed instances use a different path
            endpoint = f"{weave_host.rstrip('/')}/traces/otel/v1/traces"
            verbose_logger.debug(f"Using Weave OTEL endpoint from host: {endpoint}")
        else:
            endpoint = WEAVE_CLOUD_ENDPOINT
            verbose_logger.debug(f"Using Weave cloud endpoint: {endpoint}")

        # Weave uses Basic auth with format: api:<WANDB_API_KEY>
        auth_header = WeaveOtelLogger._get_weave_authorization_header(api_key=api_key)

        # Weave requires project_id header
        otlp_auth_headers = f"Authorization={auth_header},project_id={project_id}"

        # Set standard OTEL environment variables
        os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = endpoint
        os.environ["OTEL_EXPORTER_OTLP_HEADERS"] = otlp_auth_headers

        return WeaveOtelConfig(
            otlp_auth_headers=otlp_auth_headers,
            project_id=project_id,
            protocol="otlp_http",
        )

    @staticmethod
    def _get_weave_authorization_header(api_key: str) -> str:
        """
        Get the authorization header for Weave OpenTelemetry.

        Weave uses Basic auth with format: api:<WANDB_API_KEY>
        """
        auth_string = f"api:{api_key}"
        auth_header = base64.b64encode(auth_string.encode()).decode()
        return f"Basic {auth_header}"

    def construct_dynamic_otel_headers(
        self, standard_callback_dynamic_params: StandardCallbackDynamicParams
    ) -> dict | None:
        """
        Construct dynamic Weave headers from standard callback dynamic params.

        This is used for team/key based logging.

        Returns:
            dict: A dictionary of dynamic Weave headers
        """
        dynamic_headers = {}

        dynamic_wandb_api_key = standard_callback_dynamic_params.get("wandb_api_key")
        dynamic_weave_project_id = standard_callback_dynamic_params.get(
            "weave_project_id"
        )

        if dynamic_wandb_api_key:
            auth_header = WeaveOtelLogger._get_weave_authorization_header(
                api_key=dynamic_wandb_api_key,
            )
            dynamic_headers["Authorization"] = auth_header

        if dynamic_weave_project_id:
            dynamic_headers["project_id"] = dynamic_weave_project_id

        return dynamic_headers if dynamic_headers else None
