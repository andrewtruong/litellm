import json
import os
from unittest.mock import MagicMock, patch

import pytest

from litellm.integrations.weave.weave_otel import WeaveOtelLogger
from litellm.types.integrations.weave import WeaveOtelConfig


class TestWeaveOtelIntegration:

    def test_get_weave_otel_config_with_required_env_vars(self):
        """Test that config is created correctly with required environment variables."""
        env_vars_to_clean = [
            "WEAVE_HOST",
            "OTEL_EXPORTER_OTLP_ENDPOINT",
            "OTEL_EXPORTER_OTLP_HEADERS",
        ]
        with patch.dict(
            os.environ,
            {
                "WANDB_API_KEY": "test_api_key",
                "WEAVE_PROJECT_ID": "test-entity/test-project",
            },
            clear=False,
        ):
            # Remove any existing Weave variables
            for var in env_vars_to_clean:
                if var in os.environ:
                    del os.environ[var]

            config = WeaveOtelLogger.get_weave_otel_config()

            assert isinstance(config, WeaveOtelConfig)
            assert config.protocol == "otlp_http"
            assert config.project_id == "test-entity/test-project"
            assert "Authorization=Basic" in config.otlp_auth_headers
            assert "project_id=test-entity/test-project" in config.otlp_auth_headers
            # Check that environment variables are set correctly (cloud default)
            assert (
                os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")
                == "https://trace.wandb.ai/otel/v1/traces"
            )
            assert "Authorization=Basic" in os.environ.get(
                "OTEL_EXPORTER_OTLP_HEADERS", ""
            )

    def test_get_weave_otel_config_missing_api_key(self):
        """Test that ValueError is raised when WANDB_API_KEY is missing."""
        with patch.dict(
            os.environ, {"WEAVE_PROJECT_ID": "test-entity/test-project"}, clear=True
        ):
            with pytest.raises(
                ValueError, match="WANDB_API_KEY must be set for Weave OpenTelemetry"
            ):
                WeaveOtelLogger.get_weave_otel_config()

    def test_get_weave_otel_config_missing_project_id(self):
        """Test that ValueError is raised when WEAVE_PROJECT_ID is missing."""
        with patch.dict(os.environ, {"WANDB_API_KEY": "test_api_key"}, clear=True):
            with pytest.raises(
                ValueError, match="WEAVE_PROJECT_ID must be set for Weave OpenTelemetry"
            ):
                WeaveOtelLogger.get_weave_otel_config()

    def test_get_weave_otel_config_with_custom_host(self):
        """Test config with custom host."""
        with patch.dict(
            os.environ,
            {
                "WANDB_API_KEY": "test_api_key",
                "WEAVE_PROJECT_ID": "test-entity/test-project",
                "WEAVE_HOST": "https://my-weave.wandb.io",
            },
            clear=False,
        ):
            config = WeaveOtelLogger.get_weave_otel_config()

            assert (
                os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")
                == "https://my-weave.wandb.io/traces/otel/v1/traces"
            )

    def test_get_weave_otel_config_with_host_no_protocol(self):
        """Test config with custom host without protocol."""
        with patch.dict(
            os.environ,
            {
                "WANDB_API_KEY": "test_api_key",
                "WEAVE_PROJECT_ID": "test-entity/test-project",
                "WEAVE_HOST": "my-weave.wandb.io",
            },
            clear=False,
        ):
            config = WeaveOtelLogger.get_weave_otel_config()

            assert (
                os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")
                == "https://my-weave.wandb.io/traces/otel/v1/traces"
            )

    def test_get_weave_otel_config_with_otel_host_priority(self):
        """WEAVE_OTEL_HOST should take priority over WEAVE_HOST."""
        with patch.dict(
            os.environ,
            {
                "WANDB_API_KEY": "test_api_key",
                "WEAVE_PROJECT_ID": "test-entity/test-project",
                "WEAVE_HOST": "https://should-not-be-used.com",
                "WEAVE_OTEL_HOST": "https://otel-host.wandb.io",
            },
            clear=False,
        ):
            _ = WeaveOtelLogger.get_weave_otel_config()

            assert (
                os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")
                == "https://otel-host.wandb.io/traces/otel/v1/traces"
            )

    def test_set_weave_otel_attributes(self):
        """Test that set_weave_otel_attributes calls the Arize utils function."""
        from litellm.integrations.weave.weave_otel_attributes import (
            WeaveLLMObsOTELAttributes,
        )

        mock_span = MagicMock()
        mock_kwargs = {"test": "kwargs"}
        mock_response = {"test": "response"}

        with patch(
            "litellm.integrations.arize._utils.set_attributes"
        ) as mock_set_attributes:
            WeaveOtelLogger.set_weave_otel_attributes(
                mock_span, mock_kwargs, mock_response
            )

            mock_set_attributes.assert_called_once_with(
                mock_span, mock_kwargs, mock_response, WeaveLLMObsOTELAttributes
            )

    def test_extract_weave_metadata_basic(self):
        """Ensure metadata is correctly pulled from litellm_params."""
        metadata_in = {"generation_name": "my-gen", "custom": "data"}
        kwargs = {"litellm_params": {"metadata": metadata_in}}
        extracted = WeaveOtelLogger._extract_weave_metadata(kwargs)
        assert extracted == metadata_in

    def test_set_weave_specific_attributes_metadata(self):
        """Verify supported metadata keys map to the correct OTEL attributes."""
        metadata = {
            "thread_id": "thread-123",
            "is_turn": True,
            "trace_user_id": "user-123",
            "session_id": "sess-456",
            "trace_name": "trace-name",
            "trace_id": "trace-id",
            "trace_metadata": {"k": "v"},
            "generation_name": "gen-name",
            "generation_id": "gen-id",
        }
        kwargs = {"litellm_params": {"metadata": metadata}}

        with patch(
            "litellm.integrations.arize._utils.safe_set_attribute"
        ) as mock_safe_set_attribute:
            WeaveOtelLogger._set_weave_specific_attributes(MagicMock(), kwargs, None)

            from litellm.types.integrations.weave import WeaveSpanAttributes

            expected = {
                WeaveSpanAttributes.THREAD_ID.value: "thread-123",
                WeaveSpanAttributes.IS_TURN.value: True,
                WeaveSpanAttributes.TRACE_USER_ID.value: "user-123",
                WeaveSpanAttributes.SESSION_ID.value: "sess-456",
                WeaveSpanAttributes.TRACE_NAME.value: "trace-name",
                WeaveSpanAttributes.TRACE_ID.value: "trace-id",
                WeaveSpanAttributes.TRACE_METADATA.value: json.dumps({"k": "v"}),
                WeaveSpanAttributes.GENERATION_NAME.value: "gen-name",
                WeaveSpanAttributes.GENERATION_ID.value: "gen-id",
            }

            # Flatten the actual calls into {key: value}
            actual = {
                call.args[1]: call.args[2]  # (span, key, value)
                for call in mock_safe_set_attribute.call_args_list
            }

            assert (
                actual == expected
            ), "Mismatch between expected and actual OTEL attribute mapping."

    def test_set_weave_specific_attributes_with_content(self):
        """Test that _set_weave_specific_attributes correctly sets observation.output."""
        from litellm.types.integrations.weave import WeaveSpanAttributes
        from litellm.types.utils import Choices, ModelResponse

        response_obj = ModelResponse(
            id="chatcmpl-test",
            model="gpt-4o",
            choices=[
                Choices(
                    finish_reason="stop",
                    message={
                        "role": "assistant",
                        "content": "The weather in Tokyo is sunny.",
                    },
                )
            ],
        )

        kwargs = {
            "messages": [{"role": "user", "content": "What's the weather in Tokyo?"}],
        }

        with patch(
            "litellm.integrations.arize._utils.safe_set_attribute"
        ) as mock_safe_set_attribute:
            WeaveOtelLogger._set_weave_specific_attributes(
                MagicMock(), kwargs, response_obj
            )

            expect_output = {
                WeaveSpanAttributes.OBSERVATION_INPUT.value: [
                    {"role": "user", "content": "What's the weather in Tokyo?"}
                ],
                WeaveSpanAttributes.OBSERVATION_OUTPUT.value: {
                    "role": "assistant",
                    "content": "The weather in Tokyo is sunny.",
                },
            }

            actual = {
                call.args[1]: json.loads(call.args[2])
                for call in mock_safe_set_attribute.call_args_list
            }

            assert (
                actual == expect_output
            ), "Mismatch in observation input/output OTEL attributes."

    def test_set_weave_specific_attributes_with_tool_calls(self):
        """Test that _set_weave_specific_attributes correctly sets observation.output with tool calls."""
        from litellm.types.integrations.weave import WeaveSpanAttributes
        from litellm.types.utils import (
            ChatCompletionMessageToolCall,
            Choices,
            Function,
            ModelResponse,
        )

        response_obj = ModelResponse(
            id="chatcmpl-test",
            model="gpt-4o",
            choices=[
                Choices(
                    finish_reason="tool_calls",
                    message={
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            ChatCompletionMessageToolCall(
                                function=Function(
                                    arguments='{"location":"Tokyo"}', name="get_weather"
                                ),
                                id="call_123",
                                type="function",
                            )
                        ],
                    },
                )
            ],
        )

        with patch(
            "litellm.integrations.arize._utils.safe_set_attribute"
        ) as mock_safe_set_attribute:
            WeaveOtelLogger._set_weave_specific_attributes(
                MagicMock(), {}, response_obj
            )

            expected = {
                WeaveSpanAttributes.OBSERVATION_OUTPUT.value: [
                    {
                        "id": "chatcmpl-test",
                        "name": "get_weather",
                        "arguments": {"location": "Tokyo"},
                        "call_id": "call_123",
                        "type": "function_call",
                    }
                ]
            }

            actual = {
                call.args[1]: json.loads(call.args[2])
                for call in mock_safe_set_attribute.call_args_list
            }
            assert (
                actual == expected
            ), "Mismatch in observation output OTEL attribute for tool calls."

    def test_construct_dynamic_otel_headers_with_weave_keys(self):
        """Test that construct_dynamic_otel_headers creates proper auth headers."""
        from litellm.types.utils import StandardCallbackDynamicParams

        dynamic_params = StandardCallbackDynamicParams(
            wandb_api_key="test_api_key", weave_project_id="test-entity/test-project"
        )

        logger = WeaveOtelLogger()
        result = logger.construct_dynamic_otel_headers(dynamic_params)

        assert result is not None
        assert "Authorization" in result
        assert "project_id" in result

        auth_header = result["Authorization"]
        assert auth_header.startswith("Basic ")

        import base64

        base64_part = auth_header.replace("Basic ", "")
        decoded = base64.b64decode(base64_part).decode()

        assert decoded == "api:test_api_key"
        assert result["project_id"] == "test-entity/test-project"

    def test_construct_dynamic_otel_headers_empty_params(self):
        """Test that construct_dynamic_otel_headers returns None when no keys provided."""
        from litellm.types.utils import StandardCallbackDynamicParams

        dynamic_params = StandardCallbackDynamicParams()

        logger = WeaveOtelLogger()
        result = logger.construct_dynamic_otel_headers(dynamic_params)

        assert result is None

    def test_weave_authorization_header_format(self):
        """Test that Weave auth header uses correct api:<key> format."""
        import base64

        auth_header = WeaveOtelLogger._get_weave_authorization_header("my_api_key")

        assert auth_header.startswith("Basic ")
        base64_part = auth_header.replace("Basic ", "")
        decoded = base64.b64decode(base64_part).decode()

        # Weave uses api:<api_key> format (not user:key like Langfuse)
        assert decoded == "api:my_api_key"


class TestWeaveOtelResponsesAPI:
    """Test suite for Weave OTEL integration with ResponsesAPI"""

    def test_weave_otel_with_responses_api(self):
        """Test that Weave OTEL logger works with ResponsesAPI responses."""
        from litellm.types.llms.openai import ResponsesAPIResponse

        mock_response = ResponsesAPIResponse(
            id="response-123",
            created_at=1234567890,
            output=[
                {
                    "type": "message",
                    "content": [{"type": "text", "text": "Hello from responses API"}],
                }
            ],
            parallel_tool_calls=False,
            tool_choice="auto",
            tools=[],
            top_p=1.0,
        )

        test_metadata = {
            "thread_id": "thread123",
            "session_id": "abc456",
            "custom_field": "test_value",
            "generation_name": "responses_test_generation",
            "trace_name": "responses_api_trace",
        }

        kwargs = {
            "call_type": "responses",
            "messages": [{"role": "user", "content": "Hello"}],
            "model": "gpt-4o",
            "optional_params": {},
            "litellm_params": {"metadata": test_metadata},
        }

        mock_span = MagicMock()

        from litellm.integrations.weave.weave_otel_attributes import (
            WeaveLLMObsOTELAttributes,
        )

        with patch(
            "litellm.integrations.arize._utils.set_attributes"
        ) as mock_set_attributes:
            with patch(
                "litellm.integrations.arize._utils.safe_set_attribute"
            ) as mock_safe_set_attribute:
                logger = WeaveOtelLogger()
                logger.set_weave_otel_attributes(mock_span, kwargs, mock_response)

                mock_set_attributes.assert_called_once_with(
                    mock_span, kwargs, mock_response, WeaveLLMObsOTELAttributes
                )

                mock_safe_set_attribute.assert_any_call(
                    mock_span, "weave.generation.name", "responses_test_generation"
                )
                mock_safe_set_attribute.assert_any_call(
                    mock_span, "weave.trace.name", "responses_api_trace"
                )

    def test_responses_api_with_output(self):
        """Test Weave OTEL logger with Responses API output (reasoning + message)."""
        from openai.types.responses import (
            ResponseOutputMessage,
            ResponseOutputText,
            ResponseReasoningItem,
        )
        from openai.types.responses.response_reasoning_item import Summary

        from litellm.types.integrations.weave import WeaveSpanAttributes
        from litellm.types.llms.openai import ResponsesAPIResponse

        response_obj = ResponsesAPIResponse(
            id="response-456",
            created_at=1625247600,
            output=[
                ResponseReasoningItem(
                    id="reasoning-001",
                    type="reasoning",
                    summary=[
                        Summary(
                            text="Let me analyze this problem step by step...",
                            type="summary_text",
                        )
                    ],
                ),
                ResponseOutputMessage(
                    id="msg-001",
                    type="message",
                    role="assistant",
                    status="completed",
                    content=[
                        ResponseOutputText(
                            annotations=[],
                            text="The weather in San Francisco is sunny, 20°C.",
                            type="output_text",
                        )
                    ],
                ),
            ],
        )

        kwargs = {
            "call_type": "responses",
            "messages": [
                {"role": "user", "content": "What's the weather in San Francisco?"}
            ],
            "model": "gpt-4o",
            "optional_params": {},
        }

        mock_span = MagicMock()

        with patch(
            "litellm.integrations.arize._utils.safe_set_attribute"
        ) as mock_safe_set_attribute:
            WeaveOtelLogger._set_weave_specific_attributes(
                mock_span, kwargs, response_obj
            )

            output_calls = [
                call
                for call in mock_safe_set_attribute.call_args_list
                if call.args[1] == WeaveSpanAttributes.OBSERVATION_OUTPUT.value
            ]

            assert len(output_calls) > 0, "observation.output should be set"
            output_json = output_calls[0].args[2]
            output_data = json.loads(output_json)

            assert isinstance(output_data, list)
            assert len(output_data) == 2

            assert output_data[0]["role"] == "reasoning_summary"
            assert (
                output_data[0]["content"]
                == "Let me analyze this problem step by step..."
            )

            assert output_data[1]["role"] == "assistant"
            assert (
                output_data[1]["content"]
                == "The weather in San Francisco is sunny, 20°C."
            )

    def test_responses_api_with_function_calls(self):
        """Test Weave OTEL logger with Responses API function_call output."""
        from openai.types.responses import ResponseFunctionToolCall

        from litellm.types.integrations.weave import WeaveSpanAttributes
        from litellm.types.llms.openai import ResponsesAPIResponse

        response_obj = ResponsesAPIResponse(
            id="response-789",
            created_at=1625247700,
            output=[
                ResponseFunctionToolCall(
                    id="fc-123",
                    type="function_call",
                    name="get_weather",
                    call_id="call-abc",
                    arguments='{"location": "San Francisco", "unit": "celsius"}',
                    status="completed",
                )
            ],
        )

        kwargs = {
            "call_type": "responses",
            "messages": [
                {"role": "user", "content": "What's the weather in San Francisco?"}
            ],
            "model": "gpt-4o",
            "optional_params": {},
        }

        mock_span = MagicMock()

        with patch(
            "litellm.integrations.arize._utils.safe_set_attribute"
        ) as mock_safe_set_attribute:
            WeaveOtelLogger._set_weave_specific_attributes(
                mock_span, kwargs, response_obj
            )

            output_calls = [
                call
                for call in mock_safe_set_attribute.call_args_list
                if call.args[1] == WeaveSpanAttributes.OBSERVATION_OUTPUT.value
            ]

            assert len(output_calls) > 0, "observation.output should be set"
            output_json = output_calls[0].args[2]
            output_data = json.loads(output_json)

            assert isinstance(output_data, list)
            assert len(output_data) == 1

            assert output_data[0]["type"] == "function_call"
            assert output_data[0]["id"] == "fc-123"
            assert output_data[0]["name"] == "get_weather"
            assert output_data[0]["call_id"] == "call-abc"
            assert output_data[0]["arguments"]["location"] == "San Francisco"
            assert output_data[0]["arguments"]["unit"] == "celsius"


class TestWeaveThreadOrganization:
    """Test suite for Weave's thread organization features."""

    def test_thread_id_attribute(self):
        """Test that thread_id is correctly set for trace organization."""
        from litellm.types.integrations.weave import WeaveSpanAttributes

        metadata = {"thread_id": "conversation-123"}
        kwargs = {"litellm_params": {"metadata": metadata}}

        with patch(
            "litellm.integrations.arize._utils.safe_set_attribute"
        ) as mock_safe_set_attribute:
            WeaveOtelLogger._set_weave_specific_attributes(MagicMock(), kwargs, None)

            mock_safe_set_attribute.assert_any_call(
                mock_safe_set_attribute.call_args_list[0].args[0],
                WeaveSpanAttributes.THREAD_ID.value,
                "conversation-123",
            )

    def test_is_turn_attribute(self):
        """Test that is_turn is correctly set for conversation turns."""
        from litellm.types.integrations.weave import WeaveSpanAttributes

        metadata = {"thread_id": "conversation-123", "is_turn": True}
        kwargs = {"litellm_params": {"metadata": metadata}}

        with patch(
            "litellm.integrations.arize._utils.safe_set_attribute"
        ) as mock_safe_set_attribute:
            WeaveOtelLogger._set_weave_specific_attributes(MagicMock(), kwargs, None)

            actual = {
                call.args[1]: call.args[2]
                for call in mock_safe_set_attribute.call_args_list
            }

            assert actual.get(WeaveSpanAttributes.THREAD_ID.value) == "conversation-123"
            assert actual.get(WeaveSpanAttributes.IS_TURN.value) is True


if __name__ == "__main__":
    pytest.main([__file__])
