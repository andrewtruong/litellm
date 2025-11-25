from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel

if TYPE_CHECKING:
    Protocol = Literal["otlp_grpc", "otlp_http"]
else:
    Protocol = Any


class WeaveOtelConfig(BaseModel):
    """Configuration for Weave OpenTelemetry integration."""

    otlp_auth_headers: str | None = None
    project_id: str | None = None
    protocol: Protocol = "otlp_http"


class WeaveSpanAttributes(str, Enum):
    """
    Weave-specific span attributes for OpenTelemetry traces.

    Based on Weave's OTEL integration documentation:
    https://docs.wandb.ai/weave/guides/tracking/otel
    """

    # ---- Thread organization ----
    THREAD_ID = "wandb.thread_id"
    IS_TURN = "wandb.is_turn"

    # ---- Observation input/output ----
    # Weave maps these from various frameworks including Langfuse attributes
    OBSERVATION_INPUT = "weave.observation.input"
    OBSERVATION_OUTPUT = "weave.observation.output"

    # ---- Trace-level metadata ----
    TRACE_USER_ID = "user.id"
    SESSION_ID = "session.id"
    TRACE_NAME = "weave.trace.name"
    TRACE_ID = "weave.trace.id"
    TRACE_METADATA = "weave.trace.metadata"

    # ---- Generation-level metadata ----
    GENERATION_NAME = "weave.generation.name"
    GENERATION_ID = "weave.generation.id"
