"""OpenTelemetry tracing for the incident-resolution agent.

Spans go to a local Jaeger collector over OTLP/HTTP. Attributes are counts
and enums only: questions, retrieved text, and credentials stay out of them.
"""

import atexit
import os

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

SERVICE_NAME = "incident-copilot"
_DEFAULT_OTLP_ENDPOINT = "http://localhost:4318"
_TRACES_PATH = "/v1/traces"

_configured = False
_tracer = None


def setup_tracing():
    """Install the OTLP exporter once. A provider already in place is kept."""
    global _configured, _tracer
    if _configured:
        return _tracer
    _configured = True
    try:
        _tracer = _build_tracer()
    except Exception:
        _tracer = trace.get_tracer(SERVICE_NAME)
    return _tracer


def get_tracer():
    if not _configured:
        setup_tracing()
    return _tracer


def _build_tracer():
    provider = trace.get_tracer_provider()
    if provider.__class__.__name__ != "ProxyTracerProvider":
        return trace.get_tracer(SERVICE_NAME)
    if _sdk_disabled():
        return trace.get_tracer(SERVICE_NAME)

    provider = TracerProvider(
        resource=Resource.create({"service.name": SERVICE_NAME})
    )
    provider.add_span_processor(
        BatchSpanProcessor(OTLPSpanExporter(endpoint=_traces_endpoint()))
    )
    trace.set_tracer_provider(provider)
    atexit.register(provider.shutdown)
    return trace.get_tracer(SERVICE_NAME)


def _sdk_disabled():
    return os.environ.get("OTEL_SDK_DISABLED", "").strip().lower() in {
        "1", "true", "yes",
    }


def _traces_endpoint():
    explicit = os.environ.get("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT")
    if explicit:
        return explicit
    base = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT", _DEFAULT_OTLP_ENDPOINT)
    base = base.rstrip("/")
    if base.endswith(_TRACES_PATH):
        return base
    return base + _TRACES_PATH
