"""Fail-closed provenance binding for AttnSketch tensors exported by DMI.

AttnSketch summary tensors are intentionally small. Repeating a full kernel
fingerprint and semantic manifest in every query/head payload would make the
metadata larger than the measurements. DMI already stores ``model_id`` and
``request_id`` beside every exported tensor, so this module binds an immutable
capture manifest to ``model_id`` and request/page-table epochs to
``request_id``. The tensor payload remains metric-only.

Consumers MUST resolve and validate both identifiers before interpreting an
``attn_summary`` tensor. Unknown capture IDs, digest collisions, malformed
request IDs, or epoch mismatches are hard errors.
"""

from __future__ import annotations

import base64
import binascii
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Iterable, Mapping


ATTSKETCH_PROVENANCE_SCHEMA_VERSION = 1
_CAPTURE_PREFIX = "attnsketch:v1:"
_REQUEST_PREFIX = "as1."


def _require_digest(value: str, name: str) -> None:
    if not value.startswith("sha256:") or len(value) != len("sha256:") + 64:
        raise ValueError(f"{name} must be a full sha256:<64 hex chars> digest")
    try:
        int(value.removeprefix("sha256:"), 16)
    except ValueError as exc:
        raise ValueError(f"{name} is not hexadecimal") from exc


@dataclass(frozen=True)
class AttnSketchCaptureProvenance:
    """Immutable semantics shared by every summary tensor in one capture."""

    kernel_fingerprint: str
    artifact_hash: str
    manifest_version: str
    score_semantics_hash: str
    semantic_mapping_version: str
    layout_hash: str
    query_contract_hash: str
    metrics: tuple[str, ...]
    schema_version: int = ATTSKETCH_PROVENANCE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != ATTSKETCH_PROVENANCE_SCHEMA_VERSION:
            raise ValueError("unsupported AttnSketch provenance schema version")
        for name in (
            "kernel_fingerprint",
            "artifact_hash",
            "score_semantics_hash",
            "layout_hash",
            "query_contract_hash",
        ):
            _require_digest(getattr(self, name), name)
        if not self.manifest_version or not self.semantic_mapping_version:
            raise ValueError("manifest and semantic mapping versions are required")
        if not self.metrics or len(set(self.metrics)) != len(self.metrics):
            raise ValueError("metrics must be a non-empty tuple without duplicates")

    def canonical_bytes(self) -> bytes:
        return json.dumps(
            asdict(self), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")

    @property
    def capture_id(self) -> str:
        digest = hashlib.sha256(self.canonical_bytes()).hexdigest()
        return _CAPTURE_PREFIX + digest


@dataclass(frozen=True)
class AttnSketchRequestBinding:
    """Request identity plus allocator epochs required for safe attribution."""

    request_id: str
    request_table_epoch: int
    page_table_epoch: int

    def __post_init__(self) -> None:
        if not self.request_id:
            raise ValueError("request_id must not be empty")
        if self.request_table_epoch < 0 or self.page_table_epoch < 0:
            raise ValueError("request and page-table epochs must be non-negative")

    def encode(self) -> str:
        encoded_id = base64.urlsafe_b64encode(
            self.request_id.encode("utf-8")
        ).decode("ascii").rstrip("=")
        return (
            f"{_REQUEST_PREFIX}{self.request_table_epoch:x}."
            f"{self.page_table_epoch:x}.{encoded_id}"
        )

    @classmethod
    def decode(cls, encoded: str) -> "AttnSketchRequestBinding":
        if not encoded.startswith(_REQUEST_PREFIX):
            raise ValueError("request_id is not an AttnSketch-bound identifier")
        fields = encoded[len(_REQUEST_PREFIX) :].split(".")
        if len(fields) != 3 or not all(fields):
            raise ValueError("malformed AttnSketch request identifier")
        request_epoch_hex, page_epoch_hex, encoded_id = fields
        try:
            request_epoch = int(request_epoch_hex, 16)
            page_epoch = int(page_epoch_hex, 16)
            padding = "=" * (-len(encoded_id) % 4)
            raw_id = base64.b64decode(
                encoded_id + padding, altchars=b"-_", validate=True
            ).decode("utf-8")
        except (ValueError, UnicodeDecodeError, binascii.Error) as exc:
            raise ValueError("invalid AttnSketch request identifier") from exc
        return cls(raw_id, request_epoch, page_epoch)


class AttnSketchProvenanceRegistry:
    """Exact capture-ID registry persisted next to exported DMI records."""

    def __init__(
        self, entries: Iterable[AttnSketchCaptureProvenance] = ()
    ) -> None:
        self._entries: dict[str, AttnSketchCaptureProvenance] = {}
        for entry in entries:
            self.register(entry)

    def register(self, provenance: AttnSketchCaptureProvenance) -> str:
        capture_id = provenance.capture_id
        existing = self._entries.get(capture_id)
        if existing is not None and existing != provenance:
            raise ValueError("AttnSketch capture digest collision")
        self._entries[capture_id] = provenance
        return capture_id

    def resolve(self, capture_id: str) -> AttnSketchCaptureProvenance:
        if not capture_id.startswith(_CAPTURE_PREFIX):
            raise ValueError("model_id is not an AttnSketch capture ID")
        try:
            provenance = self._entries[capture_id]
        except KeyError as exc:
            raise ValueError("unknown AttnSketch capture ID") from exc
        if provenance.capture_id != capture_id:
            raise ValueError("AttnSketch registry entry failed digest verification")
        return provenance

    def write(self, path: str | Path) -> None:
        document = {
            "schema_version": ATTSKETCH_PROVENANCE_SCHEMA_VERSION,
            "captures": {
                capture_id: asdict(provenance)
                for capture_id, provenance in sorted(self._entries.items())
            },
        }
        Path(path).write_text(
            json.dumps(document, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def read(cls, path: str | Path) -> "AttnSketchProvenanceRegistry":
        try:
            document = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"cannot read AttnSketch registry: {exc}") from exc
        if not isinstance(document, Mapping) or set(document) != {
            "schema_version",
            "captures",
        }:
            raise ValueError("invalid AttnSketch registry fields")
        if document["schema_version"] != ATTSKETCH_PROVENANCE_SCHEMA_VERSION:
            raise ValueError("unsupported AttnSketch registry schema version")
        captures = document["captures"]
        if not isinstance(captures, Mapping):
            raise ValueError("AttnSketch captures must be an object")
        registry = cls()
        expected_fields = set(AttnSketchCaptureProvenance.__dataclass_fields__)
        for capture_id, payload in captures.items():
            if not isinstance(payload, Mapping) or set(payload) != expected_fields:
                raise ValueError("invalid AttnSketch capture entry fields")
            normalized = dict(payload)
            metrics = normalized.get("metrics")
            if not isinstance(metrics, list):
                raise ValueError("capture metrics must be a list")
            normalized["metrics"] = tuple(metrics)
            provenance = AttnSketchCaptureProvenance(**normalized)
            if provenance.capture_id != capture_id:
                raise ValueError("capture entry key does not match its digest")
            registry.register(provenance)
        return registry


def validate_attnsketch_export_identity(
    *,
    model_id: str,
    request_id: str,
    registry: AttnSketchProvenanceRegistry,
    expected_provenance: AttnSketchCaptureProvenance,
    expected_request_id: str,
    expected_request_table_epoch: int,
    expected_page_table_epoch: int,
) -> None:
    """Reject a DMI tensor unless all capture and attribution fields match."""

    observed_provenance = registry.resolve(model_id)
    if observed_provenance != expected_provenance:
        raise ValueError("AttnSketch capture provenance mismatch")
    binding = AttnSketchRequestBinding.decode(request_id)
    expected_binding = AttnSketchRequestBinding(
        expected_request_id,
        expected_request_table_epoch,
        expected_page_table_epoch,
    )
    if binding != expected_binding:
        raise ValueError("AttnSketch request or page attribution mismatch")


__all__ = [
    "ATTSKETCH_PROVENANCE_SCHEMA_VERSION",
    "AttnSketchCaptureProvenance",
    "AttnSketchProvenanceRegistry",
    "AttnSketchRequestBinding",
    "validate_attnsketch_export_identity",
]
