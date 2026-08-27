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
ATTSKETCH_TOKEN_FOCUS_SCHEMA_VERSION = 1
ATTSKETCH_TOKEN_FOCUS_FLOW_SCHEMA_VERSION = 1
ATTSKETCH_TOKEN_FOCUS_TOKEN_FLOW_SCHEMA_VERSION = 1
_CAPTURE_PREFIX = "attnsketch:v1:"
_REQUEST_PREFIX = "as1."
_PAGE_MAPPING_PREFIX = "sha256:"


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
class AttnSketchTokenFocusSchema:
    """Versioned exact compact token-focus payload contract.

    The payload is a contiguous FP32 tensor with shape
    ``[requests, layers, local_heads, 2*K+2]``.  Token IDs are represented as
    exactly integral FP32 values, interleaved with probabilities; coverage and
    tail mass occupy the last two fields.  Binary provenance remains in the
    capture manifest rather than being repeated in every GPU record.
    """

    top_k: int
    layers: int
    local_heads: int
    schema_version: int = ATTSKETCH_TOKEN_FOCUS_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != ATTSKETCH_TOKEN_FOCUS_SCHEMA_VERSION:
            raise ValueError("unsupported AttnSketch token-focus schema version")
        if min(self.top_k, self.layers, self.local_heads) < 1:
            raise ValueError("token-focus K, layers, and local heads must be positive")

    @property
    def fields(self) -> tuple[str, ...]:
        ranked = tuple(
            name
            for rank in range(self.top_k)
            for name in (f"token_id_{rank}", f"probability_{rank}")
        )
        return ranked + ("coverage", "tail_mass")

    @property
    def width(self) -> int:
        return 2 * self.top_k + 2

    def expected_shape(self, requests: int) -> tuple[int, int, int, int]:
        if requests < 1:
            raise ValueError("token-focus request count must be positive")
        return (requests, self.layers, self.local_heads, self.width)

    @property
    def contract_hash(self) -> str:
        document = {
            "observable": "exact_token_topk_focus",
            "schema_version": self.schema_version,
            "top_k": self.top_k,
            "layers": self.layers,
            "local_heads": self.local_heads,
            "dtype": "float32",
            "fields": self.fields,
            "ordering": "probability_desc_token_id_asc",
        }
        encoded = json.dumps(
            document, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return "sha256:" + hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class AttnSketchTokenFocusFlowSchema:
    """Exact Top-K head plus exact categorical flow-region samples.

    The contiguous FP32 payload has shape
    ``[requests, layers, local_heads, 2*K+2+S]``.  The first ``2*K+2``
    fields retain :class:`AttnSketchTokenFocusSchema` exactly.  Each trailing
    field is an exactly integral region ID sampled from attention mass over
    the version-pinned FA2 native split partition.  Request sequence length
    and the immutable split geometry reconstruct the sampled key interval.
    """

    top_k: int
    layers: int
    local_heads: int
    flow_samples: int
    flow_region_count: int
    flow_tile_tokens: int = 128
    schema_version: int = ATTSKETCH_TOKEN_FOCUS_FLOW_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != ATTSKETCH_TOKEN_FOCUS_FLOW_SCHEMA_VERSION:
            raise ValueError("unsupported AttnSketch token-focus-flow schema version")
        if min(
            self.top_k,
            self.layers,
            self.local_heads,
            self.flow_samples,
            self.flow_region_count,
            self.flow_tile_tokens,
        ) < 1:
            raise ValueError("token-focus-flow dimensions must be positive")
        if self.flow_samples > 16:
            raise ValueError("token-focus-flow supports at most 16 samples")
        if self.flow_region_count > 32:
            raise ValueError("token-focus-flow supports at most 32 regions")

    @property
    def fields(self) -> tuple[str, ...]:
        head = AttnSketchTokenFocusSchema(
            top_k=self.top_k,
            layers=self.layers,
            local_heads=self.local_heads,
        ).fields
        return head + tuple(
            f"flow_region_id_{sample}" for sample in range(self.flow_samples)
        )

    @property
    def width(self) -> int:
        return 2 * self.top_k + 2 + self.flow_samples

    def expected_shape(self, requests: int) -> tuple[int, int, int, int]:
        if requests < 1:
            raise ValueError("token-focus-flow request count must be positive")
        return (requests, self.layers, self.local_heads, self.width)

    @property
    def contract_hash(self) -> str:
        document = {
            "observable": "exact_token_topk_with_exact_attention_flow_regions",
            "schema_version": self.schema_version,
            "top_k": self.top_k,
            "layers": self.layers,
            "local_heads": self.local_heads,
            "flow_samples": self.flow_samples,
            "flow_region_count": self.flow_region_count,
            "flow_tile_tokens": self.flow_tile_tokens,
            "dtype": "float32",
            "fields": self.fields,
            "ordering": "probability_desc_token_id_asc_then_flow_samples",
            "flow_distribution": (
                "categorical_attention_mass_over_native_fa2_splits"
            ),
            "flow_region_mapping": (
                "tiles_per_region=ceil(ceil(sequence_length/tile_tokens)/"
                "flow_region_count)"
            ),
        }
        encoded = json.dumps(
            document, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return "sha256:" + hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class AttnSketchTokenFocusTokenFlowSchema:
    """Exact Top-K head plus exact full-distribution token samples.

    The contiguous FP32 payload has shape
    ``[requests, layers, local_heads, 2*K+2+S]``.  The Top-K prefix is
    unchanged.  Every appended field is an exactly integral logical token ID
    drawn from the complete attention distribution by native-split sampling
    followed by bounded conditional replay.  Producers must fail closed when
    the logical position domain exceeds FP32's exact-integer range ``[0,2**24)``.
    """

    top_k: int
    layers: int
    local_heads: int
    flow_samples: int
    schema_version: int = ATTSKETCH_TOKEN_FOCUS_TOKEN_FLOW_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != ATTSKETCH_TOKEN_FOCUS_TOKEN_FLOW_SCHEMA_VERSION:
            raise ValueError(
                "unsupported AttnSketch token-focus-token-flow schema version"
            )
        if min(self.top_k, self.layers, self.local_heads, self.flow_samples) < 1:
            raise ValueError("token-focus-token-flow dimensions must be positive")
        if self.flow_samples > 16:
            raise ValueError("token-focus-token-flow supports at most 16 samples")

    @property
    def fields(self) -> tuple[str, ...]:
        head = AttnSketchTokenFocusSchema(
            top_k=self.top_k,
            layers=self.layers,
            local_heads=self.local_heads,
        ).fields
        return head + tuple(
            f"flow_token_id_{sample}" for sample in range(self.flow_samples)
        )

    @property
    def width(self) -> int:
        return 2 * self.top_k + 2 + self.flow_samples

    def expected_shape(self, requests: int) -> tuple[int, int, int, int]:
        if requests < 1:
            raise ValueError("token-focus-token-flow request count must be positive")
        return (requests, self.layers, self.local_heads, self.width)

    @property
    def contract_hash(self) -> str:
        document = {
            "observable": "exact_token_topk_with_exact_attention_flow_tokens",
            "schema_version": self.schema_version,
            "top_k": self.top_k,
            "layers": self.layers,
            "local_heads": self.local_heads,
            "flow_samples": self.flow_samples,
            "dtype": "float32",
            "fields": self.fields,
            "ordering": "probability_desc_token_id_asc_then_flow_samples",
            "flow_distribution": "categorical_full_attention_over_tokens",
            "flow_construction": (
                "native_split_mass_sample_then_bounded_conditional_token_replay"
            ),
        }
        encoded = json.dumps(
            document, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return "sha256:" + hashlib.sha256(encoded).hexdigest()


def validate_attnsketch_token_focus_provenance(
    provenance: AttnSketchCaptureProvenance,
    schema: (
        AttnSketchTokenFocusSchema
        | AttnSketchTokenFocusFlowSchema
        | AttnSketchTokenFocusTokenFlowSchema
    ),
) -> None:
    """Reject a capture whose declared metrics are not this exact schema."""

    if provenance.metrics != schema.fields:
        raise ValueError("capture metrics do not match token-focus field ordering")
    if provenance.query_contract_hash != schema.contract_hash:
        raise ValueError("capture query contract does not match token-focus schema")


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


@dataclass(frozen=True)
class AttnSketchPageGroup:
    """One exported mass cell and the physical pages it represents."""

    telemetry_index: int
    key_start: int
    key_end: int
    logical_pages: tuple[int, ...]
    physical_pages: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.telemetry_index < 0 or self.key_start < 0:
            raise ValueError("page-group indices and key offsets must be non-negative")
        if self.key_end <= self.key_start:
            raise ValueError("page-group key range must be nonempty")
        if not self.logical_pages or len(self.logical_pages) != len(
            self.physical_pages
        ):
            raise ValueError("page groups require matched logical/physical pages")
        if tuple(sorted(set(self.logical_pages))) != self.logical_pages:
            raise ValueError("logical page IDs must be sorted and unique")
        if any(page < 0 for page in self.logical_pages + self.physical_pages):
            raise ValueError("page IDs must be non-negative")


@dataclass(frozen=True)
class AttnSketchPageMapping:
    """Immutable sidecar mapping for a page/superpage mass tensor."""

    request_id: str
    request_slot: int
    request_table_epoch: int
    page_table_epoch: int
    runtime_page_tokens: int
    telemetry_page_tokens: int
    valid_tokens: int
    groups: tuple[AttnSketchPageGroup, ...]

    def __post_init__(self) -> None:
        if not self.request_id:
            raise ValueError("request_id must not be empty")
        if min(
            self.request_slot,
            self.request_table_epoch,
            self.page_table_epoch,
        ) < 0:
            raise ValueError("request slots and epochs must be non-negative")
        if min(
            self.runtime_page_tokens,
            self.telemetry_page_tokens,
            self.valid_tokens,
        ) < 1:
            raise ValueError("page and valid-token sizes must be positive")
        if not self.groups:
            raise ValueError("page mapping must contain at least one group")
        if self.groups[0].key_start != 0 or self.groups[-1].key_end != self.valid_tokens:
            raise ValueError("page mapping must cover the complete valid key range")
        for index, group in enumerate(self.groups):
            if group.telemetry_index != index:
                raise ValueError("page-group indices must be contiguous")
            if index and self.groups[index - 1].key_end != group.key_start:
                raise ValueError("page groups must be contiguous")

    def canonical_bytes(self) -> bytes:
        document = {
            "request_id": self.request_id,
            "request_slot": self.request_slot,
            "request_table_epoch": self.request_table_epoch,
            "page_table_epoch": self.page_table_epoch,
            "runtime_page_tokens": self.runtime_page_tokens,
            "telemetry_page_tokens": self.telemetry_page_tokens,
            "valid_tokens": self.valid_tokens,
            "groups": [
                {
                    "telemetry_index": group.telemetry_index,
                    "key_start": group.key_start,
                    "key_end": group.key_end,
                    "logical_pages": list(group.logical_pages),
                    "physical_pages": list(group.physical_pages),
                }
                for group in self.groups
            ],
        }
        return json.dumps(
            document, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")

    @property
    def mapping_hash(self) -> str:
        return _PAGE_MAPPING_PREFIX + hashlib.sha256(self.canonical_bytes()).hexdigest()

    @classmethod
    def from_export_fields(
        cls, fields: Mapping[str, object]
    ) -> "AttnSketchPageMapping":
        required = {
            "request_id",
            "request_slot",
            "request_table_epoch",
            "page_table_epoch",
            "runtime_page_tokens",
            "telemetry_page_tokens",
            "valid_tokens",
            "physical_page_groups",
            "page_mapping_hash",
        }
        missing = required.difference(fields)
        if missing:
            raise ValueError(f"page export fields are missing {sorted(missing)}")
        physical_groups = fields["physical_page_groups"]
        if not isinstance(physical_groups, list):
            raise ValueError("physical_page_groups must be a list")
        runtime_page_tokens = int(fields["runtime_page_tokens"])
        telemetry_page_tokens = int(fields["telemetry_page_tokens"])
        valid_tokens = int(fields["valid_tokens"])
        groups: list[AttnSketchPageGroup] = []
        for index, physical_pages_value in enumerate(physical_groups):
            if not isinstance(physical_pages_value, list):
                raise ValueError("each physical page group must be a list")
            key_start = index * telemetry_page_tokens
            key_end = min(valid_tokens, key_start + telemetry_page_tokens)
            logical_start = key_start // runtime_page_tokens
            logical_end = (key_end + runtime_page_tokens - 1) // runtime_page_tokens
            groups.append(
                AttnSketchPageGroup(
                    telemetry_index=index,
                    key_start=key_start,
                    key_end=key_end,
                    logical_pages=tuple(range(logical_start, logical_end)),
                    physical_pages=tuple(int(page) for page in physical_pages_value),
                )
            )
        mapping = cls(
            request_id=str(fields["request_id"]),
            request_slot=int(fields["request_slot"]),
            request_table_epoch=int(fields["request_table_epoch"]),
            page_table_epoch=int(fields["page_table_epoch"]),
            runtime_page_tokens=runtime_page_tokens,
            telemetry_page_tokens=telemetry_page_tokens,
            valid_tokens=valid_tokens,
            groups=tuple(groups),
        )
        if mapping.mapping_hash != fields["page_mapping_hash"]:
            raise ValueError("page mapping hash does not match export fields")
        return mapping


class AttnSketchPageMappingRegistry:
    """Digest-indexed sidecar registry for allocator-sensitive page mappings."""

    def __init__(self, entries: Iterable[AttnSketchPageMapping] = ()) -> None:
        self._entries: dict[str, AttnSketchPageMapping] = {}
        for entry in entries:
            self.register(entry)

    def register(self, mapping: AttnSketchPageMapping) -> str:
        mapping_hash = mapping.mapping_hash
        existing = self._entries.get(mapping_hash)
        if existing is not None and existing != mapping:
            raise ValueError("AttnSketch page-mapping digest collision")
        self._entries[mapping_hash] = mapping
        return mapping_hash

    def resolve(self, mapping_hash: str) -> AttnSketchPageMapping:
        _require_digest(mapping_hash, "page_mapping_hash")
        try:
            mapping = self._entries[mapping_hash]
        except KeyError as exc:
            raise ValueError("unknown AttnSketch page mapping") from exc
        if mapping.mapping_hash != mapping_hash:
            raise ValueError("AttnSketch page mapping failed digest verification")
        return mapping


def validate_attnsketch_page_mapping_identity(
    *,
    mapping_hash: str,
    registry: AttnSketchPageMappingRegistry,
    expected_request_id: str,
    expected_request_slot: int,
    expected_request_table_epoch: int,
    expected_page_table_epoch: int,
) -> AttnSketchPageMapping:
    """Reject a page tensor whose allocator snapshot no longer matches."""

    mapping = registry.resolve(mapping_hash)
    observed = (
        mapping.request_id,
        mapping.request_slot,
        mapping.request_table_epoch,
        mapping.page_table_epoch,
    )
    expected = (
        expected_request_id,
        expected_request_slot,
        expected_request_table_epoch,
        expected_page_table_epoch,
    )
    if observed != expected:
        raise ValueError("AttnSketch page mapping attribution mismatch")
    return mapping


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
    "ATTSKETCH_TOKEN_FOCUS_FLOW_SCHEMA_VERSION",
    "ATTSKETCH_TOKEN_FOCUS_SCHEMA_VERSION",
    "AttnSketchCaptureProvenance",
    "AttnSketchPageGroup",
    "AttnSketchPageMapping",
    "AttnSketchPageMappingRegistry",
    "AttnSketchProvenanceRegistry",
    "AttnSketchRequestBinding",
    "AttnSketchTokenFocusSchema",
    "AttnSketchTokenFocusFlowSchema",
    "validate_attnsketch_export_identity",
    "validate_attnsketch_page_mapping_identity",
    "validate_attnsketch_token_focus_provenance",
]
