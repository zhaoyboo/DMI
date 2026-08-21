"""Contract tests for transporting materialized AttnSketch summary records."""

from __future__ import annotations

import json

from torch import nn

from monitoring.attnsketch_pipeline import (
    AttnSketchCaptureProvenance,
    AttnSketchProvenanceRegistry,
    AttnSketchRequestBinding,
    validate_attnsketch_export_identity,
)
from monitoring.ring_transport import (
    HOOK_TYPE_ATTN_SUMMARY,
    HookSpec,
    ModelShapeConfig,
    _compute_hook_shape,
    _id_by_short,
)
from monitoring.selection import select_hook_specs


def _digest(byte: str) -> str:
    return "sha256:" + byte * 64


def _provenance() -> AttnSketchCaptureProvenance:
    return AttnSketchCaptureProvenance(
        kernel_fingerprint=_digest("1"),
        artifact_hash=_digest("2"),
        manifest_version="fa2-v2.8.3-sm80-r1",
        score_semantics_hash=_digest("3"),
        semantic_mapping_version="fa2-v2.8.3-summary-v1",
        layout_hash=_digest("4"),
        query_contract_hash=_digest("5"),
        metrics=("prefix_mass", "suffix_mass", "collision"),
    )


def _config(**changes) -> ModelShapeConfig:
    values = dict(
        hidden_dim=4096,
        num_heads=32,
        num_kv_heads=8,
        head_dim=128,
        dtype=None,
        tp_size=2,
        attn_summary_width=3,
    )
    values.update(changes)
    return ModelShapeConfig(**values)


def test_attnsketch_hook_is_a_first_class_native_hook():
    assert _id_by_short["attn_summary"] == HOOK_TYPE_ATTN_SUMMARY


def test_attnsketch_summary_shape_preserves_rows_heads_and_metric_width():
    cfg = _config()
    assert _compute_hook_shape(
        HOOK_TYPE_ATTN_SUMMARY, cfg, batch=2, q_len=5, kv_dim=1024
    ) == [2, 5, 16, 3]
    assert _compute_hook_shape(
        HOOK_TYPE_ATTN_SUMMARY, cfg, batch=0, q_len=7, kv_dim=1024
    ) == [7, 16, 3]


def test_missing_summary_width_disables_the_transport_hook():
    spec = HookSpec(HOOK_TYPE_ATTN_SUMMARY, nn.Identity(), layer_no=4)
    assert select_hook_specs([spec], "attn_summary", _config()) == [spec]
    assert select_hook_specs(
        [spec], "attn_summary", _config(attn_summary_width=0)
    ) == []


def test_attnsketch_provenance_registry_round_trip(tmp_path):
    provenance = _provenance()
    registry = AttnSketchProvenanceRegistry([provenance])
    path = tmp_path / "attnsketch-provenance.json"
    registry.write(path)
    loaded = AttnSketchProvenanceRegistry.read(path)
    assert loaded.resolve(provenance.capture_id) == provenance


def test_attnsketch_registry_rejects_tampered_capture(tmp_path):
    provenance = _provenance()
    registry = AttnSketchProvenanceRegistry([provenance])
    path = tmp_path / "attnsketch-provenance.json"
    registry.write(path)
    document = json.loads(path.read_text())
    capture_id = provenance.capture_id
    document["captures"][capture_id]["layout_hash"] = _digest("a")
    path.write_text(json.dumps(document))
    try:
        AttnSketchProvenanceRegistry.read(path)
    except ValueError as exc:
        assert "does not match" in str(exc)
    else:
        raise AssertionError("tampered registry entry was accepted")


def test_attnsketch_request_binding_round_trip_and_strict_validation():
    provenance = _provenance()
    registry = AttnSketchProvenanceRegistry([provenance])
    binding = AttnSketchRequestBinding("tenant:req/7", 31, 47)
    encoded = binding.encode()
    assert AttnSketchRequestBinding.decode(encoded) == binding
    validate_attnsketch_export_identity(
        model_id=provenance.capture_id,
        request_id=encoded,
        registry=registry,
        expected_provenance=provenance,
        expected_request_id="tenant:req/7",
        expected_request_table_epoch=31,
        expected_page_table_epoch=47,
    )


def test_attnsketch_export_rejects_stale_page_epoch():
    provenance = _provenance()
    registry = AttnSketchProvenanceRegistry([provenance])
    encoded = AttnSketchRequestBinding("req", 3, 4).encode()
    try:
        validate_attnsketch_export_identity(
            model_id=provenance.capture_id,
            request_id=encoded,
            registry=registry,
            expected_provenance=provenance,
            expected_request_id="req",
            expected_request_table_epoch=3,
            expected_page_table_epoch=5,
        )
    except ValueError as exc:
        assert "attribution mismatch" in str(exc)
    else:
        raise AssertionError("stale page-table epoch was accepted")
