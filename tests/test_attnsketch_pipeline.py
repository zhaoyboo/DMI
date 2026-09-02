"""Contract tests for transporting materialized AttnSketch summary records."""

from __future__ import annotations

import json

import pytest
import torch
from torch import nn

from monitoring.attnsketch_pipeline import (
    AttnSketchCaptureProvenance,
    AttnSketchPageGroup,
    AttnSketchPageMapping,
    AttnSketchPageMappingRegistry,
    AttnSketchProvenanceRegistry,
    AttnSketchRequestBinding,
    AttnSketchTokenFocusFlowSchema,
    AttnSketchTokenFocusSchema,
    AttnSketchTokenFocusTokenFlowSchema,
    validate_attnsketch_export_identity,
    validate_attnsketch_page_mapping_identity,
    validate_attnsketch_token_focus_provenance,
)
from monitoring.ring_transport import (
    HOOK_TYPE_ATTN_SUMMARY,
    HOOK_TYPE_ATTN_SCOPE_SUMMARY,
    HOOK_TYPE_ATTN_TOKEN_FOCUS,
    HOOK_TYPE_ATTN_REPLAY_CAPSULE,
    HookRowBasis,
    HookSpec,
    ModelShapeConfig,
    _compute_hook_shape,
    _id_by_short,
    hook_row_basis,
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
        attn_scope_summary_width=128,
        attn_token_focus_layers=28,
        attn_token_focus_top_k=4,
        attn_replay_capsule_bytes=4096,
    )
    values.update(changes)
    return ModelShapeConfig(**values)


def test_attnsketch_hook_is_a_first_class_native_hook():
    assert _id_by_short["attn_summary"] == HOOK_TYPE_ATTN_SUMMARY
    assert (
        _id_by_short["attn_scope_summary"]
        == HOOK_TYPE_ATTN_SCOPE_SUMMARY
    )
    assert _id_by_short["attn_token_focus"] == HOOK_TYPE_ATTN_TOKEN_FOCUS
    assert (
        _id_by_short["attn_replay_capsule"]
        == HOOK_TYPE_ATTN_REPLAY_CAPSULE
    )


def test_attnsketch_summary_shape_preserves_rows_heads_and_metric_width():
    cfg = _config()
    assert _compute_hook_shape(
        HOOK_TYPE_ATTN_SUMMARY, cfg, batch=2, q_len=5, kv_dim=1024
    ) == [2, 5, 16, 3]
    assert _compute_hook_shape(
        HOOK_TYPE_ATTN_SUMMARY, cfg, batch=0, q_len=7, kv_dim=1024
    ) == [7, 16, 3]


def test_attnsketch_scope_summary_is_request_scoped_without_head_axis():
    cfg = _config()
    assert _compute_hook_shape(
        HOOK_TYPE_ATTN_SCOPE_SUMMARY,
        cfg,
        batch=2,
        q_len=5,
        kv_dim=1024,
    ) == [2, 128]
    assert _compute_hook_shape(
        HOOK_TYPE_ATTN_SCOPE_SUMMARY,
        cfg,
        batch=0,
        q_len=7,
        kv_dim=1024,
        logits_to_keep=3,
    ) == [3, 128]
    assert hook_row_basis(HOOK_TYPE_ATTN_SCOPE_SUMMARY) is HookRowBasis.REQUEST_ROWS


def test_attnsketch_replay_capsule_is_fixed_width_request_scoped_bytes():
    cfg = _config()
    assert _compute_hook_shape(
        HOOK_TYPE_ATTN_REPLAY_CAPSULE,
        cfg,
        batch=2,
        q_len=1,
        kv_dim=4096,
    ) == [2, 4096]
    assert _compute_hook_shape(
        HOOK_TYPE_ATTN_REPLAY_CAPSULE,
        cfg,
        batch=0,
        q_len=7,
        kv_dim=4096,
        logits_to_keep=3,
    ) == [3, 4096]
    assert (
        hook_row_basis(HOOK_TYPE_ATTN_REPLAY_CAPSULE)
        is HookRowBasis.REQUEST_ROWS
    )


def test_attnsketch_token_focus_has_explicit_request_layer_head_rank_shape():
    cfg = _config()
    assert _compute_hook_shape(
        HOOK_TYPE_ATTN_TOKEN_FOCUS,
        cfg,
        batch=2,
        q_len=1,
        kv_dim=4096,
    ) == [2, 28, 16, 10]
    assert _compute_hook_shape(
        HOOK_TYPE_ATTN_TOKEN_FOCUS,
        cfg,
        batch=0,
        q_len=7,
        kv_dim=4096,
        logits_to_keep=3,
    ) == [3, 28, 16, 10]
    assert hook_row_basis(HOOK_TYPE_ATTN_TOKEN_FOCUS) is HookRowBasis.REQUEST_ROWS


def test_attnsketch_token_focus_schema_binds_field_order_and_query_contract():
    schema = AttnSketchTokenFocusSchema(top_k=4, layers=28, local_heads=16)
    assert schema.expected_shape(2) == (2, 28, 16, 10)
    assert schema.fields == (
        "token_id_0",
        "probability_0",
        "token_id_1",
        "probability_1",
        "token_id_2",
        "probability_2",
        "token_id_3",
        "probability_3",
        "coverage",
        "tail_mass",
    )
    provenance = AttnSketchCaptureProvenance(
        kernel_fingerprint=_digest("1"),
        artifact_hash=_digest("2"),
        manifest_version="fa2-v2.8.3-sm89-token-focus-r1",
        score_semantics_hash=_digest("3"),
        semantic_mapping_version="attnsketch.exact-token-focus.v1",
        layout_hash=_digest("4"),
        query_contract_hash=schema.contract_hash,
        metrics=schema.fields,
    )
    validate_attnsketch_token_focus_provenance(provenance, schema)

    wrong = AttnSketchCaptureProvenance(
        kernel_fingerprint=_digest("1"),
        artifact_hash=_digest("2"),
        manifest_version="fa2-v2.8.3-sm89-token-focus-r1",
        score_semantics_hash=_digest("3"),
        semantic_mapping_version="attnsketch.exact-token-focus.v1",
        layout_hash=_digest("4"),
        query_contract_hash=schema.contract_hash,
        metrics=tuple(reversed(schema.fields)),
    )
    with pytest.raises(ValueError, match="field ordering"):
        validate_attnsketch_token_focus_provenance(wrong, schema)


def test_attnsketch_token_flow_schema_appends_exact_token_ids():
    schema = AttnSketchTokenFocusTokenFlowSchema(
        top_k=2,
        layers=3,
        local_heads=8,
        flow_samples=2,
    )
    assert schema.expected_shape(1) == (1, 3, 8, 8)
    assert schema.fields == (
        "token_id_0",
        "probability_0",
        "token_id_1",
        "probability_1",
        "coverage",
        "tail_mass",
        "flow_token_id_0",
        "flow_token_id_1",
    )
    assert schema.contract_hash != AttnSketchTokenFocusFlowSchema(
        top_k=2,
        layers=3,
        local_heads=8,
        flow_samples=2,
        flow_region_count=8,
    ).contract_hash


def test_attnsketch_token_focus_flow_schema_appends_versioned_region_ids():
    schema = AttnSketchTokenFocusFlowSchema(
        top_k=4,
        layers=28,
        local_heads=16,
        flow_samples=2,
        flow_region_count=8,
    )
    assert schema.expected_shape(1) == (1, 28, 16, 12)
    assert schema.fields[-4:] == (
        "coverage",
        "tail_mass",
        "flow_region_id_0",
        "flow_region_id_1",
    )
    cfg = _config(attn_token_focus_width=schema.width)
    assert _compute_hook_shape(
        HOOK_TYPE_ATTN_TOKEN_FOCUS,
        cfg,
        batch=1,
        q_len=1,
        kv_dim=1024,
    ) == [1, 28, 16, 12]
    provenance = AttnSketchCaptureProvenance(
        kernel_fingerprint=_digest("1"),
        artifact_hash=_digest("2"),
        manifest_version="fa2-v2.8.3-sm89-token-focus-flow-r1",
        score_semantics_hash=_digest("3"),
        semantic_mapping_version="attnsketch.exact-token-focus-flow.v1",
        layout_hash=_digest("4"),
        query_contract_hash=schema.contract_hash,
        metrics=schema.fields,
    )
    validate_attnsketch_token_focus_provenance(provenance, schema)

    with pytest.raises(ValueError, match="at most 32 regions"):
        AttnSketchTokenFocusFlowSchema(
            top_k=4,
            layers=28,
            local_heads=16,
            flow_samples=1,
            flow_region_count=33,
        )


def test_attnsketch_scope_summary_preflight_rejects_request_row_mismatch():
    from monitoring.ring_transport import RingTransport

    transport = RingTransport.__new__(RingTransport)
    transport.null_offload = False
    transport._model_cfg = _config()
    transport._active_specs = [
        HookSpec(HOOK_TYPE_ATTN_SCOPE_SUMMARY, None, layer_no=-1)
    ]
    transport._current_model_id = "model"
    transport._current_req_ids = ["request-0", "request-1"]
    transport._current_token_ranges = [(0, 1), (0, 1)]
    transport._current_dim0_offsets = [0, 1]
    transport._current_kv_offsets = [0, 0]

    try:
        transport.pre_push_all_metas(
            batch=1,
            q_len=1,
            kv_dim=128,
            logits_to_keep=1,
        )
    except ValueError as exc:
        assert "request-row mismatch" in str(exc)
    else:
        raise AssertionError("request-scope metadata accepted the wrong row count")


def test_bound_scope_submit_requires_capture_and_allocator_epochs():
    from monitoring.ring_transport import RingTransport

    provenance = _provenance()
    encoded = AttnSketchRequestBinding("request-0", 7, 19).encode()
    transport = RingTransport.__new__(RingTransport)
    transport._current_model_id = provenance.capture_id
    transport._current_req_ids = [encoded]
    submitted = []
    transport.submit_attnsketch_scope_summary = submitted.append
    transport.submit_attnsketch_token_focus = submitted.append

    payload = object()
    transport.submit_attnsketch_bound_scope_summary(
        payload,
        capture_id=provenance.capture_id,
        expected_requests=(("request-0", 7, 19),),
    )
    assert submitted == [payload]
    transport.submit_attnsketch_bound_token_focus(
        payload,
        capture_id=provenance.capture_id,
        expected_requests=(("request-0", 7, 19),),
    )
    assert submitted == [payload, payload]
    transport.validate_attnsketch_bound_scope(
        capture_id=provenance.capture_id,
        expected_requests=(("request-0", 7, 19),),
    )
    assert submitted == [payload, payload]

    try:
        transport.submit_attnsketch_bound_scope_summary(
            payload,
            capture_id=provenance.capture_id,
            expected_requests=(("request-0", 7, 20),),
        )
    except ValueError as exc:
        assert "epochs" in str(exc)
    else:
        raise AssertionError("stale page-table epoch was submitted")

    try:
        transport.submit_attnsketch_bound_scope_summary(
            payload,
            capture_id="attnsketch:v1:" + "f" * 64,
            expected_requests=(("request-0", 7, 19),),
        )
    except ValueError as exc:
        assert "capture" in str(exc)
    else:
        raise AssertionError("wrong capture ID was submitted")


def test_bound_scope_meta_preflight_validates_before_pushing():
    from monitoring.ring_transport import RingTransport

    provenance = _provenance()
    encoded = AttnSketchRequestBinding("request-0", 7, 19).encode()
    transport = RingTransport.__new__(RingTransport)
    transport._current_model_id = provenance.capture_id
    transport._current_req_ids = [encoded]
    pushed = []
    transport.pre_push_all_metas = lambda **kwargs: pushed.append(kwargs)

    transport.pre_push_attnsketch_bound_scope_metas(
        capture_id=provenance.capture_id,
        expected_requests=(("request-0", 7, 19),),
        batch=1,
        q_len=1,
        kv_dim=32,
    )
    assert pushed == [
        {"batch": 1, "q_len": 1, "kv_dim": 32, "logits_to_keep": 0}
    ]
    transport.pre_push_attnsketch_bound_token_focus_metas(
        capture_id=provenance.capture_id,
        expected_requests=(("request-0", 7, 19),),
        batch=1,
        q_len=1,
        kv_dim=32,
    )
    assert pushed[-1] == {
        "batch": 1,
        "q_len": 1,
        "kv_dim": 32,
        "logits_to_keep": 0,
    }

    transport._current_req_ids = [
        AttnSketchRequestBinding("request-0", 7, 20).encode()
    ]
    with pytest.raises(ValueError, match="epochs"):
        transport.pre_push_attnsketch_bound_scope_metas(
            capture_id=provenance.capture_id,
            expected_requests=(("request-0", 7, 19),),
            batch=1,
            q_len=1,
            kv_dim=32,
        )
    assert len(pushed) == 2


def test_bound_scope_meta_template_rebinds_dynamic_attribution_atomically():
    from monitoring.ring_transport import RingTransport

    class FakeTemplateEngine:
        def __init__(self):
            self.replacements = []
            self.pushes = []

        def register_step_template(self, *args):
            self.registered = args
            return 17

        def replace_step_template(self, *args):
            self.replacements.append(args)

        def push_step_template(self, template_id):
            self.pushes.append(template_id)

    provenance = _provenance()
    old = (("request-old", 7, 19),)
    new = (("request-new", 8, 21),)
    engine = FakeTemplateEngine()
    transport = RingTransport.__new__(RingTransport)
    transport._ring_engine = engine
    transport._model_cfg = _config()
    transport._active_specs = [
        HookSpec(
            HOOK_TYPE_ATTN_SCOPE_SUMMARY,
            None,
            layer_no=-1,
            dtype=torch.float32,
        )
    ]
    transport._current_model_id = provenance.capture_id
    transport._current_tp_rank = 0
    transport._current_dp_rank = 0
    transport._current_ep_rank = 0
    transport._current_pp_rank = 0
    transport._current_flattened = False
    transport._current_req_ids = [AttnSketchRequestBinding(*old[0]).encode()]
    transport._current_token_ranges = [(4095, 4096)]
    transport._current_dim0_offsets = [0]
    transport._current_kv_offsets = [0]

    template = transport.register_attnsketch_bound_scope_meta_template(
        capture_id=provenance.capture_id,
        expected_requests=old,
        batch=1,
        q_len=1,
        kv_dim=4096,
        logits_to_keep=1,
    )

    transport._current_req_ids = [AttnSketchRequestBinding(*new[0]).encode()]
    transport._current_token_ranges = [(8191, 8192)]
    transport._current_dim0_offsets = [0]
    transport._current_kv_offsets = [32]
    template.rebind(expected_requests=new)
    template.push()

    assert len(engine.replacements) == 1
    replacement = engine.replacements[0]
    assert replacement[0] == 17
    assert replacement[-4] == transport._current_req_ids
    assert replacement[-3] == [(8191, 8192)]
    assert replacement[-1] == [32]
    assert template.expected_requests == new
    assert template.expected_token_ranges == ((8191, 8192),)
    assert engine.pushes == [17]


def test_bound_token_focus_template_rebinds_and_pushes_in_one_native_call():
    from monitoring.ring_transport import RingTransport

    class FakeTemplateEngine:
        def register_step_template(self, *args):
            self.registered = args
            return 23

        def rebind_and_push_step_template(self, *args):
            self.rebound_and_pushed = args

    provenance = _provenance()
    old = (("request-old", 7, 19),)
    new = (("request-new", 8, 21),)
    engine = FakeTemplateEngine()
    transport = RingTransport.__new__(RingTransport)
    transport._ring_engine = engine
    transport._model_cfg = _config()
    transport._active_specs = [
        HookSpec(
            HOOK_TYPE_ATTN_TOKEN_FOCUS,
            None,
            layer_no=-1,
            dtype=torch.float32,
        )
    ]
    transport._current_model_id = provenance.capture_id
    transport._current_tp_rank = 0
    transport._current_dp_rank = 0
    transport._current_ep_rank = 0
    transport._current_pp_rank = 0
    transport._current_flattened = False
    transport._current_req_ids = [AttnSketchRequestBinding(*old[0]).encode()]
    transport._current_token_ranges = [(4095, 4096)]
    transport._current_dim0_offsets = [0]
    transport._current_kv_offsets = [0]

    template = transport.register_attnsketch_bound_token_focus_meta_template(
        capture_id=provenance.capture_id,
        expected_requests=old,
        batch=1,
        q_len=1,
        kv_dim=4096,
        logits_to_keep=1,
    )
    assert engine.registered[0] == [HOOK_TYPE_ATTN_TOKEN_FOCUS]
    assert engine.registered[2] == [[1, 28, 16, 10]]

    transport._current_req_ids = [AttnSketchRequestBinding(*new[0]).encode()]
    transport._current_token_ranges = [(8191, 8192)]
    transport._current_kv_offsets = [32]
    template.rebind_and_push(expected_requests=new)

    call = engine.rebound_and_pushed
    assert call[0] == 23
    assert call[-4] == transport._current_req_ids
    assert call[-3] == [(8191, 8192)]
    assert call[-1] == [32]
    assert template.expected_hook_type == HOOK_TYPE_ATTN_TOKEN_FOCUS
    assert template.expected_requests == new
    assert template.expected_token_ranges == ((8191, 8192),)


def test_bound_token_focus_template_static_publish_uses_one_native_call():
    from monitoring.ring_transport import RingTransport

    class FakeTemplateEngine:
        def register_step_template(self, *args):
            self.registered = args
            return 29

        def publish_step_template_static(self, *args):
            self.published = args
            return 0

    provenance = _provenance()
    old = (("request-old", 7, 19),)
    new = (("request-new", 8, 21),)
    engine = FakeTemplateEngine()
    transport = RingTransport.__new__(RingTransport)
    transport._ring_engine = engine
    transport._model_cfg = _config()
    transport._active_specs = [
        HookSpec(
            HOOK_TYPE_ATTN_TOKEN_FOCUS,
            None,
            layer_no=-1,
            dtype=torch.float32,
        )
    ]
    transport._current_model_id = provenance.capture_id
    transport._current_tp_rank = 0
    transport._current_dp_rank = 0
    transport._current_ep_rank = 0
    transport._current_pp_rank = 0
    transport._current_flattened = False
    transport._current_req_ids = [AttnSketchRequestBinding(*old[0]).encode()]
    transport._current_token_ranges = [(4095, 4096)]
    transport._current_dim0_offsets = [0]
    transport._current_kv_offsets = [0]

    template = transport.register_attnsketch_bound_token_focus_meta_template(
        capture_id=provenance.capture_id,
        expected_requests=old,
        batch=1,
        q_len=1,
        kv_dim=4096,
        logits_to_keep=1,
    )
    tensor = torch.zeros(1, 28, 16, 10)
    transport._current_req_ids = [AttnSketchRequestBinding(*new[0]).encode()]
    transport._current_token_ranges = [(8191, 8192)]
    transport._current_kv_offsets = [32]

    assert template.publish_static(tensor, expected_requests=new) == 0
    call = engine.published
    assert call[0] == 29
    assert call[-2] is tensor
    assert call[-1] == HOOK_TYPE_ATTN_TOKEN_FOCUS
    assert call[-6] == transport._current_req_ids
    assert call[-5] == [(8191, 8192)]
    assert call[-3] == [32]
    assert template.expected_requests == new
    assert template.expected_token_ranges == ((8191, 8192),)


def test_bound_token_focus_template_static_publish_does_not_adopt_rejected_context():
    from monitoring.ring_transport import RingTransport

    class CapacityRejectingEngine:
        def register_step_template(self, *args):
            return 31

        def publish_step_template_static(self, *args):
            self.published = args
            return 2

    provenance = _provenance()
    old = (("request-old", 7, 19),)
    new = (("request-new", 8, 21),)
    engine = CapacityRejectingEngine()
    transport = RingTransport.__new__(RingTransport)
    transport._ring_engine = engine
    transport._model_cfg = _config()
    transport._active_specs = [
        HookSpec(
            HOOK_TYPE_ATTN_TOKEN_FOCUS,
            None,
            layer_no=-1,
            dtype=torch.float32,
        )
    ]
    transport._current_model_id = provenance.capture_id
    transport._current_tp_rank = 0
    transport._current_dp_rank = 0
    transport._current_ep_rank = 0
    transport._current_pp_rank = 0
    transport._current_flattened = False
    transport._current_req_ids = [AttnSketchRequestBinding(*old[0]).encode()]
    transport._current_token_ranges = [(4095, 4096)]
    transport._current_dim0_offsets = [0]
    transport._current_kv_offsets = [0]

    template = transport.register_attnsketch_bound_token_focus_meta_template(
        capture_id=provenance.capture_id,
        expected_requests=old,
        batch=1,
        q_len=1,
        kv_dim=4096,
        logits_to_keep=1,
    )
    transport._current_req_ids = [AttnSketchRequestBinding(*new[0]).encode()]
    transport._current_token_ranges = [(8191, 8192)]
    transport._current_kv_offsets = [32]

    assert template.publish_static(
        torch.zeros(1, 28, 16, 10), expected_requests=new
    ) == 2
    assert template.expected_requests == old
    assert template.expected_token_ranges == ((4095, 4096),)


def test_bound_scope_expected_encoding_cache_is_bounded_under_churn():
    from monitoring.ring_transport import RingTransport

    provenance = _provenance()
    transport = RingTransport.__new__(RingTransport)
    transport._current_model_id = provenance.capture_id
    transport._attnsketch_expected_encoding_cache = {}
    for epoch in range(256):
        expected = ((f"request-{epoch}", epoch, epoch),)
        transport._current_req_ids = [
            AttnSketchRequestBinding(*expected[0]).encode()
        ]
        transport.validate_attnsketch_bound_scope(
            capture_id=provenance.capture_id,
            expected_requests=expected,
        )
    assert len(transport._attnsketch_expected_encoding_cache) <= 128


def test_missing_summary_width_disables_the_transport_hook():
    spec = HookSpec(HOOK_TYPE_ATTN_SUMMARY, nn.Identity(), layer_no=4)
    assert select_hook_specs([spec], "attn_summary", _config()) == [spec]
    assert select_hook_specs(
        [spec], "attn_summary", _config(attn_summary_width=0)
    ) == []
    scope_spec = HookSpec(HOOK_TYPE_ATTN_SCOPE_SUMMARY, nn.Identity(), layer_no=-1)
    assert select_hook_specs(
        [scope_spec], "attn_scope_summary", _config()
    ) == [scope_spec]
    assert select_hook_specs(
        [scope_spec],
        "attn_scope_summary",
        _config(attn_scope_summary_width=0),
    ) == []
    focus_spec = HookSpec(HOOK_TYPE_ATTN_TOKEN_FOCUS, nn.Identity(), layer_no=-1)
    assert select_hook_specs(
        [focus_spec], "attn_token_focus", _config()
    ) == [focus_spec]
    assert select_hook_specs(
        [focus_spec],
        "attn_token_focus",
        _config(attn_token_focus_top_k=0),
    ) == []
    capsule_spec = HookSpec(
        HOOK_TYPE_ATTN_REPLAY_CAPSULE,
        nn.Identity(),
        layer_no=-1,
        dtype=torch.uint8,
    )
    assert select_hook_specs(
        [capsule_spec], "attn_replay_capsule", _config()
    ) == [capsule_spec]
    assert select_hook_specs(
        [capsule_spec],
        "attn_replay_capsule",
        _config(attn_replay_capsule_bytes=0),
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


def test_argmax_token_metrics_preserve_request_and_page_epochs():
    provenance = AttnSketchCaptureProvenance(
        kernel_fingerprint=_digest("1"),
        artifact_hash=_digest("1"),
        manifest_version="fa2-v2.8.3-sm80-argmax-token-pmax-r1",
        score_semantics_hash=_digest("3"),
        semantic_mapping_version="fa2-argmax-token-pmax-v1",
        layout_hash=_digest("4"),
        query_contract_hash=_digest("5"),
        metrics=("argmax_logical_token_f32", "p_max"),
    )
    registry = AttnSketchProvenanceRegistry([provenance])
    encoded = AttnSketchRequestBinding("req-token", 9, 12).encode()
    validate_attnsketch_export_identity(
        model_id=provenance.capture_id,
        request_id=encoded,
        registry=registry,
        expected_provenance=provenance,
        expected_request_id="req-token",
        expected_request_table_epoch=9,
        expected_page_table_epoch=12,
    )


def _page_mapping() -> AttnSketchPageMapping:
    return AttnSketchPageMapping(
        request_id="request-3",
        request_slot=3,
        request_table_epoch=11,
        page_table_epoch=7,
        runtime_page_tokens=16,
        telemetry_page_tokens=32,
        valid_tokens=128,
        groups=(
            AttnSketchPageGroup(0, 0, 32, (0, 1), (100, 101)),
            AttnSketchPageGroup(1, 32, 64, (2, 3), (102, 103)),
            AttnSketchPageGroup(2, 64, 96, (4, 5), (104, 105)),
            AttnSketchPageGroup(3, 96, 128, (6, 7), (106, 107)),
        ),
    )


def test_page_mapping_reconstructs_parent_export_fields_and_validates_epochs():
    mapping = _page_mapping()
    assert mapping.mapping_hash == (
        "sha256:941de768ab6a9454f09ec7c65fd549f3f33e33cb3f5f7849d785c252cf651111"
    )
    fields = {
        "request_id": mapping.request_id,
        "request_slot": mapping.request_slot,
        "request_table_epoch": mapping.request_table_epoch,
        "page_table_epoch": mapping.page_table_epoch,
        "runtime_page_tokens": mapping.runtime_page_tokens,
        "telemetry_page_tokens": mapping.telemetry_page_tokens,
        "valid_tokens": mapping.valid_tokens,
        "physical_page_groups": [
            [100, 101],
            [102, 103],
            [104, 105],
            [106, 107],
        ],
        "page_mapping_hash": mapping.mapping_hash,
    }
    reconstructed = AttnSketchPageMapping.from_export_fields(fields)
    assert reconstructed == mapping
    registry = AttnSketchPageMappingRegistry([reconstructed])
    assert validate_attnsketch_page_mapping_identity(
        mapping_hash=mapping.mapping_hash,
        registry=registry,
        expected_request_id="request-3",
        expected_request_slot=3,
        expected_request_table_epoch=11,
        expected_page_table_epoch=7,
    ) == mapping


def test_page_mapping_rejects_stale_epoch_and_tampered_physical_page():
    mapping = _page_mapping()
    registry = AttnSketchPageMappingRegistry([mapping])
    try:
        validate_attnsketch_page_mapping_identity(
            mapping_hash=mapping.mapping_hash,
            registry=registry,
            expected_request_id="request-3",
            expected_request_slot=3,
            expected_request_table_epoch=11,
            expected_page_table_epoch=8,
        )
    except ValueError as exc:
        assert "attribution mismatch" in str(exc)
    else:
        raise AssertionError("stale page-table epoch was accepted")

    fields = {
        "request_id": mapping.request_id,
        "request_slot": mapping.request_slot,
        "request_table_epoch": mapping.request_table_epoch,
        "page_table_epoch": mapping.page_table_epoch,
        "runtime_page_tokens": mapping.runtime_page_tokens,
        "telemetry_page_tokens": mapping.telemetry_page_tokens,
        "valid_tokens": mapping.valid_tokens,
        "physical_page_groups": [
            [999, 101],
            [102, 103],
            [104, 105],
            [106, 107],
        ],
        "page_mapping_hash": mapping.mapping_hash,
    }
    try:
        AttnSketchPageMapping.from_export_fields(fields)
    except ValueError as exc:
        assert "hash does not match" in str(exc)
    else:
        raise AssertionError("tampered physical page was accepted")
