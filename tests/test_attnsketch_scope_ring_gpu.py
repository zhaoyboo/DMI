"""GPU integration test for the request-scoped AttnSketch DMI hook."""

from __future__ import annotations

import pytest
import torch

from monitoring import _native_engine
from monitoring.attnsketch_pipeline import AttnSketchRequestBinding
from monitoring.ring_transport import (
    HOOK_TYPE_ATTN_SCOPE_SUMMARY,
    HOOK_TYPE_ATTN_TOKEN_FOCUS,
    HookSpec,
    ModelShapeConfig,
    RingTransport,
    activate,
    deactivate,
)


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required"
)


@pytest.mark.parametrize("requests", (1, 4))
def test_exact_token_focus_traverses_real_gpu_ring_without_mutating_source(
    requests: int,
) -> None:
    layers, heads, top_k = 3, 8, 4
    width = 2 * top_k + 2
    config = _native_engine.RingConfig()
    config.task_ring_entries = 1024
    config.payload_ring_bytes = 1024 * 1024
    config.pinned_staging_bytes = 1024 * 1024
    config.drain_poll_timeout_us = 50
    config.drain_flush_timeout_us = 50
    sink = _native_engine.InMemoryRingSink()
    engine = _native_engine.RingEngine(config, sink)
    engine.init()
    engine.start()
    transport = RingTransport(engine)
    transport.set_model_cfg(
        ModelShapeConfig(
            hidden_dim=1024,
            num_heads=heads,
            num_kv_heads=heads,
            head_dim=128,
            dtype=torch.float16,
            attn_token_focus_layers=layers,
            attn_token_focus_top_k=top_k,
        )
    )
    transport.set_step_context(
        model_id="attnsketch-token-focus-ring-test",
        req_ids=[f"request-{index}" for index in range(requests)],
        token_ranges=[(4095, 4096)] * requests,
        dim0_offsets=list(range(requests)),
        kv_offsets=[0] * requests,
        flattened=False,
    )
    transport._active_specs = [
        HookSpec(
            HOOK_TYPE_ATTN_TOKEN_FOCUS,
            None,
            layer_no=-1,
            dtype=torch.float32,
        )
    ]
    source = torch.zeros(
        requests, layers, heads, width, device="cuda", dtype=torch.float32
    )
    for rank in range(top_k):
        source[..., 2 * rank] = float(rank)
        source[..., 2 * rank + 1] = 0.125
    source[..., -2] = 0.5
    source[..., -1] = 0.5
    expected = source.clone()

    activate(transport)
    try:
        assert engine.prepare_step(source.nbytes, 1) != 2
        transport.pre_push_all_metas(
            batch=requests,
            q_len=1,
            kv_dim=4096,
            logits_to_keep=1,
        )
        transport.submit_attnsketch_token_focus(source)
        engine.notify_drain()
        engine.flush_and_wait()
        assert torch.equal(source, expected)
        rows = sink.rows()
        assert len(rows) == requests
        for request_index, row in enumerate(rows):
            assert row["model_id"] == "attnsketch-token-focus-ring-test"
            assert row["request_id"] == f"request-{request_index}"
            assert row["activation_name"] == "attn.attnsketch_token_focus"
            assert row["layer_no"] == -1
            assert row["shard_rank"] == 0
            assert (row["start_token"], row["end_token"]) == (4095, 4096)
            assert torch.equal(
                row["tensor"], expected[request_index : request_index + 1].cpu()
            )
    finally:
        engine.stop()
        deactivate()


@pytest.mark.parametrize("requests", (1, 4))
def test_scope_summary_traverses_real_gpu_ring_without_mutating_source(
    requests: int,
) -> None:
    config = _native_engine.RingConfig()
    config.task_ring_entries = 1024
    config.payload_ring_bytes = 1024 * 1024
    config.pinned_staging_bytes = 1024 * 1024
    config.drain_poll_timeout_us = 50
    config.drain_flush_timeout_us = 50
    engine = _native_engine.RingEngine(config, None)
    engine.init()
    engine.start()
    transport = RingTransport(engine)
    transport.set_model_cfg(
        ModelShapeConfig(
            hidden_dim=1024,
            num_heads=8,
            num_kv_heads=8,
            head_dim=128,
            dtype=torch.float16,
            attn_scope_summary_width=128,
        )
    )
    transport.set_step_context(
        model_id="attnsketch-ring-test",
        req_ids=[f"request-{index}" for index in range(requests)],
        token_ranges=[(4095, 4096)] * requests,
        dim0_offsets=list(range(requests)),
        kv_offsets=[0] * requests,
        flattened=False,
    )
    transport._active_specs = [
        HookSpec(
            HOOK_TYPE_ATTN_SCOPE_SUMMARY,
            None,
            layer_no=-1,
            dtype=torch.float32,
        )
    ]
    source = torch.arange(
        requests * 128, device="cuda", dtype=torch.float32
    ).view(requests, 128)
    expected = source.clone()

    activate(transport)
    try:
        assert engine.prepare_step(source.nbytes, 1) != 2
        transport.pre_push_all_metas(
            batch=requests,
            q_len=1,
            kv_dim=4096,
            logits_to_keep=1,
        )
        transport.submit_attnsketch_scope_summary(source)
        engine.notify_drain()
        engine.flush_and_wait()
        assert torch.equal(source, expected)
    finally:
        engine.stop()
        deactivate()


def test_cached_scope_metadata_traverses_ring_and_fails_on_context_drift() -> None:
    config = _native_engine.RingConfig()
    config.task_ring_entries = 1024
    config.payload_ring_bytes = 1024 * 1024
    config.pinned_staging_bytes = 1024 * 1024
    engine = _native_engine.RingEngine(config, None)
    engine.init()
    engine.start()
    transport = RingTransport(engine)
    transport.set_model_cfg(
        ModelShapeConfig(
            hidden_dim=1024,
            num_heads=8,
            num_kv_heads=2,
            head_dim=128,
            dtype=torch.float16,
            attn_scope_summary_width=128,
        )
    )
    capture_id = "attnsketch:v1:cached-meta-test"
    expected_requests = (("request-0", 23, 17),)
    transport.set_step_context(
        model_id=capture_id,
        req_ids=[AttnSketchRequestBinding(*expected_requests[0]).encode()],
        token_ranges=[(4095, 4096)],
        dim0_offsets=[0],
        kv_offsets=[0],
        flattened=False,
    )
    transport._active_specs = [
        HookSpec(
            HOOK_TYPE_ATTN_SCOPE_SUMMARY,
            None,
            layer_no=-1,
            dtype=torch.float32,
        )
    ]
    template = transport.register_attnsketch_bound_scope_meta_template(
        capture_id=capture_id,
        expected_requests=expected_requests,
        batch=1,
        q_len=1,
        kv_dim=4096,
        logits_to_keep=1,
    )
    source = torch.arange(128, device="cuda", dtype=torch.float32).view(1, 128)

    activate(transport)
    try:
        assert engine.prepare_step(source.nbytes, 1) != 2
        template.push()
        transport.submit_attnsketch_scope_summary(source)
        engine.notify_drain()
        engine.flush_and_wait()

        rebound_requests = (("request-1", 24, 18),)
        transport.set_step_context(
            model_id=capture_id,
            req_ids=[AttnSketchRequestBinding(*rebound_requests[0]).encode()],
            token_ranges=[(8191, 8192)],
            dim0_offsets=[0],
            kv_offsets=[32],
            flattened=False,
        )
        template.rebind(expected_requests=rebound_requests)
        assert engine.prepare_step(source.nbytes, 1) != 2
        template.push()
        transport.submit_attnsketch_scope_summary(source)
        engine.notify_drain()
        engine.flush_and_wait()
        assert template.expected_requests == rebound_requests

        transport._current_token_ranges = [(8190, 8191)]
        with pytest.raises(ValueError, match="token ranges"):
            template.push()
    finally:
        engine.stop()
        deactivate()
