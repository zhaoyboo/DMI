# AttnSketch summary transport

The `attnsketch-pipeline` branch adds one first-class hook for compact attention
summaries that have already been produced and validated by AttnSketch-lib:

```text
activation name: attn.attnsketch_summary
short name:      attn_summary
shape:           [batch, query, local_heads, summary_width]
                 or [packed_queries, local_heads, summary_width]
```

Set `ModelShapeConfig.attn_summary_width` to the number of FP32 metrics and
select `attn_summary`. A zero width disables the hook. DMI transports the
metric tensor; it does not derive scores or summaries itself.

Every capture run must register an `AttnSketchCaptureProvenance` from
`monitoring.attnsketch_pipeline` and use its full SHA-256 `capture_id` as the
DMI `model_id`. Each request ID must be wrapped in an
`AttnSketchRequestBinding`, which adds the immutable request-table and
page-table epochs. Consumers resolve the persisted provenance registry and
call `validate_attnsketch_export_identity` before interpreting the payload.
Unknown captures, malformed IDs, changed semantics, and stale allocator epochs
fail closed. This binding uses DMI's existing per-record string metadata, so it
does not repeat large fingerprints in every query/head payload.

This branch does not yet install a FlashAttention model adapter that fires the
hook. That adapter belongs to AttnSketch-lib and must be pinned to a validated
kernel manifest.

The transport is metric-agnostic. The pinned positional adapter uses width 2
with metrics `("argmax_logical_token_f32", "p_max")`. Its token index is
transported as FP32 only because the manifested sequence length is below
`2**24`; the capture layout hash and metric tuple make that convention
explicit. DMI does not convert, reconstruct, or reinterpret either field.
