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

Page/superpage mass tensors additionally register an
`AttnSketchPageMapping`. `AttnSketchPageMapping.from_export_fields()` consumes
the sidecar emitted by `SelectedTileMassBatch.bind_page_table()`, recomputes
its digest, and rejects tampered physical IDs. Consumers resolve that digest
through `AttnSketchPageMappingRegistry` and call
`validate_attnsketch_page_mapping_identity`; a changed request slot,
request-table epoch, or page-table epoch is a hard error. The large mapping is
stored once as sidecar metadata rather than repeated in every layer/head
payload.

This branch does not yet install a FlashAttention model adapter that fires the
hook. That adapter belongs to AttnSketch-lib and must be pinned to a validated
kernel manifest.

## Request-scoped layer/head aggregates

AttnSketch may apply a declared linear reduction across all selected layers
and heads on the GPU before transport. That output has a different semantic
axis and uses a separate hook:

```text
activation name: attn.attnsketch_scope_summary
short name:      attn_scope_summary
shape:           [requests, scope_summary_width]
```

Set `ModelShapeConfig.attn_scope_summary_width` and activate the hook
explicitly. `RingTransport.submit_attnsketch_scope_summary()` accepts only a
preallocated contiguous CUDA FP32 tensor on the ring device. It rejects an
inactive hook, missing request context, implicit dtype/layout conversion, and
any mismatch between the number of tensor rows and request records. Packed
and batched drain paths both slice this tensor by request position rather than
by scheduled-token offsets.

The low-level method above remains useful for isolated transport benchmarks.
Production AttnSketch adapters use
`submit_attnsketch_bound_scope_summary()`: it additionally requires the exact
capture ID plus `(raw request ID, request-table epoch, page-table epoch)` for
every row, decodes the active DMI request bindings, and rejects the tensor
before ring publication if any identity differs. The public batch adapter also
returns one `BoundPageMassSummary` per request so the physical-page mapping can
be registered and validated by the consumer.

The canonical 32-layer/8-head decode experiment reduces 256 semantic rows to
one 128-entry (512 B) vector, then traverses the real GPU ring and asynchronous
D2H drain. It is a raw `SUM`; consumers divide by the declared normalization
denominator only when they need a probability distribution. The hook never
silently substitutes the per-layer/per-head `attn_summary` semantics.

A separate batch-safe producer/reducer has also traversed this hook with
1/4/8 request rows. A 200-pair admission sweep accepts 8/16/32-layer groups
only when both paired p50 and p99 overhead are below 5%; shorter groups fail
closed. At eight requests and 32 modeled layers it publishes 4 KiB instead of
1 MiB of separate layer/head vectors, while retaining distinct request rows
and bitwise O/LSE identity against an unmodified pinned FA2 control. This
validates request-row, capture, and allocator-epoch transport contracts; the
CUDA producer still uses contiguous fixed-length KV and therefore is not yet
a paged-attention serving result.

The transport is metric-agnostic. The pinned positional adapter uses width 2
with metrics `("argmax_logical_token_f32", "p_max")`. Its token index is
transported as FP32 only because the manifested sequence length is below
`2**24`; the capture layout hash and metric tuple make that convention
explicit. DMI does not convert, reconstruct, or reinterpret either field.

## K-independent deferred-replay capsules

AttnSketch also defines a request-scoped opaque transport for deferred replay:

```text
activation name: attn.attnsketch_replay_capsule
short name:      attn_replay_capsule
shape:           [requests, fixed_capsule_bytes]
dtype:           uint8
```

Set `ModelShapeConfig.attn_replay_capsule_bytes` and select the hook
explicitly. A zero width disables it. The capsule is not a Top-K result. It is
a versioned, checksum-protected AttnSketch record containing the captured Q
bytes, producer bounds, inherited LSE, request/page epochs and kernel semantic
fingerprints required by a deferred worker. Consequently, K, a token
probability threshold, or a region aggregate can be chosen after transport
without changing or recapturing the inference graph.

DMI deliberately treats this tensor as opaque bytes. It neither interprets
the replay contract nor reads KV. The graph-facing
`submit_attnsketch_replay_capsule()` accepts only a preallocated contiguous
CUDA `uint8` tensor with exactly one row per request. The off-path connector
`submit_attnsketch_replay_capsule_cpu()` accepts an already copied contiguous
CPU tensor and traverses the same native metadata, sink and attribution path;
it requires the capsule hook to be the sole active hook for that queued
metadata batch. Neither method performs an implicit cast, reshape or copy.

The first AttnSketch integration drains its existing nonblocking device
capture ring on a service thread, encodes one complete decode generation, and
then submits the CPU capsule through DMI. A failed submission poisons that
publisher and becomes an explicit observation gap; it must not leave DMI's
metadata FIFO aligned by guesswork. The receiving worker authenticates the
capsule and joins its request/page epochs against a byte-preserving KV-offload
lease before replay. Compressed or requantized KV is outside this exact
contract and fails closed.

This hook has native-ring unit evidence for both CUDA-producer and already-
offloaded CPU-producer entry points. It does not yet constitute a live vLLM
serving performance claim, a remote deployment admission, or evidence that
DMI transport is free. Those are separate integration measurements.
