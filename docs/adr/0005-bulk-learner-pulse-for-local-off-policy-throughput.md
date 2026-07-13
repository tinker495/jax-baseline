# Bulk Learner Pulse for Local Off-Policy Throughput

The local Q-Net and DPG families currently execute multiple optimizer updates as repeated Python
calls that each sample replay, transfer a mini-batch to the accelerator, invoke a JIT train step,
and optionally update replay priorities. We will replace that default path on a separate comparison
branch with a **Bulk learner pulse** for TD7, the local DPG family, and the local DQN/Q-Net family:
sample multiple mini-batches in one host operation, transfer them once with a leading update
dimension, execute the per-mini-batch optimizer updates in order on device, and return aligned
priority updates as one batch. The goal is **Device transfer amortization** — higher training
throughput from fewer CPU/accelerator boundaries — not changing sample efficiency semantics.

## Decisions

- **Full branch conversion, not an opt-in flag.** The implementation branch's default local
  off-policy learner path uses bulk pulses. The explicit comparison is branch-to-branch:
  `main` keeps the legacy Python loop and the bulk branch carries the new default.
- **Preserve mini-batch update semantics.** A bulk pulse represents K existing optimizer updates,
  not one large-batch optimizer update. Each mini-update keeps its own ordered
  `train_steps_count` / step value so policy-delay and target-network update cadence remain
  equivalent to the legacy loop.
- **Apply beyond checkpointing.** TD7-style checkpointing is a strong producer of larger pulses,
  but the abstraction belongs to `train(steps, gradient_steps)` for all local Q-Net/DPG paths,
  including non-checkpoint loops with `gradient_steps > 1`.
- **Own the abstraction in the family training lifecycles.** `RolloutEngine` and
  `CheckpointTrainPulse` already expose the right boundary by passing an update count into
  `train(steps, gradient_steps)`. The Q-Net and DPG training lifecycles own chunking, bulk replay
  sampling, delayed priority writes, and report aggregation; concrete algorithms provide bulk
  train hooks where their JIT update shape differs.
- **Defer PER feedback inside a pulse.** Prioritized replay priorities are collected during the
  on-device bulk pulse and written back to replay once per chunk. This deliberately gives up the
  legacy loop's immediate priority feedback between mini-updates in exchange for fewer host/replay
  round trips.
- **Cap each bulk chunk and bound bulk shapes with buckets.** A `max_bulk_updates_per_pulse`
  safety limit creates deterministic candidate bucket sizes: `cap`, `cap // 2`, `cap // 4`, and
  so on while the value is at least `2`, plus a terminal `2` bucket when halving an odd cap would
  otherwise skip it. A cached planner first minimizes host-side training calls, then scalar
  leftovers, then prefers a larger first chunk. This preserves the total requested update count
  while bounding memory growth, compile-shape churn, and priority staleness.
- **Route only unsupported leftovers through scalar updates.** Single-update pulses, `cap == 1`,
  and missing bulk hooks use the legacy scalar mini-batch path. Valid multi-update pulses may use
  bulk even when they are smaller than the cap, for example `cap=8, gradient_steps=2 -> [2]`.
  Unsupported final leftovers remain scalar: `cap=8, gradient_steps=7 -> [4, 2] + scalar 1`,
  while the terminal bucket avoids that fallback for `cap=3, gradient_steps=5 -> [3, 2]`.
- **Weight mixed bulk reports by update count.** Each concrete bulk train hook reports
  `update_count=len(contexts)` so family-level aggregation weights `8`, `4`, `2`, and scalar chunks
  by the number of optimizer updates they represent rather than by the number of Python reports.

## Consequences

The bulk branch changes the default execution shape and therefore must be compared explicitly
against `main` for throughput and learning parity. Correctness tests should lock the observable
training semantics — ordered step counters, total optimizer update count, checkpoint residual
consumption, and aligned PER priority writes — before relying on performance measurements.
