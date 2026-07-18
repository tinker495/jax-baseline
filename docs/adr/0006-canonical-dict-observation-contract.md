# Canonical Dict Observation Contract

## Context

Environment backends expose observations as arrays, tuples, or nested mappings. The previous core
surface represented modalities as positional lists, which discarded semantic keys and forced
rollout, replay, normalization, and model code to infer ordering. Named mapping backends make that
ambiguity especially costly.

## Decision

- `Observation` is `dict[str, array]` throughout the Algorithm Core and repo-local replay/model
  adapters.
- `EnvInfo.observation_space` is `dict[str, list[int]]` with the same keys.
- Environment adapters perform the only backend conversion. A non-mapping observation becomes
  `{"obs": value}`; nested mappings are flattened to stable dotted keys.
- Replay fields preserve semantic keys instead of positional `obs0`, `obs1`, and model preprocessing
  consumes observations by key.
- Legacy list-shaped running-statistics checkpoint state may be read during deserialization, but new
  runtime state and serialized state use dictionaries.

## Consequences

Observation keys remain stable from environment reset through rollout, replay sampling,
normalization, and model input. Adding a backend no longer changes core observation structure.
Internal callers that still pass arrays or positional lists fail at the contract boundary and must
be migrated rather than receiving an implicit compatibility conversion.
