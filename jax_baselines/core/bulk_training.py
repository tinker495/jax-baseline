"""Small helpers for chunked replay training pulses."""


def bulk_train_hook(agent):
    if not getattr(agent, "supports_bulk_training", False):
        return None
    return getattr(agent, "_train_on_bulk", None)


def bulk_chunk_size(agent):
    max_chunk = int(agent.max_bulk_updates_per_pulse)
    if max_chunk <= 0:
        raise ValueError("max_bulk_updates_per_pulse must be greater than 0")
    return max_chunk


def uses_bulk_pulse(agent, gradient_steps):
    if bulk_train_hook(agent) is None or gradient_steps <= 1:
        return False
    max_chunk = bulk_chunk_size(agent)
    return max_chunk > 1 and gradient_steps >= max_chunk


def make_train_contexts(agent, context_type, steps, chunk_size, **kwargs):
    contexts = []
    for _ in range(chunk_size):
        agent.train_steps_count += 1
        contexts.append(
            context_type(
                steps=steps,
                train_steps_count=agent.train_steps_count,
                **kwargs,
            )
        )
    return tuple(contexts)


def reshape_bulk_batch(data, chunk_size, batch_size):
    return {key: reshape_bulk_value(value, chunk_size, batch_size) for key, value in data.items()}


def normalize_bulk_weights(data):
    weights = data.get("weights")
    if weights is None:
        return data
    data = dict(data)
    data["weights"] = normalize_bulk_weight_value(weights)
    return data


def normalize_bulk_weight_value(value):
    shape = getattr(value, "shape", None)
    if shape is None or len(shape) < 2:
        return value
    return value / value.max(axis=1, keepdims=True)


def reshape_bulk_value(value, chunk_size, batch_size):
    if isinstance(value, list):
        return [reshape_bulk_value(item, chunk_size, batch_size) for item in value]
    if isinstance(value, tuple):
        return tuple(reshape_bulk_value(item, chunk_size, batch_size) for item in value)
    shape = getattr(value, "shape", None)
    if shape is None or len(shape) == 0:
        return value
    if len(shape) >= 2 and shape[0] == chunk_size and shape[1] == batch_size:
        return value
    expected_flat = chunk_size * batch_size
    if shape[0] != expected_flat:
        return value
    return value.reshape((chunk_size, batch_size, *shape[1:]))


def iter_bulk_batches(data, contexts):
    """Yield per-mini-update slices from a bulk replay sample."""
    for index in range(len(contexts)):
        yield {key: slice_bulk_value(value, index) for key, value in data.items()}


def slice_bulk_value(value, index):
    if isinstance(value, list):
        return [slice_bulk_value(item, index) for item in value]
    if isinstance(value, tuple):
        return tuple(slice_bulk_value(item, index) for item in value)
    shape = getattr(value, "shape", None)
    if shape is None or len(shape) == 0:
        return value
    return value[index]


def flatten_bulk_batch(data):
    """Collapse ``(chunk, batch, ...)`` bulk data into ``(chunk * batch, ...)``."""
    return {key: flatten_bulk_value(value) for key, value in data.items()}


def flatten_bulk_value(value):
    if isinstance(value, list):
        return [flatten_bulk_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(flatten_bulk_value(item) for item in value)
    shape = getattr(value, "shape", None)
    if shape is None or len(shape) < 2:
        return value
    return value.reshape((shape[0] * shape[1], *shape[2:]))


def flatten_priority_values(values):
    shape = getattr(values, "shape", None)
    if shape is None or len(shape) <= 1:
        return values
    return values.reshape((-1,))
