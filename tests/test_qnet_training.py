import ast
from pathlib import Path

import numpy as np
import pytest

from jax_baselines.core.bulk_training import (
    flatten_bulk_batch,
    iter_bulk_batches,
    iter_bulk_chunk_sizes,
)
from jax_baselines.DQN.base_class import Q_Network_Family
from jax_baselines.DQN.dqn import DQN
from jax_baselines.DQN.training import (
    QNetTrainContext,
    QNetTrainingLifecycle,
    QNetTrainReport,
    QNetTrainResult,
)
from jax_baselines.SPR.spr import SPR


class FakeReplayBuffer:
    def __init__(self):
        self.sample_calls = []
        self.priority_update_calls = []

    def sample(self, batch_size, beta=None):
        self.sample_calls.append((batch_size, beta))
        return {
            "obses": np.array([1.0]),
            "indexes": np.array([3]),
        }

    def update_priorities(self, indexes, priorities):
        self.priority_update_calls.append((indexes, priorities))


def _aggregate_bulk_results(agent, results):
    report = agent._aggregate_train_reports([result.report for result in results])
    report.update_count = sum(result.report.update_count for result in results)
    priorities = [result.replay_priorities for result in results]
    replay_priorities = None
    if all(priority is not None for priority in priorities):
        replay_priorities = np.stack(priorities)
    return QNetTrainResult(report=report, replay_priorities=replay_priorities)


class FakeBulkReplayBuffer(FakeReplayBuffer):
    def sample(self, batch_size, beta=None):
        self.sample_calls.append((batch_size, beta))
        update_count = batch_size // 4
        indexes = np.arange(update_count * 4).reshape(update_count, 4)
        weights = np.arange(1, update_count * 4 + 1, dtype=np.float32)
        return {
            "obses": np.ones((update_count, 4)),
            "indexes": indexes,
            "weights": weights,
        }


class FakeLoggerRun:
    def __init__(self):
        self.metrics = []
        self.histograms = []

    def log_metric(self, name, value, steps):
        self.metrics.append((name, value, steps))

    def log_histogram(self, name, value, steps):
        self.histograms.append((name, value, steps))


class FakeAgent:
    def __init__(self):
        self.replay_buffer = FakeReplayBuffer()
        self.batch_size = 4
        self.prioritized_replay = True
        self.prioritized_replay_beta0 = 0.4
        self.train_steps_count = 0
        self.logger_run = FakeLoggerRun()
        self._last_log_step = 0
        self.log_interval = 5
        self.contexts = []
        self.batches = []

    def _sample_batch(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        return self.replay_buffer.sample(batch_size, self.prioritized_replay_beta0)

    def _train_on_batch(self, data, context):
        self.contexts.append(context)
        self.batches.append(data)
        return QNetTrainResult.from_values(
            loss=float(context.train_steps_count),
            target=20.0 + context.train_steps_count,
            replay_priorities=np.array([context.train_steps_count]),
            metrics={"loss/extra": 100.0 + context.train_steps_count},
            histograms={"loss/tau": np.array([0.1, 0.9])},
        )

    def _aggregate_train_reports(self, reports):
        return reports[-1]


def test_qnet_training_lifecycle_samples_updates_priorities_and_logs():
    agent = FakeAgent()
    lifecycle = QNetTrainingLifecycle(agent)

    loss = lifecycle.train(steps=10, gradient_steps=2)

    assert loss == 2.0
    assert agent.train_steps_count == 2
    assert agent.replay_buffer.sample_calls == [(4, 0.4), (4, 0.4)]
    assert agent.contexts[0].steps == 10
    assert agent.contexts[0].train_steps_count == 1
    assert len(agent.replay_buffer.priority_update_calls) == 2
    assert np.array_equal(agent.replay_buffer.priority_update_calls[-1][0], np.array([3]))
    assert np.array_equal(agent.replay_buffer.priority_update_calls[-1][1], np.array([2]))
    assert agent.logger_run.metrics == [
        ("loss/extra", 102.0, 10),
        ("loss/qloss", 2.0, 10),
        ("loss/targets", 22.0, 10),
    ]
    assert len(agent.logger_run.histograms) == 1
    assert agent.logger_run.histograms[0][0] == "loss/tau"


def test_qnet_train_report_keeps_replay_priorities_out_of_report_surface():
    report = QNetTrainReport(loss=1.0, target=2.0)
    result = QNetTrainResult(report=report, replay_priorities=np.array([3.0]))

    assert not hasattr(report, "replay_priorities")
    assert np.array_equal(result.replay_priorities, np.array([3.0]))


class FakeMissingPrioritiesAgent(FakeAgent):
    def _train_on_batch(self, data, context):
        return QNetTrainResult.from_values(loss=1.0, target=2.0)


def test_qnet_training_lifecycle_requires_priorities_when_per_enabled():
    agent = FakeMissingPrioritiesAgent()
    lifecycle = QNetTrainingLifecycle(agent)

    with pytest.raises(ValueError, match="replay_priorities"):
        lifecycle.train(steps=10, gradient_steps=1)


class FakeReportOnlyNonPerAgent(FakeMissingPrioritiesAgent):
    def __init__(self):
        super().__init__()
        self.prioritized_replay = False

    def _train_on_batch(self, data, context):
        return QNetTrainReport(loss=1.0, target=2.0)


def test_qnet_training_lifecycle_accepts_report_only_when_per_disabled():
    agent = FakeReportOnlyNonPerAgent()
    lifecycle = QNetTrainingLifecycle(agent)

    loss = lifecycle.train(steps=10, gradient_steps=1)

    assert loss == 1.0
    assert agent.replay_buffer.priority_update_calls == []


class FakePulseAgent(FakeAgent):
    supports_bulk_training = True

    def __init__(self):
        super().__init__()
        self.gradient_steps = 2
        self.max_bulk_updates_per_pulse = 2
        self.replay_buffer = FakeBulkReplayBuffer()
        self.bulk_contexts = []
        self.bulk_batches = []

    def _train_on_batch(self, data, context):
        self.contexts.append(context)
        self.batches.append(data)
        return QNetTrainResult.from_values(
            loss=float(context.train_steps_count),
            target=25.0,
            replay_priorities=np.repeat(context.train_steps_count, self.batch_size),
        )

    def _train_on_bulk(self, data, contexts):
        assert data["indexes"] is not None
        assert all(context.gradient_steps == len(contexts) for context in contexts)
        self.bulk_batches.append(data)
        self.bulk_contexts.append(contexts)
        results = [
            self._train_on_batch(batch, context)
            for batch, context in zip(iter_bulk_batches(data, contexts), contexts)
        ]
        return _aggregate_bulk_results(self, results)


def test_qnet_training_lifecycle_handles_priority_update_calls_for_chunked_pulses():
    agent = FakePulseAgent()
    lifecycle = QNetTrainingLifecycle(agent)

    loss = lifecycle.train(steps=10, gradient_steps=5)

    assert loss == 5.0
    assert agent.train_steps_count == 5
    assert agent.replay_buffer.sample_calls == [(8, 0.4), (8, 0.4), (4, 0.4)]
    assert [[ctx.train_steps_count for ctx in chunk] for chunk in agent.bulk_contexts] == [
        [1, 2],
        [3, 4],
    ]
    assert np.array_equal(
        agent.bulk_batches[0]["weights"],
        np.array([[0.25, 0.5, 0.75, 1.0], [0.625, 0.75, 0.875, 1.0]], dtype=np.float32),
    )
    assert [ctx.train_steps_count for ctx in agent.contexts] == [1, 2, 3, 4, 5]
    assert len(agent.replay_buffer.priority_update_calls) == 3
    assert np.array_equal(
        agent.replay_buffer.priority_update_calls[0][0],
        np.arange(8),
    )
    assert np.array_equal(
        agent.replay_buffer.priority_update_calls[0][1],
        np.array([1, 1, 1, 1, 2, 2, 2, 2]),
    )
    assert np.array_equal(
        agent.replay_buffer.priority_update_calls[-1][0],
        np.arange(4),
    )
    assert np.array_equal(
        agent.replay_buffer.priority_update_calls[-1][1],
        np.array([5, 5, 5, 5]),
    )


class FakeMissingBulkHookPulseAgent(FakeAgent):
    def __init__(self):
        super().__init__()
        self.max_bulk_updates_per_pulse = 2


def test_qnet_training_lifecycle_uses_scalar_path_without_bulk_hook():
    agent = FakeMissingBulkHookPulseAgent()
    lifecycle = QNetTrainingLifecycle(agent)

    lifecycle.train(steps=10, gradient_steps=2)

    assert agent.replay_buffer.sample_calls == [(4, 0.4), (4, 0.4)]
    assert [ctx.train_steps_count for ctx in agent.contexts] == [1, 2]


@pytest.mark.parametrize(
    ("max_chunk", "gradient_steps", "expected"),
    (
        (8, 2, (2,)),
        (8, 7, (4, 2)),
        (8, 13, (8, 4)),
        (3, 5, (3,)),
        (5, 9, (5, 2, 2)),
        (5, 8, (5, 2)),
        (7, 10, (7, 3)),
        (7, 9, (7,)),
    ),
)
def test_bulk_chunk_schedule_uses_bounded_greedy_buckets(max_chunk, gradient_steps, expected):
    assert tuple(iter_bulk_chunk_sizes(gradient_steps, max_chunk)) == expected


def test_qnet_training_lifecycle_requires_explicit_bulk_marker():
    agent = FakePulseAgent()
    agent.supports_bulk_training = False
    lifecycle = QNetTrainingLifecycle(agent)

    lifecycle.train(steps=10, gradient_steps=2)

    assert agent.bulk_contexts == []
    assert [ctx.train_steps_count for ctx in agent.contexts] == [1, 2]


def test_qnet_training_lifecycle_uses_scalar_tail_after_full_bulk_chunks():
    agent = FakePulseAgent()
    lifecycle = QNetTrainingLifecycle(agent)

    lifecycle.train(steps=10, gradient_steps=3)

    assert agent.replay_buffer.sample_calls == [(8, 0.4), (4, 0.4)]
    assert [[ctx.train_steps_count for ctx in chunk] for chunk in agent.bulk_contexts] == [
        [1, 2],
    ]
    assert [ctx.train_steps_count for ctx in agent.contexts] == [1, 2, 3]


def test_qnet_training_lifecycle_uses_scalar_path_for_single_pulse_update():
    agent = FakePulseAgent()
    lifecycle = QNetTrainingLifecycle(agent)

    lifecycle.train(steps=10, gradient_steps=1)

    assert agent.replay_buffer.sample_calls == [(4, 0.4)]
    assert agent.bulk_contexts == []
    assert [ctx.train_steps_count for ctx in agent.contexts] == [1]


def test_qnet_training_lifecycle_uses_bucket_when_pulse_is_smaller_than_cap():
    agent = FakePulseAgent()
    agent.max_bulk_updates_per_pulse = 8
    lifecycle = QNetTrainingLifecycle(agent)

    lifecycle.train(steps=10, gradient_steps=2)

    assert agent.replay_buffer.sample_calls == [(8, 0.4)]
    assert [[ctx.train_steps_count for ctx in chunk] for chunk in agent.bulk_contexts] == [[1, 2]]
    assert [[ctx.gradient_steps for ctx in chunk] for chunk in agent.bulk_contexts] == [[2, 2]]
    assert [ctx.train_steps_count for ctx in agent.contexts] == [1, 2]


def test_qnet_training_lifecycle_uses_smaller_buckets_before_scalar_tail():
    agent = FakePulseAgent()
    agent.max_bulk_updates_per_pulse = 8
    lifecycle = QNetTrainingLifecycle(agent)

    lifecycle.train(steps=10, gradient_steps=7)

    assert agent.replay_buffer.sample_calls == [(16, 0.4), (8, 0.4), (4, 0.4)]
    assert [[ctx.train_steps_count for ctx in chunk] for chunk in agent.bulk_contexts] == [
        [1, 2, 3, 4],
        [5, 6],
    ]
    assert [[ctx.gradient_steps for ctx in chunk] for chunk in agent.bulk_contexts] == [
        [4, 4, 4, 4],
        [2, 2],
    ]
    assert [ctx.train_steps_count for ctx in agent.contexts] == [1, 2, 3, 4, 5, 6, 7]


def test_qnet_training_lifecycle_uses_scalar_tail_larger_than_one_after_full_bulk_chunk():
    agent = FakePulseAgent()
    agent.max_bulk_updates_per_pulse = 3
    lifecycle = QNetTrainingLifecycle(agent)

    lifecycle.train(steps=10, gradient_steps=5)

    assert agent.replay_buffer.sample_calls == [(12, 0.4), (4, 0.4), (4, 0.4)]
    assert [[ctx.train_steps_count for ctx in chunk] for chunk in agent.bulk_contexts] == [
        [1, 2, 3],
    ]
    assert [ctx.train_steps_count for ctx in agent.contexts] == [1, 2, 3, 4, 5]


def test_qnet_training_lifecycle_uses_scalar_path_when_cap_is_one():
    agent = FakePulseAgent()
    agent.max_bulk_updates_per_pulse = 1
    lifecycle = QNetTrainingLifecycle(agent)

    lifecycle.train(steps=10, gradient_steps=3)

    assert agent.replay_buffer.sample_calls == [(4, 0.4), (4, 0.4), (4, 0.4)]
    assert agent.bulk_contexts == []
    assert [ctx.train_steps_count for ctx in agent.contexts] == [1, 2, 3]


def test_qnet_training_lifecycle_keeps_non_positive_bulk_cap_invalid():
    agent = FakePulseAgent()
    agent.max_bulk_updates_per_pulse = 0
    lifecycle = QNetTrainingLifecycle(agent)

    with pytest.raises(ValueError, match="max_bulk_updates_per_pulse"):
        lifecycle.train(steps=10, gradient_steps=2)


def test_spr_lineage_single_update_uses_scalar_path_without_bulk_hook():
    agent = SPR.__new__(SPR)
    agent.max_bulk_updates_per_pulse = 8
    agent.replay_buffer = FakeReplayBuffer()
    agent.batch_size = 4
    agent.prioritized_replay = True
    agent.prioritized_replay_beta0 = 0.4
    agent.train_steps_count = 0
    agent.logger_run = None
    agent._last_log_step = 0
    agent.log_interval = 5
    contexts = []

    def sample_batch(batch_size=None):
        batch_size = agent.batch_size if batch_size is None else batch_size
        return agent.replay_buffer.sample(batch_size, agent.prioritized_replay_beta0)

    def train_on_batch(data, context):
        contexts.append(context)
        return QNetTrainResult.from_values(
            loss=float(context.train_steps_count),
            replay_priorities=np.array([context.train_steps_count]),
        )

    def train_on_bulk(data, contexts):
        raise AssertionError("single-update SPR pulse must not call _train_on_bulk")

    agent._sample_batch = sample_batch
    agent._train_on_batch = train_on_batch
    agent._train_on_bulk = train_on_bulk
    agent._aggregate_train_reports = lambda reports: reports[-1]
    lifecycle = QNetTrainingLifecycle(agent)

    loss = lifecycle.train(steps=10, gradient_steps=1)

    assert loss == 1.0
    assert agent.replay_buffer.sample_calls == [(4, 0.4)]
    assert contexts == [QNetTrainContext(steps=10, train_steps_count=1, gradient_steps=1)]


def test_qnet_bulk_reshape_handles_list_leaves_and_batch_size_one():
    agent = FakePulseAgent()
    agent.batch_size = 1
    lifecycle = QNetTrainingLifecycle(agent)
    data = {
        "obses": [np.ones((2, 3))],
        "indexes": np.arange(2),
    }

    reshaped = lifecycle._reshape_bulk_batch(data, chunk_size=2)

    assert reshaped["obses"][0].shape == (2, 1, 3)
    assert reshaped["indexes"].shape == (2, 1)
    sliced = list(iter_bulk_batches(reshaped, (object(), object())))
    assert sliced[0]["obses"][0].shape == (1, 3)
    assert sliced[0]["indexes"].shape == (1,)


def test_local_qnet_algorithms_use_train_on_batch_and_inherit_train_step():
    targets = {
        "jax_baselines/DQN/dqn.py": "DQN",
        "jax_baselines/C51/c51.py": "C51",
        "jax_baselines/C51/hl_gauss_c51.py": "HL_GAUSS_C51",
        "jax_baselines/QRDQN/qrdqn.py": "QRDQN",
        "jax_baselines/IQN/iqn.py": "IQN",
        "jax_baselines/FQF/fqf.py": "FQF",
        "jax_baselines/SPR/spr.py": "SPR",
        "jax_baselines/SPR/hl_gauss_spr.py": "HL_GAUSS_SPR",
        "jax_baselines/BBF/bbf.py": "BBF",
        "jax_baselines/BBF/hl_gauss_bbf.py": "HL_GAUSS_BBF",
    }

    repo_root = Path(__file__).resolve().parents[1]
    for relative_path, class_name in targets.items():
        tree = ast.parse((repo_root / relative_path).read_text())
        class_node = next(
            node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name
        )
        method_names = {node.name for node in class_node.body if isinstance(node, ast.FunctionDef)}

        assert "_train_on_batch" in method_names
        assert "train_step" not in method_names


def test_dqn_train_on_bulk_scans_updates_and_stacks_priorities():
    agent = DQN.__new__(DQN)
    agent.params = np.asarray(0)
    agent.target_params = np.asarray(10)
    agent.opt_state = np.asarray(20)
    agent.param_noise = False

    def train_step(params, target_params, opt_state, step, key, obses, indexes):
        del key, indexes
        priorities = np.repeat(step, obses.shape[0])
        return (
            params + 1,
            target_params + 1,
            opt_state + 1,
            step.astype(float),
            step.astype(float) + 10.0,
            priorities,
        )

    agent._train_step = train_step
    data = {
        "obses": np.ones((2, 4, 3)),
        "indexes": np.arange(8).reshape(2, 4),
    }
    contexts = [
        type("Context", (), {"train_steps_count": 1})(),
        type("Context", (), {"train_steps_count": 2})(),
    ]

    result = agent._train_on_bulk(data, contexts)

    assert agent.params == 2
    assert agent.target_params == 12
    assert agent.opt_state == 22
    assert result.report.loss == pytest.approx(1.5)
    assert result.report.target == pytest.approx(11.5)
    assert result.report.update_count == 2
    assert np.array_equal(result.replay_priorities, np.array([[1, 1, 1, 1], [2, 2, 2, 2]]))


def test_flatten_bulk_batch_collapses_chunk_and_batch_axes():
    data = {
        "obses": [np.ones((2, 4, 3))],
        "actions": np.arange(8).reshape(2, 4, 1),
        "weights": 1,
    }

    flattened = flatten_bulk_batch(data)

    assert flattened["obses"][0].shape == (8, 3)
    assert flattened["actions"].shape == (8, 1)
    assert flattened["weights"] == 1


def test_qnet_family_default_aggregation_weights_bulk_chunks():
    agent = Q_Network_Family.__new__(Q_Network_Family)
    reports = [
        QNetTrainReport(
            loss=4.0,
            target=14.0,
            metrics={"loss/extra": 24.0},
            histograms={"loss/tau": np.array([4.0, 14.0])},
            update_count=4,
        ),
        QNetTrainReport(
            loss=2.0,
            target=12.0,
            metrics={"loss/extra": 22.0},
            histograms={"loss/tau": np.array([2.0, 12.0])},
            update_count=2,
        ),
        QNetTrainReport(
            loss=1.0,
            target=11.0,
            metrics={"loss/extra": 21.0},
            histograms={"loss/tau": np.array([1.0, 11.0])},
            update_count=1,
        ),
    ]

    report = Q_Network_Family._aggregate_train_reports(agent, reports)

    assert report.loss == pytest.approx(3.0)
    assert report.target == pytest.approx(13.0)
    assert report.metrics["loss/extra"] == pytest.approx(23.0)
    assert np.allclose(report.histograms["loss/tau"], np.array([3.0, 13.0]))
    assert report.update_count == 7


def test_spr_lineage_bulk_hook_flattens_and_delegates_to_existing_pulse_train_step():
    agent = SPR.__new__(SPR)
    calls = []

    def train_on_batch(data, context):
        calls.append((data, context))
        return QNetTrainResult.from_values(loss=1.0, replay_priorities=np.ones(8))

    agent._train_on_batch = train_on_batch
    data = {
        "obses": [np.ones((2, 4, 3))],
        "actions": np.arange(8).reshape(2, 4, 1),
    }
    contexts = (
        QNetTrainContext(steps=10, train_steps_count=3, gradient_steps=2),
        QNetTrainContext(steps=10, train_steps_count=4, gradient_steps=2),
    )

    result = SPR._train_on_bulk(agent, data, contexts)

    assert result.report.loss == 1.0
    assert result.report.update_count == 2
    assert calls[0][0]["obses"][0].shape == (8, 3)
    assert calls[0][0]["actions"].shape == (8, 1)
    assert calls[0][1] == QNetTrainContext(steps=10, train_steps_count=3, gradient_steps=2)


def test_spr_aggregate_weights_bulk_chunks_by_update_count():
    agent = SPR.__new__(SPR)
    reports = [
        QNetTrainReport(loss=2.0, target=4.0, metrics={"loss/rprloss": 6.0}, update_count=2),
        QNetTrainReport(loss=5.0, target=7.0, metrics={"loss/rprloss": 9.0}, update_count=1),
    ]

    report = SPR._aggregate_train_reports(agent, reports)

    assert report.loss == pytest.approx(3.0)
    assert report.target == pytest.approx(5.0)
    assert report.metrics["loss/rprloss"] == pytest.approx(7.0)
    assert report.update_count == 3


def test_apex_qnet_actor_builders_close_over_nstep_gamma():
    """APE-X actor builders must bind the n-step gamma (``self._gamma``), never the
    single-step ``self.gamma``.

    The actor-side ``get_abs_td_error`` closure scales the bootstrap with the
    captured ``gamma``; binding ``self.gamma`` instead of ``self._gamma`` mis-weights
    the n-step PER priority. apex_c51 regressed to ``self.gamma`` once before, so this
    pins the invariant across every APE-X Q-Net variant.
    """
    targets = {
        "jax_baselines/DQN/apex_dqn.py": "APE_X_DQN",
        "jax_baselines/C51/apex_c51.py": "APE_X_C51",
        "jax_baselines/QRDQN/apex_qrdqn.py": "APE_X_QRDQN",
        "jax_baselines/IQN/apex_iqn.py": "APE_X_IQN",
    }

    repo_root = Path(__file__).resolve().parents[1]
    for relative_path, class_name in targets.items():
        tree = ast.parse((repo_root / relative_path).read_text())
        class_node = next(
            node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name
        )
        builder = next(
            node
            for node in class_node.body
            if isinstance(node, ast.FunctionDef) and node.name == "get_actor_builder"
        )
        # Collect every `gamma = self.<attr>` binding inside the builder.
        gamma_attrs = []
        for assign in ast.walk(builder):
            if not isinstance(assign, ast.Assign):
                continue
            value = assign.value
            if not (
                isinstance(value, ast.Attribute)
                and isinstance(value.value, ast.Name)
                and value.value.id == "self"
            ):
                continue
            if any(isinstance(t, ast.Name) and t.id == "gamma" for t in assign.targets):
                gamma_attrs.append(value.attr)

        assert gamma_attrs == [
            "_gamma"
        ], f"{class_name} actor builder must bind gamma from self._gamma, got {gamma_attrs}"
