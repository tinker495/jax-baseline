import ast
from pathlib import Path

import numpy as np
import pytest

from jax_baselines.DQN.training import (
    QNetTrainingLifecycle,
    QNetTrainReport,
    QNetTrainResult,
)


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


class FakeLoggerRun:
    def __init__(self):
        self.metrics = []
        self.histograms = []

    def log_metric(self, name, value, steps):
        self.metrics.append((name, value, steps))

    def log_histogram(self, name, value, steps):
        self.histograms.append((name, value, steps))


class FakeAgent:
    _qnet_handles_train_pulse = False

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
    _qnet_handles_train_pulse = True

    def __init__(self):
        super().__init__()
        self.gradient_steps = 2

    def _train_on_batch(self, data, context):
        assert data["indexes"] is not None
        assert context.gradient_steps == self.gradient_steps
        return QNetTrainResult.from_values(
            loss=float(context.train_steps_count + self.gradient_steps),
            target=25.0,
            replay_priorities=np.array([context.train_steps_count + self.gradient_steps]),
        )


def test_qnet_training_lifecycle_handles_priority_update_calls_for_chunked_pulses():
    agent = FakePulseAgent()
    lifecycle = QNetTrainingLifecycle(agent)

    loss = lifecycle.train(steps=10, gradient_steps=4)

    assert loss == 4.0
    assert agent.train_steps_count == 4
    assert agent.replay_buffer.sample_calls == [(8, 0.4), (8, 0.4)]
    assert len(agent.replay_buffer.priority_update_calls) == 2
    assert np.array_equal(agent.replay_buffer.priority_update_calls[-1][0], np.array([3]))
    assert np.array_equal(agent.replay_buffer.priority_update_calls[-1][1], np.array([4]))


def test_qnet_training_lifecycle_rejects_partial_chunked_pulses():
    agent = FakePulseAgent()
    lifecycle = QNetTrainingLifecycle(agent)

    with pytest.raises(ValueError, match="divisible"):
        lifecycle.train(steps=10, gradient_steps=3)


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
