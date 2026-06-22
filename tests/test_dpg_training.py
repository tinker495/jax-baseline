import ast
from pathlib import Path

import numpy as np
import pytest

from jax_baselines.core.rollout import CheckpointTrainPulse
from jax_baselines.core.seeding import key_gen
from jax_baselines.DDPG.base_class import Deteministic_Policy_Gradient_Family
from jax_baselines.DDPG.ddpg import DDPG
from jax_baselines.DDPG.training import DPGTrainingLifecycle, DPGTrainReport
from jax_baselines.TD7.td7 import TD7


class FakeReplayBuffer:
    def __init__(self):
        self.sample_calls = []
        self.priority_updates = []

    def sample(self, batch_size, beta=None):
        self.sample_calls.append((batch_size, beta))
        return {
            "obses": np.array([1.0]),
            "nxtobses": np.array([2.0]),
            "indexes": np.array([3]),
        }

    def update_priorities(self, indexes, priorities):
        self.priority_updates.append((indexes, priorities))


class FakeBulkReplayBuffer(FakeReplayBuffer):
    def sample(self, batch_size, beta=None):
        self.sample_calls.append((batch_size, beta))
        update_count = batch_size // 4
        indexes = np.arange(update_count * 4).reshape(update_count, 4)
        weights = np.arange(1, update_count * 4 + 1, dtype=np.float32)
        return {
            "obses": np.ones((update_count, 4)),
            "nxtobses": np.full((update_count, 4), 2.0),
            "indexes": indexes,
            "weights": weights,
        }


class FakeNormalizer:
    def __init__(self, offset):
        self.offset = offset

    def normalize(self, value):
        return value + self.offset


class FakeLoggerRun:
    def __init__(self):
        self.metrics = []

    def log_metric(self, name, value, steps):
        self.metrics.append((name, value, steps))


class FakeAgent:
    def __init__(self):
        self.replay_buffer = FakeReplayBuffer()
        self.batch_size = 4
        self.prioritized_replay = True
        self.prioritized_replay_beta0 = 0.4
        self.simba = True
        self.obs_rms = FakeNormalizer(100.0)
        self.action_obs_rms = FakeNormalizer(10.0)
        self.train_steps_count = 0
        self.logger_run = FakeLoggerRun()
        self._last_log_step = 0
        self.log_interval = 5
        self.contexts = []
        self.batches = []

    def _train_on_batch(self, data, context):
        self.contexts.append(context)
        self.batches.append(data)
        return DPGTrainReport(
            loss=float(context.train_steps_count),
            target=20.0 + context.train_steps_count,
            new_priorities=np.array([context.train_steps_count]),
        )

    def _aggregate_train_reports(self, reports):
        return reports[-1]

    def _policy_update_obs_rms(self):
        return self.action_obs_rms if self.action_obs_rms is not None else self.obs_rms


class FakeBulkAgent(FakeAgent):
    supports_bulk_training = True

    def __init__(self):
        super().__init__()
        self.replay_buffer = FakeBulkReplayBuffer()
        self.max_bulk_updates_per_pulse = 2
        self.bulk_contexts = []
        self.bulk_batches = []

    def _train_on_bulk(self, data, contexts):
        self.bulk_contexts.append(contexts)
        self.bulk_batches.append(data)
        train_counts = np.array([context.train_steps_count for context in contexts])
        return DPGTrainReport(
            loss=float(train_counts[-1]),
            target=20.0 + float(train_counts[-1]),
            new_priorities=np.repeat(train_counts[:, None], self.batch_size, axis=1),
            update_count=len(contexts),
        )

    def _train_on_batch(self, data, context):
        self.contexts.append(context)
        self.batches.append(data)
        return DPGTrainReport(
            loss=float(context.train_steps_count),
            target=20.0 + context.train_steps_count,
            new_priorities=np.repeat(context.train_steps_count, self.batch_size),
        )


def test_dpg_training_lifecycle_samples_normalizes_updates_priorities_and_logs():
    agent = FakeAgent()
    lifecycle = DPGTrainingLifecycle(agent)

    loss = lifecycle.train(steps=10, gradient_steps=2)

    assert loss == 2.0
    assert agent.train_steps_count == 2
    assert agent.replay_buffer.sample_calls == [(4, 0.4), (4, 0.4)]
    assert agent.contexts[0].steps == 10
    assert agent.contexts[0].train_steps_count == 1
    assert np.array_equal(agent.batches[0]["obses"], np.array([11.0]))
    assert np.array_equal(agent.batches[0]["nxtobses"], np.array([12.0]))
    assert len(agent.replay_buffer.priority_updates) == 2
    assert np.array_equal(agent.replay_buffer.priority_updates[-1][0], np.array([3]))
    assert np.array_equal(agent.replay_buffer.priority_updates[-1][1], np.array([2]))
    assert agent.logger_run.metrics == [
        ("loss/qloss", 2.0, 10),
        ("loss/targets", 22.0, 10),
    ]


def test_dpg_training_lifecycle_falls_back_to_live_rms_before_policy_snapshot():
    agent = FakeAgent()
    agent.action_obs_rms = None
    lifecycle = DPGTrainingLifecycle(agent)

    lifecycle.train(steps=10, gradient_steps=1)

    assert np.array_equal(agent.batches[0]["obses"], np.array([101.0]))
    assert np.array_equal(agent.batches[0]["nxtobses"], np.array([102.0]))


def test_dpg_training_lifecycle_bulk_pulse_chunks_steps_and_delays_priorities():
    agent = FakeBulkAgent()
    lifecycle = DPGTrainingLifecycle(agent)

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
    assert [ctx.train_steps_count for ctx in agent.contexts] == [5]
    assert len(agent.replay_buffer.priority_updates) == 3
    assert np.array_equal(agent.replay_buffer.priority_updates[0][0], np.arange(8))
    assert np.array_equal(
        agent.replay_buffer.priority_updates[0][1],
        np.array([1, 1, 1, 1, 2, 2, 2, 2]),
    )
    assert np.array_equal(agent.replay_buffer.priority_updates[-1][0], np.arange(4))
    assert np.array_equal(agent.replay_buffer.priority_updates[-1][1], np.array([5, 5, 5, 5]))


def test_dpg_bulk_reshape_handles_batch_size_one():
    agent = FakeBulkAgent()
    agent.batch_size = 1
    lifecycle = DPGTrainingLifecycle(agent)
    data = {
        "obses": np.ones((2, 3)),
        "nxtobses": np.full((2, 3), 2.0),
        "indexes": np.arange(2),
    }

    reshaped = lifecycle._reshape_bulk_batch(data, chunk_size=2)

    assert reshaped["obses"].shape == (2, 1, 3)
    assert reshaped["nxtobses"].shape == (2, 1, 3)
    assert reshaped["indexes"].shape == (2, 1)


class FakeMissingBulkHookAgent(FakeAgent):
    max_bulk_updates_per_pulse = 2


def test_dpg_training_lifecycle_uses_scalar_path_without_bulk_hook():
    agent = FakeMissingBulkHookAgent()
    lifecycle = DPGTrainingLifecycle(agent)

    lifecycle.train(steps=10, gradient_steps=2)

    assert agent.replay_buffer.sample_calls == [(4, 0.4), (4, 0.4)]
    assert [ctx.train_steps_count for ctx in agent.contexts] == [1, 2]


def test_dpg_training_lifecycle_requires_explicit_bulk_marker():
    agent = FakeBulkAgent()
    agent.supports_bulk_training = False
    lifecycle = DPGTrainingLifecycle(agent)

    lifecycle.train(steps=10, gradient_steps=2)

    assert agent.bulk_contexts == []
    assert [ctx.train_steps_count for ctx in agent.contexts] == [1, 2]


def test_td3_train_on_batch_uses_ordered_train_step_count():
    repo_root = Path(__file__).resolve().parents[1]
    tree = ast.parse((repo_root / "jax_baselines/TD3/td3.py").read_text())
    class_node = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "TD3"
    )
    method = next(
        node
        for node in class_node.body
        if isinstance(node, ast.FunctionDef) and node.name == "_train_on_batch"
    )

    attrs = [
        node.attr
        for node in ast.walk(method)
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "context"
    ]

    assert "train_steps_count" in attrs
    assert "steps" not in attrs


def test_ddpg_train_on_bulk_scans_updates_and_stacks_priorities():
    agent = DDPG.__new__(DDPG)
    agent.key_seq = key_gen(0)
    agent.policy_params = np.asarray(0)
    agent.critic_params = np.asarray(10)
    agent.target_policy_params = np.asarray(20)
    agent.target_critic_params = np.asarray(30)
    agent.opt_policy_state = np.asarray(40)
    agent.opt_critic_state = np.asarray(50)

    def train_step(
        policy_params,
        critic_params,
        target_policy_params,
        target_critic_params,
        opt_policy_state,
        opt_critic_state,
        step,
        key,
        obses,
        indexes,
    ):
        del key, indexes
        priorities = np.repeat(step, obses.shape[0])
        return (
            policy_params + 1,
            critic_params + 1,
            target_policy_params + 1,
            target_critic_params + 1,
            opt_policy_state + 1,
            opt_critic_state + 1,
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

    report = agent._train_on_bulk(data, contexts)

    assert agent.policy_params == 2
    assert agent.critic_params == 12
    assert agent.target_policy_params == 22
    assert agent.target_critic_params == 32
    assert agent.opt_policy_state == 42
    assert agent.opt_critic_state == 52
    assert report.loss == pytest.approx(1.5)
    assert report.target == pytest.approx(11.5)
    assert report.update_count == 2
    assert np.array_equal(report.new_priorities, np.array([[1, 1, 1, 1], [2, 2, 2, 2]]))


def test_td7_aggregate_weights_bulk_chunks_by_update_count():
    agent = TD7.__new__(TD7)
    agent.critic_params = {"values": {"min_value": -1.0, "max_value": 1.0}}
    reports = [
        DPGTrainReport(
            loss=2.0,
            target=4.0,
            metrics={"loss/encoder_loss": 6.0},
            update_count=2,
        ),
        DPGTrainReport(
            loss=5.0,
            target=7.0,
            metrics={"loss/encoder_loss": 9.0},
            update_count=1,
        ),
    ]

    report = TD7._aggregate_train_reports(agent, reports)

    assert report.loss == pytest.approx(3.0)
    assert report.target == pytest.approx(5.0)
    assert report.metrics["loss/encoder_loss"] == pytest.approx(7.0)
    assert report.update_count == 3


def test_dpg_family_default_aggregation_weights_bulk_chunks():
    agent = Deteministic_Policy_Gradient_Family.__new__(Deteministic_Policy_Gradient_Family)
    reports = [
        DPGTrainReport(loss=3.0, target=13.0, metrics={"loss/extra": 23.0}, update_count=3),
        DPGTrainReport(loss=1.0, target=11.0, metrics={"loss/extra": 21.0}, update_count=1),
        DPGTrainReport(loss=5.0, target=15.0, metrics={"loss/extra": 25.0}, update_count=1),
    ]

    report = Deteministic_Policy_Gradient_Family._aggregate_train_reports(agent, reports)

    assert report.loss == pytest.approx(3.0)
    assert report.target == pytest.approx(13.0)
    assert report.metrics["loss/extra"] == pytest.approx(23.0)
    assert report.update_count == 5


def test_checkpoint_train_pulse_drives_dpg_bulk_chunks_under_cap():
    agent = FakeBulkAgent()
    lifecycle = DPGTrainingLifecycle(agent)
    state = {"residual": 0, "losses": []}
    pulse = CheckpointTrainPulse(
        train_freq=1,
        gradient_steps=1,
        train=lifecycle.train,
        record_loss=state["losses"].append,
        read_residual=lambda: state["residual"],
        write_residual=lambda value: state.__setitem__("residual", value),
    )

    pulse(steps=50, accumulated_timesteps=5)

    assert state["residual"] == 0
    assert state["losses"] == [5.0]
    assert agent.train_steps_count == 5
    assert agent.replay_buffer.sample_calls == [(8, 0.4), (8, 0.4), (4, 0.4)]
    assert [[ctx.train_steps_count for ctx in chunk] for chunk in agent.bulk_contexts] == [
        [1, 2],
        [3, 4],
    ]
    assert [ctx.train_steps_count for ctx in agent.contexts] == [5]


def test_dpg_training_lifecycle_uses_scalar_tail_after_full_bulk_chunks():
    agent = FakeBulkAgent()
    lifecycle = DPGTrainingLifecycle(agent)

    lifecycle.train(steps=10, gradient_steps=3)

    assert agent.replay_buffer.sample_calls == [(8, 0.4), (4, 0.4)]
    assert [[ctx.train_steps_count for ctx in chunk] for chunk in agent.bulk_contexts] == [
        [1, 2],
    ]
    assert [ctx.train_steps_count for ctx in agent.contexts] == [3]


def test_dpg_training_lifecycle_uses_bucket_when_pulse_is_smaller_than_cap():
    agent = FakeBulkAgent()
    agent.max_bulk_updates_per_pulse = 8
    lifecycle = DPGTrainingLifecycle(agent)

    lifecycle.train(steps=10, gradient_steps=2)

    assert agent.replay_buffer.sample_calls == [(8, 0.4)]
    assert [[ctx.train_steps_count for ctx in chunk] for chunk in agent.bulk_contexts] == [[1, 2]]
    assert agent.contexts == []


def test_dpg_training_lifecycle_uses_smaller_buckets_before_scalar_tail():
    agent = FakeBulkAgent()
    agent.max_bulk_updates_per_pulse = 8
    lifecycle = DPGTrainingLifecycle(agent)

    lifecycle.train(steps=10, gradient_steps=7)

    assert agent.replay_buffer.sample_calls == [(16, 0.4), (8, 0.4), (4, 0.4)]
    assert [[ctx.train_steps_count for ctx in chunk] for chunk in agent.bulk_contexts] == [
        [1, 2, 3, 4],
        [5, 6],
    ]
    assert [ctx.train_steps_count for ctx in agent.contexts] == [7]


def test_dpg_training_lifecycle_uses_scalar_tail_larger_than_one_after_full_bulk_chunk():
    agent = FakeBulkAgent()
    agent.max_bulk_updates_per_pulse = 3
    lifecycle = DPGTrainingLifecycle(agent)

    lifecycle.train(steps=10, gradient_steps=5)

    assert agent.replay_buffer.sample_calls == [(12, 0.4), (4, 0.4), (4, 0.4)]
    assert [[ctx.train_steps_count for ctx in chunk] for chunk in agent.bulk_contexts] == [
        [1, 2, 3],
    ]
    assert [ctx.train_steps_count for ctx in agent.contexts] == [4, 5]


def test_dpg_training_lifecycle_uses_scalar_path_when_cap_is_one():
    agent = FakeBulkAgent()
    agent.max_bulk_updates_per_pulse = 1
    lifecycle = DPGTrainingLifecycle(agent)

    lifecycle.train(steps=10, gradient_steps=3)

    assert agent.replay_buffer.sample_calls == [(4, 0.4), (4, 0.4), (4, 0.4)]
    assert agent.bulk_contexts == []
    assert [ctx.train_steps_count for ctx in agent.contexts] == [1, 2, 3]


def test_dpg_training_lifecycle_keeps_non_positive_bulk_cap_invalid():
    agent = FakeBulkAgent()
    agent.max_bulk_updates_per_pulse = 0
    lifecycle = DPGTrainingLifecycle(agent)

    with pytest.raises(ValueError, match="max_bulk_updates_per_pulse"):
        lifecycle.train(steps=10, gradient_steps=2)
