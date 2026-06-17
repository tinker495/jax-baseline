import numpy as np

from jax_baselines.DDPG.training import DPGTrainingLifecycle, DPGTrainReport


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
