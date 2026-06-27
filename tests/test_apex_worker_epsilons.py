from jax_baselines.APE_X.exploration import worker_epsilons


def test_apex_single_worker_epsilon_schedule_uses_initial_epsilon():
    assert worker_epsilons(0.4, 3.0, 1) == [0.4]


def test_apex_multi_worker_epsilon_schedule_preserves_decay():
    assert worker_epsilons(0.4, 3.0, 3) == [0.4, 0.4**2.5, 0.4**4.0]
