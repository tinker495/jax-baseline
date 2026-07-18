from jax_baselines.core.env_info import get_worker_env_info


class _Event:
    def __init__(self):
        self._set = False

    def set(self):
        self._set = True

    def clear(self):
        self._set = False

    def is_set(self):
        return self._set


class _ParamServer:
    def __init__(self, params):
        self.params = params

    def get_params(self):
        return self.params

    def update_params(self, params):
        self.params = params


class _LoggerServer:
    def __init__(self):
        self.trainer_logs = []
        self.worker_logs = []

    def get_log_dir(self):
        return "/tmp/fake-run"

    def add_multiline(self, eps):
        self.eps = eps

    def log_trainer(self, step, log_dict):
        self.trainer_logs.append((step, log_dict))

    def log_worker(self, log_dict, episode):
        self.worker_logs.append((episode, log_dict))

    def last_update(self):
        self.last_updated = True

    def close(self):
        self.closed = True

    def register_hparams(self, hparams):
        self.hparams = hparams


class _Runtime:
    def replay_manager(self):
        return "manager"

    def create_event(self):
        return _Event()

    def worker_info(self, worker):
        return worker.get_info()

    def create_param_server(self, params):
        return _ParamServer(params)

    def create_logger_server(self, *_args, **_kwargs):
        return _LoggerServer()


class _Worker:
    def get_info(self):
        return {
            "observation_space": {"obs": [4]},
            "action_size": [2],
            "action_type": "discrete",
            "env_type": "single",
            "env_id": "FakeEnv-v0",
            "worker_num": 1,
            "core_env_type": "SingleEnv",
        }


def test_distributed_runtime_core_surface_is_normal_python_calls():
    runtime = _Runtime()

    assert get_worker_env_info([_Worker()], runtime.worker_info) == (
        {"obs": [4]},
        [2],
        "SingleEnv",
    )

    event = runtime.create_event()
    assert not event.is_set()
    event.set()
    assert event.is_set()

    params = runtime.create_param_server({"w": 1})
    assert params.get_params() == {"w": 1}
    params.update_params({"w": 2})
    assert params.get_params() == {"w": 2}

    logger = runtime.create_logger_server(None, "run")
    assert not hasattr(logger.log_worker, "remote")
    logger.log_trainer(3, {"loss/q": 1.0})
    logger.log_worker({"rollout/episode_reward": 2.0}, 4)
    assert logger.trainer_logs == [(3, {"loss/q": 1.0})]
    assert logger.worker_logs == [(4, {"rollout/episode_reward": 2.0})]
