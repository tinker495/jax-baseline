from collections import deque

import jax
import jax.numpy as jnp
import numpy as np
from tqdm.auto import trange

from jax_baselines.common.cpprb_buffers import EpochBuffer
from jax_baselines.common.env_info import get_local_env_info, infer_action_meta
from jax_baselines.common.eval import evaluate_policy, record_and_test
from jax_baselines.common.logger import TensorboardLogger
from jax_baselines.common.optimizer import select_optimizer
from jax_baselines.common.utils import convert_jax, key_gen, restore, save


class Actor_Critic_Policy_Gradient_Family(object):
    def __init__(
        self,
        env_builder,
        model_builder_maker,
        num_workers=1,
        eval_eps=20,
        gamma=0.995,
        learning_rate=3e-4,
        batch_size=32,
        val_coef=0.2,
        ent_coef=0.01,
        use_entropy_adv_shaping=True,
        entropy_adv_shaping_kappa=2.0,
        log_interval=200,
        log_dir=None,
        _init_setup_model=True,
        policy_kwargs=None,
        seed=None,
        optimizer="adamw",
    ):
        self.name = "Actor_Critic_Policy_Gradient_Family"
        self.env_builder = env_builder
        self.model_builder_maker = model_builder_maker
        self.num_workers = num_workers
        self.eval_eps = eval_eps
        self.log_interval = log_interval
        self.policy_kwargs = policy_kwargs
        self.seed = 42 if seed is None else seed
        self.key_seq = key_gen(self.seed)

        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.val_coef = val_coef
        self.ent_coef = ent_coef
        self.log_dir = log_dir
        self.use_entropy_adv_shaping = use_entropy_adv_shaping
        self.entropy_adv_shaping_kappa = entropy_adv_shaping_kappa

        self.params = None
        self.save_path = None
        self.optimizer = select_optimizer(optimizer, self.learning_rate)

        self.get_env_setup()
        # Control model initialization timing across children
        self._init_setup_model = _init_setup_model
        if self._init_setup_model:
            self.setup_model()

    def save_params(self, path):
        save(path, self.params)

    def load_params(self, path):
        self.params = restore(path)

    def get_memory_setup(self):
        self.buffer = EpochBuffer(
            self.batch_size,
            self.observation_space,
            self.worker_size,
            [1] if self.action_type == "discrete" else self.action_size,
        )

    def get_env_setup(self):
        # Use helper to standardize environment info
        (
            self.env,
            self.eval_env,
            self.observation_space,
            self.action_size,
            self.worker_size,
            self.env_type,
        ) = get_local_env_info(self.env_builder, self.num_workers)

        # infer action metadata (type and conversion)
        # For vectorized envs the underlying action_space is stored in env.env_info
        if self.env_type == "VectorizedEnv":
            action_space = self.env.env_info["action_space"]
        else:
            action_space = self.env.action_space
        self.action_type, self.conv_action = infer_action_meta(action_space)

        print("observation size : ", self.observation_space)
        print("action size : ", self.action_size)
        print("worker_size : ", self.worker_size)
        print("-------------------------------------------------")
        if self.action_type == "discrete":
            self._get_actions = self._get_actions_discrete
            self.get_logprob = self.get_logprob_discrete
            self._loss = self._loss_discrete
            self.actions = self.action_discrete
        elif self.action_type == "continuous":
            self._get_actions = self._get_actions_continuous
            self.get_logprob = self.get_logprob_continuous
            self._loss = self._loss_continuous
            self.actions = self.action_continuous

    def setup_model(self):
        pass

    def _train_step(self, steps):
        pass

    def _get_actions_discrete(self, params, obses, key=None) -> jnp.ndarray:
        prob = jax.nn.softmax(
            self.actor(params, key, self.preproc(params, key, convert_jax(obses))),
            axis=1,
        )
        return prob

    def _get_actions_continuous(self, params, obses, key=None) -> jnp.ndarray:
        mu, std = self.actor(params, key, self.preproc(params, key, convert_jax(obses)))
        return mu, jnp.exp(std)

    def action_discrete(self, obs):
        prob = np.asarray(self._get_actions(self.params, obs))
        return np.expand_dims(
            np.stack([np.random.choice(self.action_size[0], p=p) for p in prob], axis=0),
            axis=1,
        )

    def action_continuous(self, obs):
        mu, std = self._get_actions(self.params, obs)
        return np.random.normal(mu, std)

    def get_logprob_discrete(self, prob, action, key, out_prob=False):
        prob = jax.nn.softmax(prob)
        prob = jnp.clip(prob, 1e-8, 1.0)
        prob = prob / jnp.sum(prob, axis=-1, keepdims=True)
        action = action.astype(jnp.int32)
        if out_prob:
            return prob, jnp.log(jnp.take_along_axis(prob, action, axis=1))
        else:
            return jnp.log(jnp.take_along_axis(prob, action, axis=1))

    def get_logprob_continuous(self, prob, action, key, out_prob=False):
        mu, log_std = prob
        std = jnp.exp(log_std)
        if out_prob:
            return prob, -(
                0.5 * jnp.sum(jnp.square((action - mu) / (std + 1e-7)), axis=-1, keepdims=True)
                + jnp.sum(log_std, axis=-1, keepdims=True)
                + 0.5 * jnp.log(2 * np.pi) * jnp.asarray(action.shape[-1], dtype=jnp.float32)
            )
        else:
            return -(
                0.5 * jnp.sum(jnp.square((action - mu) / (std + 1e-7)), axis=-1, keepdims=True)
                + jnp.sum(log_std, axis=-1, keepdims=True)
                + 0.5 * jnp.log(2 * np.pi) * jnp.asarray(action.shape[-1], dtype=jnp.float32)
            )

    def _loss_continuous(self):
        pass

    def _loss_discrete(self):
        pass

    def _get_actions(self, params, obses) -> np.ndarray:
        pass

    def discription(self, eval_result=None):
        discription = ""
        if eval_result is not None:
            for k, v in eval_result.items():
                discription += f"{k} : {v:8.2f}, "

        discription += f"loss : {np.mean(self.lossque):.3f}"

        return discription

    def learn(
        self,
        total_timesteps,
        callback=None,
        log_interval=1000,
        experiment_name="A2C",
        run_name="A2C",
    ):
        self.update_eps = 1.0
        self.eval_freq = ((total_timesteps // 100) // self.worker_size) * self.worker_size

        pbar = trange(0, total_timesteps, self.worker_size, miniters=log_interval)
        self.logger = TensorboardLogger(run_name, experiment_name, self.log_dir, self)
        with self.logger as self.logger_run:
            if self.env_type == "SingleEnv":
                self.learn_SingleEnv(pbar, callback, log_interval)
            if self.env_type == "VectorizedEnv":
                self.learn_VectorizedEnv(pbar, callback, log_interval)

            self.eval(total_timesteps)

            self.save_params(self.logger_run.get_local_path("params"))

    def learn_SingleEnv(self, pbar, callback=None, log_interval=1000):
        obs, info = self.env.reset()
        obs = [np.expand_dims(obs, axis=0)]
        self.lossque = deque(maxlen=10)
        for steps in pbar:
            actions = self.actions(obs)[0]
            next_obs, reward, terminated, truncated, info = self.env.step(
                self.conv_action(actions)[0]
            )
            next_obs = [np.expand_dims(next_obs, axis=0)]
            self.buffer.add(obs, actions, [reward], next_obs, [terminated], [truncated])
            obs = next_obs

            if terminated or truncated:
                obs, info = self.env.reset()
                obs = [np.expand_dims(obs, axis=0)]

            if (steps + 1) % self.batch_size == 0:  # train in step the environments
                loss = self.train_step(steps)
                self.lossque.append(loss)

            if steps % self.eval_freq == 0:
                eval_result = self.eval(steps)

            if steps % log_interval == 0 and eval_result is not None and len(self.lossque) > 0:
                pbar.set_description(self.discription(eval_result))

    def learn_VectorizedEnv(self, pbar, callback=None, log_interval=1000):
        self.lossque = deque(maxlen=10)
        eval_result = None

        for steps in pbar:
            obs = self.env.current_obs()
            actions = self.actions([obs])
            self.env.step(actions)

            (
                next_obses,
                rewards,
                terminateds,
                truncateds,
                infos,
            ) = self.env.get_result()

            self.buffer.add([obs], actions, rewards, [next_obses], terminateds, truncateds)

            if (steps + self.worker_size) % (
                self.batch_size * self.worker_size
            ) == 0:  # train in step the environments
                loss = self.train_step(steps)
                self.lossque.append(loss)

            if steps % self.eval_freq == 0:
                eval_result = self.eval(steps)

            if steps % log_interval == 0 and eval_result is not None and len(self.lossque) > 0:
                pbar.set_description(self.discription(eval_result))

    def eval(self, steps):
        return evaluate_policy(
            self.eval_env,
            self.eval_eps,
            self.actions,
            logger_run=self.logger_run,
            steps=steps,
            conv_action=self.conv_action,
        )

    def test(self, episode=10):
        with self.logger as self.logger_run:
            self.test_eval_env(episode)

    def test_eval_env(self, episode):
        return record_and_test(
            self.env_builder,
            self.logger_run,
            self.actions,
            episode,
            conv_action=self.conv_action,
        )
