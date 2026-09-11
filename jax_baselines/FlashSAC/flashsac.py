"""FlashSAC with categorical critics, unit-normalized models and correlated exploration.

Algorithm reference: Holiday-Robot/FlashSAC at 87edc9061150ae9e962dd84e6544e27a1554b3ab.
Environment, replay, network construction and storage use the existing DPG adapter seams.
"""

import math
from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
import optax
from flax import struct

from jax_baselines.core.normalization import FlashSACRewardNormalizer
from jax_baselines.DDPG.base_class import Deteministic_Policy_Gradient_Family
from jax_baselines.DDPG.training import DPGTrainReport
from jax_baselines.math.distributional import categorical_projection
from jax_baselines.math.jax_utils import convert_normalized_obs
from jax_baselines.math.param_updates import project_unit_norm_params
from jax_baselines.math.policy_math import entropy_target_from_sigma


@struct.dataclass
class FlashSACCheckpointParams:
    policy_params: Any
    critic_params: Any
    target_critic_params: Any
    log_ent_coef: Any


def sample_policy(mean: jax.Array, log_std: jax.Array, key: jax.Array):
    noise = jax.random.normal(key, mean.shape)
    raw_action = mean + jnp.exp(log_std) * noise
    log_prob = -0.5 * (noise**2 + 2 * log_std + math.log(2 * math.pi))
    log_prob -= 2 * (math.log(2) - raw_action - jax.nn.softplus(-2 * raw_action))
    return jnp.tanh(raw_action), jnp.sum(log_prob, axis=-1, keepdims=True)


class FlashSAC(Deteministic_Policy_Gradient_Family):
    _run_name = "FlashSAC"
    supports_bulk_training = True

    def __init__(
        self,
        env_builder: Callable,
        model_builder_maker: Callable,
        ent_coef="auto_0.01",
        sigma_target: float = 0.15,
        actor_update_period: int = 2,
        learning_rate: float = 3e-4,
        learning_rate_end: float = 1.5e-4,
        lr_transition_steps: int = 97_658,
        normalized_G_max: float = 5.0,
        actor_noise_zeta_mu: float = 2.0,
        actor_noise_zeta_max: int = 16,
        n_atoms: int = 101,
        gamma: float = 0.99,
        n_step: int = 3,
        target_network_update_tau: float = 0.01,
        reward_normalization: bool = True,
        policy_kwargs: dict[str, Any] | None = None,
        simba: bool = False,
        simba_v2: bool = False,
        scaled_by_reset: bool = False,
        prioritized_replay: bool = False,
        **kwargs: Any,
    ):
        if simba or simba_v2 or scaled_by_reset or prioritized_replay:
            raise ValueError("FlashSAC requires its unit-normalized model and uniform replay")
        if actor_update_period < 1 or lr_transition_steps < 1 or n_step < 1 or n_atoms < 2:
            raise ValueError(
                "update periods and n_step must be positive; n_atoms must be at least 2"
            )
        if not math.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive")
        if not math.isfinite(learning_rate_end) or not 0 <= learning_rate_end <= learning_rate:
            raise ValueError("learning_rate_end must be finite and in [0, learning_rate]")
        if not math.isfinite(normalized_G_max) or normalized_G_max <= 0:
            raise ValueError("normalized_G_max must be finite and positive")
        if (
            actor_noise_zeta_max < 1
            or not math.isfinite(actor_noise_zeta_mu)
            or actor_noise_zeta_mu <= 0
        ):
            raise ValueError("Zeta exponent and maximum repeat length must be positive")
        if not 0 <= gamma <= 1 or not 0 < target_network_update_tau <= 1:
            raise ValueError("gamma must be in [0, 1] and target tau in (0, 1]")
        self._ent_coef = ent_coef
        self.actor_update_period = actor_update_period
        self.learning_rate_end = learning_rate_end
        self.lr_transition_steps = lr_transition_steps
        self.n_atoms = n_atoms
        self.value_min = -normalized_G_max
        self.value_max = normalized_G_max
        self.support_delta = 2 * normalized_G_max / (n_atoms - 1)
        self.value_support = jnp.linspace(self.value_min, self.value_max, n_atoms)
        self.noise_logits = -actor_noise_zeta_mu * jnp.log(
            jnp.arange(1, actor_noise_zeta_max + 1, dtype=jnp.float32)
        )
        model_options = {} if policy_kwargs is None else dict(policy_kwargs)
        if "n_atoms" in model_options and model_options["n_atoms"] != n_atoms:
            raise ValueError("model and critic support must use the same n_atoms")
        model_options["n_atoms"] = n_atoms
        super().__init__(
            env_builder,
            model_builder_maker,
            learning_rate=learning_rate,
            gamma=gamma,
            n_step=n_step,
            target_network_update_tau=target_network_update_tau,
            reward_normalization=reward_normalization,
            policy_kwargs=model_options,
            **kwargs,
        )
        self.target_entropy = entropy_target_from_sigma(math.prod(self.action_size), sigma_target)
        if reward_normalization:
            with jax.default_device(self.memory_device):
                self.reward_normalizer = FlashSACRewardNormalizer(
                    self.worker_size, gamma, normalized_G_max
                )

    def _make_optimizer(self, learning_rate):
        self.ent_coef_learning_rate = optax.cosine_decay_schedule(
            learning_rate, self.lr_transition_steps, alpha=self.learning_rate_end / learning_rate
        )
        return self.optimizer_factory(self.ent_coef_learning_rate)

    def setup_model(self):
        (
            self.preproc,
            self.actor,
            self.critic,
            self.policy_params,
            self.critic_params,
        ) = self.model_builder_maker(self.observation_space, self.action_size, self.policy_kwargs)(
            next(self.key_seq), print_model=True
        )
        self.target_critic_params = jax.tree.map(jnp.array, self.critic_params)
        self.opt_policy_state = self.optimizer.init(self.policy_params["params"])
        self.opt_critic_state = self.optimizer.init(self.critic_params["params"])
        self._setup_entropy_coef()
        self._noise = jnp.zeros((self.worker_size, math.prod(self.action_size)))
        self._noise_count = jnp.asarray(0, dtype=jnp.int32)
        self._noise_period = jnp.asarray(0, dtype=jnp.int32)
        self._compiled_actions = jax.jit(self._get_actions)
        self._compiled_eval = jax.jit(self._get_eval_actions)
        self._compiled_updates = jax.jit(self._bulk_scan)

    def checkpoint_params(self):
        return FlashSACCheckpointParams(
            self.policy_params,
            self.critic_params,
            self.target_critic_params,
            self.log_ent_coef,
        )

    def load_checkpoint_params(self, bundle: FlashSACCheckpointParams):
        self.policy_params = bundle.policy_params
        self.critic_params = bundle.critic_params
        self.target_critic_params = bundle.target_critic_params
        self.log_ent_coef = bundle.log_ent_coef
        self.opt_policy_state = self.optimizer.init(self.policy_params["params"])
        self.opt_critic_state = self.optimizer.init(self.critic_params["params"])
        self.opt_ent_coef_state = self.ent_coef_optimizer.init(self.log_ent_coef)
        self._noise_count = jnp.asarray(0, dtype=jnp.int32)

    def _policy_action_from_state(self, state, obs, eval, steps):
        if eval:
            return self._compiled_eval(state["policy"], obs)
        actions, self._noise, self._noise_count, self._noise_period = self._compiled_actions(
            state["policy"],
            obs,
            next(self.key_seq),
            self._noise,
            self._noise_count,
            self._noise_period,
        )
        return actions

    def _get_actions(self, params, obses, key, noise, count, period):
        (mean, log_std), _ = self.actor(
            params, None, self.preproc(params, None, convert_normalized_obs(obses)), False
        )
        noise_key, period_key = jax.random.split(key)
        refresh = (count == 0) | (count >= period)
        noise = jnp.where(refresh, jax.random.normal(noise_key, mean.shape), noise)
        period = jnp.where(
            refresh, jax.random.categorical(period_key, self.noise_logits) + 1, period
        )
        count = jnp.where(refresh, 0, count) + 1
        return jnp.tanh(mean + jnp.exp(log_std) * noise), noise, count, period

    def _get_eval_actions(self, params, obses):
        (mean, _), _ = self.actor(
            params, None, self.preproc(params, None, convert_normalized_obs(obses)), False
        )
        return jnp.tanh(mean)

    def _train_on_batch(self, data, context):
        return self._train_on_bulk(jax.tree.map(lambda x: jnp.expand_dims(x, 0), data), [context])

    def _train_on_bulk(self, data, contexts):
        carry, metrics = self._compiled_updates(
            (
                self.policy_params,
                self.critic_params,
                self.target_critic_params,
                self.opt_policy_state,
                self.opt_critic_state,
                self.opt_ent_coef_state,
                self.log_ent_coef,
            ),
            jax.random.split(next(self.key_seq), len(contexts)),
            jnp.asarray([context.train_steps_count for context in contexts]),
            data,
        )
        (
            self.policy_params,
            self.critic_params,
            self.target_critic_params,
            self.opt_policy_state,
            self.opt_critic_state,
            self.opt_ent_coef_state,
            self.log_ent_coef,
        ) = carry
        actor_updates = max(
            sum(
                (context.train_steps_count - 1) % self.actor_update_period == 0
                for context in contexts
            ),
            1,
        )
        return DPGTrainReport(
            loss=jnp.mean(metrics[0]),
            target=jnp.mean(metrics[1]),
            metrics={
                "loss/ent_coef": jnp.mean(metrics[2]),
                "loss/actor_loss": jnp.sum(metrics[3]) / actor_updates,
                "loss/entropy": jnp.sum(metrics[4]) / actor_updates,
            },
            update_count=len(contexts),
        )

    def _bulk_scan(self, carry, keys, steps, data):
        return jax.lax.scan(self._train_step, carry, (keys, steps, data))

    def _train_step(self, carry, sample):
        policy, critic, target, actor_opt, critic_opt, alpha_opt, log_alpha = carry
        key, step, data = sample
        actor_key, target_key = jax.random.split(key)
        obs = convert_normalized_obs(data["obses"])
        next_obs = convert_normalized_obs(data["nxtobses"])
        joint_obs = {name: jnp.concatenate((value, next_obs[name])) for name, value in obs.items()}
        batch_size = data["actions"].shape[0]

        def update_actor(state):
            policy, actor_opt, log_alpha, alpha_opt = state
            (loss, (log_prob, stats)), grad = jax.value_and_grad(self._actor_loss, has_aux=True)(
                policy["params"],
                policy["batch_stats"],
                critic,
                joint_obs,
                actor_key,
                jnp.exp(log_alpha),
            )
            updates, actor_opt = self.optimizer.update(grad, actor_opt, policy["params"])
            policy = {
                "params": project_unit_norm_params(optax.apply_updates(policy["params"], updates)),
                "batch_stats": stats,
            }
            if self.auto_entropy:
                log_alpha, alpha_opt = self._train_ent_coef(log_alpha, alpha_opt, log_prob)
            return policy, actor_opt, log_alpha, alpha_opt, loss, -jnp.mean(log_prob)

        policy, actor_opt, log_alpha, alpha_opt, actor_loss, entropy = jax.lax.cond(
            (step - 1) % self.actor_update_period == 0,
            update_actor,
            lambda state: (*state, jnp.asarray(0.0), jnp.asarray(0.0)),
            (policy, actor_opt, log_alpha, alpha_opt),
        )
        features = self.preproc(policy, None, joint_obs)
        next_features = jax.tree.map(lambda value: value[batch_size:], features)
        (mean, log_std), _ = self.actor(policy, None, next_features, False)
        next_actions, log_prob = sample_policy(mean, log_std, target_key)
        joint_actions = jnp.concatenate((data["actions"], next_actions))
        (target_logits1, target_logits2), target_updates = self.critic(
            target, None, features, joint_actions, True
        )
        target_probs1 = jax.nn.softmax(target_logits1[batch_size:])
        target_probs2 = jax.nn.softmax(target_logits2[batch_size:])
        select_first = jnp.sum(target_probs1 * self.value_support, axis=-1) <= jnp.sum(
            target_probs2 * self.value_support, axis=-1
        )
        target_probs = categorical_projection(
            jnp.where(select_first[:, None], target_probs1, target_probs2),
            data["rewards"].reshape(-1, 1)
            + (1 - data["terminateds"].reshape(-1, 1))
            * self._gamma
            * (self.value_support - jnp.exp(log_alpha) * log_prob),
            self.value_min,
            self.value_max,
            self.support_delta,
            self.n_atoms,
        )
        target_probs = jax.lax.stop_gradient(target_probs)
        (critic_loss, stats), grad = jax.value_and_grad(self._critic_loss, has_aux=True)(
            critic["params"], critic["batch_stats"], features, joint_actions, target_probs
        )
        updates, critic_opt = self.optimizer.update(grad, critic_opt, critic["params"])
        critic = {
            "params": project_unit_norm_params(optax.apply_updates(critic["params"], updates)),
            "batch_stats": stats,
        }
        target = {
            "params": optax.incremental_update(
                critic["params"], target["params"], self.target_network_update_tau
            ),
            "batch_stats": target_updates["batch_stats"],
        }
        return (policy, critic, target, actor_opt, critic_opt, alpha_opt, log_alpha), (
            critic_loss,
            jnp.mean(jnp.sum(target_probs * self.value_support, axis=-1)),
            jnp.exp(log_alpha),
            actor_loss,
            entropy,
        )

    def _actor_loss(self, params, stats, critic, joint_obs, key, alpha):
        variables = {"params": params, "batch_stats": stats}
        features = self.preproc(variables, None, joint_obs)
        (mean, log_std), updates = self.actor(variables, None, features, True)
        actions, log_prob = sample_policy(mean, log_std, key)
        size = actions.shape[0] // 2
        (logits1, logits2), _ = self.critic(
            critic, None, jax.tree.map(lambda value: value[:size], features), actions[:size], False
        )
        minimum = jnp.minimum(
            jnp.sum(jax.nn.softmax(logits1) * self.value_support, axis=-1),
            jnp.sum(jax.nn.softmax(logits2) * self.value_support, axis=-1),
        )
        return jnp.mean(alpha * log_prob[:size, 0] - minimum), (
            log_prob[:size],
            updates["batch_stats"],
        )

    def _critic_loss(self, params, stats, features, actions, target_probs):
        (logits1, logits2), updates = self.critic(
            {"params": params, "batch_stats": stats}, None, features, actions, True
        )
        size = target_probs.shape[0]
        loss = -jnp.mean(
            jnp.sum(
                target_probs
                * (jax.nn.log_softmax(logits1[:size]) + jax.nn.log_softmax(logits2[:size]))
                / 2,
                axis=-1,
            )
        )
        return loss, updates["batch_stats"]
