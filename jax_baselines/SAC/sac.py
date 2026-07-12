from copy import deepcopy
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import struct

from jax_baselines.DDPG.base_class import Deteministic_Policy_Gradient_Family
from jax_baselines.DDPG.training import DPGTrainReport
from jax_baselines.math.jax_utils import convert_jax
from jax_baselines.math.param_updates import scaled_by_reset, soft_update


def sample_action(mu, log_std, key):
    eps = jax.random.normal(key, mu.shape)
    return jnp.tanh(mu + jnp.exp(log_std) * eps)


def mode_action(mu):
    return jnp.tanh(mu)


def entropy_target_from_sigma(action_dim: int, sigma_target: float) -> float:
    if sigma_target <= 0:
        raise ValueError("sigma_target must be greater than 0")
    return 0.5 * action_dim * np.log(2.0 * np.pi * np.e * sigma_target**2)


@struct.dataclass
class SACCheckpointParams:
    policy_params: Any
    critic_params: Any
    target_critic_params: Any
    log_ent_coef: Any


class SAC(Deteministic_Policy_Gradient_Family):
    _run_name = "SAC"
    supports_bulk_training = True

    def __init__(
        self,
        env_builder: callable,
        model_builder_maker,
        ent_coef="auto_0.01",
        sigma_target=0.15,
        actor_update_period=2,
        **kwargs,
    ):
        if actor_update_period <= 0:
            raise ValueError("actor_update_period must be greater than 0")

        self._ent_coef = ent_coef
        self.ent_coef_learning_rate = 1e-4
        self.actor_update_period = actor_update_period

        super().__init__(env_builder, model_builder_maker, **kwargs)

        self.target_entropy = entropy_target_from_sigma(
            int(np.prod(self.action_size)), sigma_target
        )

    def setup_model(self):
        model_builder = self.model_builder_maker(
            self.observation_space,
            self.action_size,
            self.policy_kwargs,
        )
        (
            self.preproc,
            self.actor,
            self.critic,
            self.policy_params,
            self.critic_params,
        ) = model_builder(next(self.key_seq), print_model=True)
        self.target_critic_params = deepcopy(self.critic_params)
        self.opt_policy_state = self.optimizer.init(self.policy_params)
        self.opt_critic_state = self.optimizer.init(self.critic_params)

        self._setup_entropy_coef()

        self._get_actions = jax.jit(self._get_actions)
        self._get_eval_actions = jax.jit(self._get_eval_actions)
        self._train_step = jax.jit(self._train_step)
        self._train_ent_coef = jax.jit(self._train_ent_coef)
        self._bulk_scan = jax.jit(self._bulk_scan)

    def checkpoint_params(self):
        return SACCheckpointParams(
            policy_params=self.policy_params,
            critic_params=self.critic_params,
            target_critic_params=self.target_critic_params,
            log_ent_coef=self.log_ent_coef,
        )

    def load_checkpoint_params(self, bundle):
        self.policy_params = bundle.policy_params
        self.critic_params = bundle.critic_params
        self.target_critic_params = bundle.target_critic_params
        self.log_ent_coef = bundle.log_ent_coef

    def _get_pi_log_prob(self, params, feature, key=None) -> jnp.ndarray:
        mu, log_std = self.actor(params, None, feature)
        std = jnp.exp(log_std)
        x_t = mu + std * jax.random.normal(key, std.shape)
        pi = jax.nn.tanh(x_t)
        log_prob = jnp.sum(
            -0.5 * (jnp.square((x_t - mu) / (std + 1e-6)) + 2 * log_std + jnp.log(2 * np.pi))
            - jnp.log(1 - jnp.square(pi) + 1e-6),
            axis=1,
            keepdims=True,
        )
        return pi, log_prob

    def _get_actions(self, params, obses, key=None) -> jnp.ndarray:
        mu, log_std = self.actor(params, None, self.preproc(params, None, convert_jax(obses)))
        return sample_action(mu, log_std, key)

    def _get_eval_actions(self, params, obses) -> jnp.ndarray:
        mu, _ = self.actor(params, None, self.preproc(params, None, convert_jax(obses)))
        return mode_action(mu)

    def _train_on_batch(self, data, context):
        (
            self.policy_params,
            self.critic_params,
            self.target_critic_params,
            self.opt_policy_state,
            self.opt_critic_state,
            self.opt_ent_coef_state,
            loss,
            t_mean,
            self.log_ent_coef,
            new_priorities,
        ) = self._train_step(
            self.policy_params,
            self.critic_params,
            self.target_critic_params,
            self.opt_policy_state,
            self.opt_critic_state,
            self.opt_ent_coef_state,
            next(self.key_seq),
            context.train_steps_count,
            self.log_ent_coef,
            **data,
        )
        return DPGTrainReport(
            loss=loss,
            target=t_mean,
            new_priorities=new_priorities,
            metrics={"loss/ent_coef": np.exp(self.log_ent_coef)},
        )

    def _train_on_bulk(self, data, contexts):
        steps = jnp.asarray([context.train_steps_count for context in contexts])
        keys = jax.random.split(next(self.key_seq), len(contexts))
        carry = (
            self.policy_params,
            self.critic_params,
            self.target_critic_params,
            self.opt_policy_state,
            self.opt_critic_state,
            self.opt_ent_coef_state,
            self.log_ent_coef,
        )
        (
            self.policy_params,
            self.critic_params,
            self.target_critic_params,
            self.opt_policy_state,
            self.opt_critic_state,
            self.opt_ent_coef_state,
            self.log_ent_coef,
        ), (losses, targets, ent_coefs, priorities) = self._bulk_scan(carry, keys, steps, data)
        return DPGTrainReport(
            loss=jnp.mean(losses),
            target=jnp.mean(targets),
            new_priorities=priorities,
            metrics={"loss/ent_coef": jnp.mean(jnp.exp(ent_coefs))},
            update_count=len(contexts),
        )

    def _bulk_scan(self, carry, keys, steps, data):
        def train_one(carry, xs):
            (
                policy_params,
                critic_params,
                target_critic_params,
                opt_policy_state,
                opt_critic_state,
                opt_ent_coef_state,
                log_ent_coef,
            ) = carry
            key, step, batch = xs
            (
                policy_params,
                critic_params,
                target_critic_params,
                opt_policy_state,
                opt_critic_state,
                opt_ent_coef_state,
                loss,
                t_mean,
                log_ent_coef,
                priorities,
            ) = self._train_step(
                policy_params,
                critic_params,
                target_critic_params,
                opt_policy_state,
                opt_critic_state,
                opt_ent_coef_state,
                key,
                step,
                log_ent_coef,
                **batch,
            )
            return (
                policy_params,
                critic_params,
                target_critic_params,
                opt_policy_state,
                opt_critic_state,
                opt_ent_coef_state,
                log_ent_coef,
            ), (loss, t_mean, log_ent_coef, priorities)

        return jax.lax.scan(train_one, carry, (keys, steps, data))

    def _train_step(
        self,
        policy_params,
        critic_params,
        target_critic_params,
        opt_policy_state,
        opt_critic_state,
        opt_ent_coef_state,
        key,
        step,
        log_ent_coef,
        obses,
        actions,
        rewards,
        nxtobses,
        terminateds,
        weights=1,
        indexes=None,
    ):
        obses = convert_jax(obses)
        nxtobses = convert_jax(nxtobses)
        not_terminateds = 1.0 - terminateds
        ent_coef = jnp.exp(log_ent_coef)
        key1, key2, key3 = jax.random.split(key, 3)
        targets = self._target(
            policy_params,
            target_critic_params,
            rewards,
            nxtobses,
            not_terminateds,
            key1,
            ent_coef,
        )

        (critic_loss, abs_error), grad = jax.value_and_grad(self._critic_loss, has_aux=True)(
            critic_params, policy_params, obses, actions, targets, weights, key2
        )
        updates, opt_critic_state = self.optimizer.update(
            grad, opt_critic_state, params=critic_params
        )
        critic_params = optax.apply_updates(critic_params, updates)

        def update_actor(state):
            policy_params, opt_policy_state, log_ent_coef, opt_ent_coef_state = state
            (actor_loss, log_prob), grad = jax.value_and_grad(self._actor_loss, has_aux=True)(
                policy_params, critic_params, obses, key3, ent_coef
            )
            updates, opt_policy_state = self.optimizer.update(
                grad, opt_policy_state, params=policy_params
            )
            policy_params = optax.apply_updates(policy_params, updates)
            if self.auto_entropy:
                log_ent_coef, opt_ent_coef_state = self._train_ent_coef(
                    log_ent_coef, opt_ent_coef_state, log_prob
                )
            return (
                policy_params,
                opt_policy_state,
                log_ent_coef,
                opt_ent_coef_state,
                actor_loss,
            )

        (
            policy_params,
            opt_policy_state,
            log_ent_coef,
            opt_ent_coef_state,
            actor_loss,
        ) = jax.lax.cond(
            (step - 1) % self.actor_update_period == 0,
            update_actor,
            lambda state: (*state, jnp.asarray(0.0)),
            (policy_params, opt_policy_state, log_ent_coef, opt_ent_coef_state),
        )

        target_critic_params = soft_update(
            critic_params, target_critic_params, self.target_network_update_tau
        )

        new_priorities = None
        if self.prioritized_replay:
            new_priorities = abs_error
        if self.scaled_by_reset:
            policy_params, opt_policy_state = scaled_by_reset(
                policy_params,
                opt_policy_state,
                self.optimizer,
                key,
                step,
                self.reset_freq,
                0.1,  # tau = 0.1 is softreset, but original paper uses 1.0
            )
            critic_params, opt_critic_state = scaled_by_reset(
                critic_params,
                opt_critic_state,
                self.optimizer,
                key,
                step,
                self.reset_freq,
                0.1,  # tau = 0.1 is softreset, but original paper uses 1.0
            )
        return (
            policy_params,
            critic_params,
            target_critic_params,
            opt_policy_state,
            opt_critic_state,
            opt_ent_coef_state,
            critic_loss,
            -actor_loss,
            log_ent_coef,
            new_priorities,
        )

    def _critic_loss(self, critic_params, policy_params, obses, actions, targets, weights, key):
        feature = self.preproc(policy_params, key, obses)
        q1, q2 = self.critic(critic_params, key, feature, actions)
        error1 = jnp.squeeze(q1 - targets)
        error2 = jnp.squeeze(q2 - targets)
        critic_loss = jnp.mean(weights * jnp.square(error1)) + jnp.mean(
            weights * jnp.square(error2)
        )
        return critic_loss, jnp.abs(error1)

    def _actor_loss(self, policy_params, critic_params, obses, key, ent_coef):
        feature = self.preproc(policy_params, key, obses)
        policy, log_prob = self._get_pi_log_prob(policy_params, feature, key)
        q1_pi, q2_pi = self.critic(critic_params, key, feature, policy)
        actor_loss = jnp.mean(ent_coef * log_prob - jnp.minimum(q1_pi, q2_pi))
        return actor_loss, log_prob

    def _target(
        self,
        policy_params,
        target_critic_params,
        rewards,
        nxtobses,
        not_terminateds,
        key,
        ent_coef,
    ):
        next_feature = self.preproc(policy_params, key, nxtobses)
        policy, log_prob = self._get_pi_log_prob(policy_params, next_feature, key)
        q1_pi, q2_pi = self.critic(target_critic_params, key, next_feature, policy)
        next_q = jnp.minimum(q1_pi, q2_pi) - ent_coef * log_prob
        return (not_terminateds * next_q * self._gamma) + rewards
