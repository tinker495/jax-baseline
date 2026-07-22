from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import struct

from jax_baselines.DDPG.base_class import Deteministic_Policy_Gradient_Family
from jax_baselines.DDPG.training import DPGTrainReport
from jax_baselines.math.jax_utils import convert_jax
from jax_baselines.math.param_updates import scaled_by_reset
from jax_baselines.math.policy_math import entropy_target_from_sigma


@struct.dataclass
class CrossQCheckpointParams:
    policy_params: Any
    critic_params: Any
    log_ent_coef: Any


class CrossQ(Deteministic_Policy_Gradient_Family):
    _run_name = "CrossQ"
    supports_bulk_training = True

    def __init__(
        self,
        env_builder: callable,
        model_builder_maker,
        ent_coef="auto",
        sigma_target=0.15,
        policy_delay=3,
        **kwargs,
    ):
        # Set CrossQ-specific defaults
        crossq_kwargs = {
            "target_network_update_tau": 0,  # CrossQ doesn't use target network updates
            **kwargs,
        }

        self._ent_coef = ent_coef
        self.ent_coef_learning_rate = 1e-4
        self.policy_delay = policy_delay

        super().__init__(env_builder, model_builder_maker, **crossq_kwargs)

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
        self.opt_policy_state = self.optimizer.init(self.policy_params)
        self.opt_critic_state = self.optimizer.init(self.critic_params)

        self._setup_entropy_coef()

        self._get_actions = jax.jit(self._get_actions)
        self._get_eval_actions = jax.jit(self._get_eval_actions)
        self._train_step = jax.jit(self._train_step)
        self._train_ent_coef = jax.jit(self._train_ent_coef)
        self._bulk_scan = jax.jit(self._bulk_scan)

    def checkpoint_params(self):
        return CrossQCheckpointParams(
            policy_params=self.policy_params,
            critic_params=self.critic_params,
            log_ent_coef=self.log_ent_coef,
        )

    def load_checkpoint_params(self, bundle):
        self.policy_params = bundle.policy_params
        self.critic_params = bundle.critic_params
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
        std = jnp.exp(log_std)
        pi = jax.nn.tanh(mu + std * jax.random.normal(key, std.shape))
        return pi

    def _get_eval_actions(self, params, obses) -> jnp.ndarray:
        mu, _ = self.actor(params, None, self.preproc(params, None, convert_jax(obses)))
        return jax.nn.tanh(mu)

    def _train_on_batch(self, data, context):
        (
            self.policy_params,
            self.critic_params,
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
            self.opt_policy_state,
            self.opt_critic_state,
            self.opt_ent_coef_state,
            self.log_ent_coef,
        )
        (
            self.policy_params,
            self.critic_params,
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
                opt_policy_state,
                opt_critic_state,
                opt_ent_coef_state,
                log_ent_coef,
            ) = carry
            key, step, batch = xs
            (
                policy_params,
                critic_params,
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
        key1, key2 = jax.random.split(key, 2)

        (critic_loss, (abs_error, targets, critic_params)), grad = jax.value_and_grad(
            self._critic_loss, has_aux=True
        )(
            critic_params,
            policy_params,
            obses,
            actions,
            rewards,
            nxtobses,
            not_terminateds,
            ent_coef,
            weights,
            key1,
        )
        updates, opt_critic_state = self.optimizer.update(
            grad, opt_critic_state, params=critic_params
        )
        critic_params = optax.apply_updates(critic_params, updates)

        def _opt_actor(
            policy_params,
            critic_params,
            log_ent_coef,
            opt_policy_state,
            opt_ent_coef_state,
            key,
        ):
            (_, log_prob), grad = jax.value_and_grad(self._actor_loss, has_aux=True)(
                policy_params, critic_params, obses, key, ent_coef
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
                critic_params,
                log_ent_coef,
                opt_policy_state,
                opt_ent_coef_state,
                key,
            )

        (
            policy_params,
            critic_params,
            log_ent_coef,
            opt_policy_state,
            opt_ent_coef_state,
            key,
        ) = jax.lax.cond(
            step % self.policy_delay == 0,
            lambda x: _opt_actor(*x),
            lambda x: x,
            (
                policy_params,
                critic_params,
                log_ent_coef,
                opt_policy_state,
                opt_ent_coef_state,
                key2,
            ),
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
            opt_policy_state,
            opt_critic_state,
            opt_ent_coef_state,
            critic_loss,
            jnp.mean(targets),
            log_ent_coef,
            new_priorities,
        )

    def _critic_loss(
        self,
        critic_params,
        policy_params,
        obses,
        actions,
        rewards,
        nxtobses,
        not_terminateds,
        ent_coef,
        weights,
        key,
    ):
        concated_obses = jax.tree_util.tree_map(
            lambda obs, nxtobs: jnp.concatenate([obs, nxtobs]),
            obses,
            nxtobses,
        )
        concated_preproc = self.preproc(policy_params, key, concated_obses)
        next_preproc = jnp.split(concated_preproc, 2, axis=0)[1]
        next_policy, log_prob = self._get_pi_log_prob(policy_params, next_preproc, key)
        concated_actions = jnp.concatenate([actions, next_policy])
        (q1, q2), variable_updates = self.critic(
            critic_params, key, concated_preproc, concated_actions, True
        )
        critic_params["batch_stats"] = variable_updates["batch_stats"]
        q1, next_q1 = jnp.split(q1, 2, axis=0)
        q2, next_q2 = jnp.split(q2, 2, axis=0)
        next_q = jnp.minimum(next_q1, next_q2) - ent_coef * log_prob
        targets = jax.lax.stop_gradient((not_terminateds * next_q * self._gamma) + rewards)
        error1 = jnp.squeeze(q1 - targets)
        error2 = jnp.squeeze(q2 - targets)
        critic_loss = jnp.mean(weights * jnp.square(error1)) + jnp.mean(
            weights * jnp.square(error2)
        )
        return critic_loss, (jnp.abs(error1), targets, critic_params)

    def _actor_loss(self, policy_params, critic_params, obses, key, ent_coef):
        feature = self.preproc(policy_params, key, obses)
        policy, log_prob = self._get_pi_log_prob(policy_params, feature, key)
        (q1_pi, q2_pi), _ = self.critic(critic_params, key, feature, policy, False)
        actor_loss = jnp.mean(ent_coef * log_prob - jnp.minimum(q1_pi, q2_pi))
        return actor_loss, log_prob
