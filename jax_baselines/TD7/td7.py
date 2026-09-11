from collections.abc import Callable
from copy import deepcopy
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import struct

from jax_baselines.DDPG.base_class import Deteministic_Policy_Gradient_Family
from jax_baselines.DDPG.training import DPGTrainReport
from jax_baselines.math.jax_utils import convert_normalized_obs
from jax_baselines.math.losses import hubberloss
from jax_baselines.math.param_updates import hard_update, scaled_by_reset


@struct.dataclass
class TD7CheckpointParams:
    actor_encoder_params: Any
    critic_encoder_params: Any
    policy_params: Any
    critic_params: Any
    fixed_actor_encoder_params: Any
    fixed_critic_encoder_params: Any
    fixed_actor_encoder_target_params: Any
    fixed_critic_encoder_target_params: Any
    target_policy_params: Any
    target_critic_params: Any


class TD7(Deteministic_Policy_Gradient_Family):
    _run_name = "TD7"
    supports_bulk_training = True

    def __init__(
        self,
        env_builder: Callable,
        model_builder_maker,
        target_action_noise_mul=2.0,
        action_noise=0.1,
        policy_delay=2,
        target_network_update_freq=250,
        **kwargs,
    ):
        # Set TD7-specific defaults - always enable checkpointing
        td7_kwargs: dict[str, Any] = {
            "n_step": 1,
            "target_network_update_tau": 0,
            "prioritized_replay": True,
            "prioritized_replay_beta0": 0,
            "prioritized_replay_eps": 0,
            "use_checkpointing": True,  # TD7 always uses checkpointing
            **kwargs,
        }

        super().__init__(env_builder, model_builder_maker, **td7_kwargs)

        self.action_noise = action_noise
        self.target_action_noise = action_noise * target_action_noise_mul
        self.action_noise_clamp = 0.5
        self.target_network_update_freq = target_network_update_freq
        self.policy_delay = policy_delay

    def setup_model(self):
        model_builder = self.model_builder_maker(
            self.observation_space,
            self.action_size,
            self.policy_kwargs,
        )
        (
            self.actor_encoder,
            self.critic_encoder,
            self.actor_action_encoder,
            self.critic_action_encoder,
            self.actor,
            self.critic,
            self.actor_encoder_params,
            self.critic_encoder_params,
            self.policy_params,
            self.critic_params,
        ) = model_builder(next(self.key_seq), print_model=True)
        self.fixed_actor_encoder_params = deepcopy(self.actor_encoder_params)
        self.fixed_critic_encoder_params = deepcopy(self.critic_encoder_params)
        self.fixed_actor_encoder_target_params = deepcopy(self.actor_encoder_params)
        self.fixed_critic_encoder_target_params = deepcopy(self.critic_encoder_params)
        self.target_policy_params = deepcopy(self.policy_params)
        self.target_critic_params = deepcopy(self.critic_params)

        self.critic_params["values"] = {
            "min_value": jnp.array([np.inf], dtype=jnp.float32),
            "max_value": jnp.array([-np.inf], dtype=jnp.float32),
        }
        self.target_critic_params["values"] = {
            "min_value": jnp.array([0], dtype=jnp.float32),
            "max_value": jnp.array([0], dtype=jnp.float32),
        }

        self.actor_encoder_opt_state = self.optimizer.init(self.actor_encoder_params)
        self.critic_encoder_opt_state = self.optimizer.init(self.critic_encoder_params)
        self.opt_policy_state = self.optimizer.init(self.policy_params)
        self.opt_critic_state = self.optimizer.init(self.critic_params)
        self._compiled_get_actions: Callable = jax.jit(self._get_actions)
        self._compiled_train_step: Callable = jax.jit(self._train_step)
        self._compiled_bulk_scan: Callable = jax.jit(self._bulk_scan)

    def checkpoint_params(self):
        return TD7CheckpointParams(
            actor_encoder_params=self.actor_encoder_params,
            critic_encoder_params=self.critic_encoder_params,
            policy_params=self.policy_params,
            critic_params=self.critic_params,
            fixed_actor_encoder_params=self.fixed_actor_encoder_params,
            fixed_critic_encoder_params=self.fixed_critic_encoder_params,
            fixed_actor_encoder_target_params=self.fixed_actor_encoder_target_params,
            fixed_critic_encoder_target_params=self.fixed_critic_encoder_target_params,
            target_policy_params=self.target_policy_params,
            target_critic_params=self.target_critic_params,
        )

    def load_checkpoint_params(self, bundle):
        self.actor_encoder_params = bundle.actor_encoder_params
        self.critic_encoder_params = bundle.critic_encoder_params
        self.policy_params = bundle.policy_params
        self.critic_params = bundle.critic_params
        self.fixed_actor_encoder_params = bundle.fixed_actor_encoder_params
        self.fixed_critic_encoder_params = bundle.fixed_critic_encoder_params
        self.fixed_actor_encoder_target_params = bundle.fixed_actor_encoder_target_params
        self.fixed_critic_encoder_target_params = bundle.fixed_critic_encoder_target_params
        self.target_policy_params = bundle.target_policy_params
        self.target_critic_params = bundle.target_critic_params

    def _get_actions(self, actor_encoder_params, policy_params, obses, key=None):
        feature, zs = self.actor_encoder(actor_encoder_params, key, convert_normalized_obs(obses))
        return self.actor(policy_params, key, feature, zs)

    def get_behavior_state(self):
        return {
            "actor_encoder": self.fixed_actor_encoder_params,
            "policy": self.policy_params,
        }

    def _select_action_state(self, eval, steps):
        if eval and self.use_checkpointing and self.ckpt.enabled and self.eval_snapshot is not None:
            return self.eval_snapshot
        return self.get_behavior_state()

    def _policy_action_from_state(self, state, obs, eval, steps):
        return self._compiled_get_actions(state["actor_encoder"], state["policy"], obs, None)

    def _apply_action_noise(self, actions, steps, eval):
        if eval:
            return actions
        if self.memory_backend == "gpu":
            return jnp.clip(
                actions + self.action_noise * jax.random.normal(next(self.key_seq), actions.shape),
                -1,
                1,
            )
        return np.clip(
            actions
            + self.action_noise
            * np.random.normal(0, 1, size=(self.worker_size, self.action_size[0])),
            -1,
            1,
        )

    def _train_on_batch(self, data, context):
        (
            self.actor_encoder_params,
            self.critic_encoder_params,
            self.policy_params,
            self.critic_params,
            self.fixed_actor_encoder_params,
            self.fixed_critic_encoder_params,
            self.fixed_actor_encoder_target_params,
            self.fixed_critic_encoder_target_params,
            self.target_policy_params,
            self.target_critic_params,
            self.actor_encoder_opt_state,
            self.critic_encoder_opt_state,
            self.opt_policy_state,
            self.opt_critic_state,
            repr_loss,
            loss,
            t_mean,
            new_priorities,
        ) = self._compiled_train_step(
            self.actor_encoder_params,
            self.critic_encoder_params,
            self.policy_params,
            self.critic_params,
            self.fixed_actor_encoder_params,
            self.fixed_critic_encoder_params,
            self.fixed_actor_encoder_target_params,
            self.fixed_critic_encoder_target_params,
            self.target_policy_params,
            self.target_critic_params,
            self.actor_encoder_opt_state,
            self.critic_encoder_opt_state,
            self.opt_policy_state,
            self.opt_critic_state,
            next(self.key_seq),
            context.train_steps_count,
            **data,
        )
        return DPGTrainReport(
            loss=loss,
            target=t_mean,
            new_priorities=new_priorities,
            metrics={"loss/encoder_loss": repr_loss},
        )

    def _train_on_bulk(self, data, contexts):
        steps = jnp.asarray([context.train_steps_count for context in contexts])
        keys = jax.random.split(next(self.key_seq), len(contexts))
        carry = (
            self.actor_encoder_params,
            self.critic_encoder_params,
            self.policy_params,
            self.critic_params,
            self.fixed_actor_encoder_params,
            self.fixed_critic_encoder_params,
            self.fixed_actor_encoder_target_params,
            self.fixed_critic_encoder_target_params,
            self.target_policy_params,
            self.target_critic_params,
            self.actor_encoder_opt_state,
            self.critic_encoder_opt_state,
            self.opt_policy_state,
            self.opt_critic_state,
        )
        (
            (
                self.actor_encoder_params,
                self.critic_encoder_params,
                self.policy_params,
                self.critic_params,
                self.fixed_actor_encoder_params,
                self.fixed_critic_encoder_params,
                self.fixed_actor_encoder_target_params,
                self.fixed_critic_encoder_target_params,
                self.target_policy_params,
                self.target_critic_params,
                self.actor_encoder_opt_state,
                self.critic_encoder_opt_state,
                self.opt_policy_state,
                self.opt_critic_state,
            ),
            (repr_losses, losses, targets, priorities),
        ) = self._compiled_bulk_scan(carry, keys, steps, data)
        return DPGTrainReport(
            loss=jnp.mean(losses),
            target=jnp.mean(targets),
            new_priorities=priorities,
            metrics={"loss/encoder_loss": jnp.mean(repr_losses)},
            update_count=len(contexts),
        )

    def _bulk_scan(self, carry, keys, steps, data):
        def train_one(carry, xs):
            (
                actor_encoder_params,
                critic_encoder_params,
                policy_params,
                critic_params,
                fixed_actor_encoder_params,
                fixed_critic_encoder_params,
                fixed_actor_encoder_target_params,
                fixed_critic_encoder_target_params,
                target_policy_params,
                target_critic_params,
                actor_encoder_opt_state,
                critic_encoder_opt_state,
                opt_policy_state,
                opt_critic_state,
            ) = carry
            key, step, batch = xs
            (
                actor_encoder_params,
                critic_encoder_params,
                policy_params,
                critic_params,
                fixed_actor_encoder_params,
                fixed_critic_encoder_params,
                fixed_actor_encoder_target_params,
                fixed_critic_encoder_target_params,
                target_policy_params,
                target_critic_params,
                actor_encoder_opt_state,
                critic_encoder_opt_state,
                opt_policy_state,
                opt_critic_state,
                repr_loss,
                loss,
                t_mean,
                priorities,
            ) = self._train_step(
                actor_encoder_params,
                critic_encoder_params,
                policy_params,
                critic_params,
                fixed_actor_encoder_params,
                fixed_critic_encoder_params,
                fixed_actor_encoder_target_params,
                fixed_critic_encoder_target_params,
                target_policy_params,
                target_critic_params,
                actor_encoder_opt_state,
                critic_encoder_opt_state,
                opt_policy_state,
                opt_critic_state,
                key,
                step,
                **batch,
            )
            return (
                actor_encoder_params,
                critic_encoder_params,
                policy_params,
                critic_params,
                fixed_actor_encoder_params,
                fixed_critic_encoder_params,
                fixed_actor_encoder_target_params,
                fixed_critic_encoder_target_params,
                target_policy_params,
                target_critic_params,
                actor_encoder_opt_state,
                critic_encoder_opt_state,
                opt_policy_state,
                opt_critic_state,
            ), (repr_loss, loss, t_mean, priorities)

        return jax.lax.scan(train_one, carry, (keys, steps, data))

    def _aggregate_train_reports(self, reports):
        counts = jnp.array([report.update_count for report in reports])
        total = sum(report.update_count for report in reports)
        mean_repr_loss = (
            jnp.sum(jnp.array([report.metrics["loss/encoder_loss"] for report in reports]) * counts)
            / total
        )
        mean_loss = jnp.sum(jnp.array([report.loss for report in reports]) * counts) / total
        mean_target = jnp.sum(jnp.array([report.target for report in reports]) * counts) / total
        return DPGTrainReport(
            loss=mean_loss,
            target=mean_target,
            update_count=total,
            metrics={
                "loss/encoder_loss": mean_repr_loss,
                "loss/min_value": self.critic_params["values"]["min_value"],
                "loss/max_value": self.critic_params["values"]["max_value"],
            },
        )

    def _train_step(
        self,
        actor_encoder_params,
        critic_encoder_params,
        policy_params,
        critic_params,
        fixed_actor_encoder_params,
        fixed_critic_encoder_params,
        fixed_actor_encoder_target_params,
        fixed_critic_encoder_target_params,
        target_policy_params,
        target_critic_params,
        actor_encoder_opt_state,
        critic_encoder_opt_state,
        opt_policy_state,
        opt_critic_state,
        key,
        step,
        obses,
        actions,
        rewards,
        nxtobses,
        terminateds,
        weights=1,
        indexes=None,
    ):
        obses = convert_normalized_obs(obses)
        nxtobses = convert_normalized_obs(nxtobses)
        repr_loss, (actor_encoder_grad, critic_encoder_grad) = jax.value_and_grad(
            self._encoder_loss, argnums=(0, 1)
        )(actor_encoder_params, critic_encoder_params, obses, nxtobses, actions, key)
        updates, actor_encoder_opt_state = self.optimizer.update(
            actor_encoder_grad, actor_encoder_opt_state, params=actor_encoder_params
        )
        actor_encoder_params = optax.apply_updates(actor_encoder_params, updates)
        updates, critic_encoder_opt_state = self.optimizer.update(
            critic_encoder_grad, critic_encoder_opt_state, params=critic_encoder_params
        )
        critic_encoder_params = optax.apply_updates(critic_encoder_params, updates)

        targets = self._target(
            fixed_actor_encoder_target_params,
            fixed_critic_encoder_target_params,
            target_policy_params,
            target_critic_params,
            rewards,
            nxtobses,
            1.0 - terminateds,
            key,
        )
        critic_params["values"]["min_value"] = jnp.minimum(
            jnp.min(targets), critic_params["values"]["min_value"]
        )
        critic_params["values"]["max_value"] = jnp.maximum(
            jnp.max(targets), critic_params["values"]["max_value"]
        )
        actor_feature, actor_zs = self.actor_encoder(fixed_actor_encoder_params, key, obses)
        critic_feature, critic_zs = self.critic_encoder(
            fixed_critic_encoder_params, fixed_actor_encoder_params, key, obses
        )
        (critic_loss, priority), grad = jax.value_and_grad(self._critic_loss, has_aux=True)(
            critic_params,
            fixed_critic_encoder_params,
            critic_feature,
            critic_zs,
            actions,
            targets,
            key,
        )
        updates, opt_critic_state = self.optimizer.update(
            grad, opt_critic_state, params=critic_params
        )
        critic_params = optax.apply_updates(critic_params, updates)

        def update_actor(state):
            policy_params, opt_policy_state = state
            grad = jax.grad(self._actor_loss)(
                policy_params,
                critic_params,
                fixed_critic_encoder_params,
                actor_feature,
                actor_zs,
                critic_feature,
                critic_zs,
                key,
            )
            updates, opt_policy_state = self.optimizer.update(
                grad, opt_policy_state, params=policy_params
            )
            return optax.apply_updates(policy_params, updates), opt_policy_state

        policy_params, opt_policy_state = jax.lax.cond(
            step % self.policy_delay == 0,
            update_actor,
            lambda state: state,
            (policy_params, opt_policy_state),
        )
        target_policy_params = hard_update(
            policy_params, target_policy_params, step, self.target_network_update_freq
        )
        target_critic_params = hard_update(
            critic_params, target_critic_params, step, self.target_network_update_freq
        )
        fixed_actor_encoder_target_params = hard_update(
            fixed_actor_encoder_params,
            fixed_actor_encoder_target_params,
            step,
            self.target_network_update_freq,
        )
        fixed_critic_encoder_target_params = hard_update(
            fixed_critic_encoder_params,
            fixed_critic_encoder_target_params,
            step,
            self.target_network_update_freq,
        )
        fixed_actor_encoder_params = hard_update(
            actor_encoder_params,
            fixed_actor_encoder_params,
            step,
            self.target_network_update_freq,
        )
        fixed_critic_encoder_params = hard_update(
            critic_encoder_params,
            fixed_critic_encoder_params,
            step,
            self.target_network_update_freq,
        )
        if self.scaled_by_reset:
            policy_params, opt_policy_state = scaled_by_reset(
                policy_params,
                opt_policy_state,
                self.optimizer,
                key,
                step,
                self.reset_freq,
                0.1,
            )
            critic_params, opt_critic_state = scaled_by_reset(
                critic_params,
                opt_critic_state,
                self.optimizer,
                key,
                step,
                self.reset_freq,
                0.1,
            )
        return (
            actor_encoder_params,
            critic_encoder_params,
            policy_params,
            critic_params,
            fixed_actor_encoder_params,
            fixed_critic_encoder_params,
            fixed_actor_encoder_target_params,
            fixed_critic_encoder_target_params,
            target_policy_params,
            target_critic_params,
            actor_encoder_opt_state,
            critic_encoder_opt_state,
            opt_policy_state,
            opt_critic_state,
            repr_loss,
            critic_loss,
            jnp.mean(targets),
            priority,
        )

    def _encoder_loss(
        self,
        actor_encoder_params,
        critic_encoder_params,
        obses,
        next_obses,
        actions,
        key,
    ):
        _, next_actor_zs = self.actor_encoder(actor_encoder_params, key, next_obses)
        _, actor_zs = self.actor_encoder(actor_encoder_params, key, obses)
        pred_actor_zs = self.actor_action_encoder(actor_encoder_params, key, actor_zs, actions)
        _, next_critic_zs = self.critic_encoder(
            critic_encoder_params, actor_encoder_params, key, next_obses
        )
        _, critic_zs = self.critic_encoder(critic_encoder_params, actor_encoder_params, key, obses)
        pred_critic_zs = self.critic_action_encoder(critic_encoder_params, key, critic_zs, actions)
        return jnp.mean(
            jnp.square(jax.lax.stop_gradient(next_actor_zs) - pred_actor_zs)
        ) + jnp.mean(jnp.square(jax.lax.stop_gradient(next_critic_zs) - pred_critic_zs))

    def _actor_loss(
        self,
        policy_params,
        critic_params,
        fixed_critic_encoder_params,
        actor_feature,
        actor_zs,
        critic_feature,
        critic_zs,
        key,
    ):
        actions = self.actor(policy_params, key, actor_feature, actor_zs)
        zsa = self.critic_action_encoder(fixed_critic_encoder_params, key, critic_zs, actions)
        q1, q2 = self.critic(critic_params, key, critic_feature, critic_zs, zsa, actions)
        return -jnp.mean(jnp.minimum(q1, q2))

    def _critic_loss(
        self,
        critic_params,
        fixed_critic_encoder_params,
        feature,
        zs,
        actions,
        targets,
        key,
    ):
        zsa = self.critic_action_encoder(fixed_critic_encoder_params, key, zs, actions)
        q1, q2 = self.critic(critic_params, key, feature, zs, zsa, actions)
        error1 = jnp.squeeze(q1 - targets)
        error2 = jnp.squeeze(q2 - targets)
        critic_loss = jnp.mean(hubberloss(error1, 1.0)) + jnp.mean(hubberloss(error2, 1.0))
        priority = jnp.maximum(jnp.maximum(jnp.abs(error1), jnp.abs(error2)), 1.0)
        return critic_loss, priority

    def _target(
        self,
        fixed_actor_encoder_target_params,
        fixed_critic_encoder_target_params,
        target_policy_params,
        target_critic_params,
        rewards,
        nxtobses,
        not_terminateds,
        key,
    ):
        actor_feature, actor_zs = self.actor_encoder(
            fixed_actor_encoder_target_params, key, nxtobses
        )
        critic_feature, critic_zs = self.critic_encoder(
            fixed_critic_encoder_target_params, fixed_actor_encoder_target_params, key, nxtobses
        )
        next_action = jnp.clip(
            self.actor(target_policy_params, key, actor_feature, actor_zs)
            + jnp.clip(
                self.target_action_noise
                * jax.random.normal(key, (self.batch_size, self.action_size[0])),
                -self.action_noise_clamp,
                self.action_noise_clamp,
            ),
            -1.0,
            1.0,
        )
        critic_zsa = self.critic_action_encoder(
            fixed_critic_encoder_target_params, key, critic_zs, next_action
        )
        q1, q2 = self.critic(
            target_critic_params,
            key,
            critic_feature,
            critic_zs,
            critic_zsa,
            next_action,
        )
        next_q = jnp.clip(
            jnp.minimum(q1, q2),
            target_critic_params["values"]["min_value"],
            target_critic_params["values"]["max_value"],
        )
        return rewards + not_terminateds * self.gamma * next_q

    def description(self, eval_result=None):
        description = ""
        if eval_result is not None:
            for k, v in eval_result.items():
                description += f"{k} : {v:8.2f}, "

        array_module = jnp if any(isinstance(loss, jax.Array) for loss in self.lossque) else np
        description += f"loss : {array_module.mean(array_module.asarray(tuple(self.lossque))):.3f}"
        description += self._rollout_pbar_suffix()
        return description

    def run_name_update(self, run_name):
        if self.simba:
            run_name = "Simba_" + run_name
        if self.n_step_method:
            run_name = f"{self.n_step}Step_" + run_name
        return run_name
