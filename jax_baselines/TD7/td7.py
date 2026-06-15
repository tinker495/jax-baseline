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
from jax_baselines.math.losses import hubberloss
from jax_baselines.math.param_updates import hard_update, scaled_by_reset


@struct.dataclass
class TD7CheckpointParams:
    encoder_params: Any
    policy_params: Any
    critic_params: Any
    fixed_encoder_params: Any
    fixed_encoder_target_params: Any
    target_policy_params: Any
    target_critic_params: Any


class TD7(Deteministic_Policy_Gradient_Family):
    def __init__(
        self,
        env_builder: callable,
        model_builder_maker,
        target_action_noise_mul=2.0,
        action_noise=0.1,
        policy_delay=2,
        target_network_update_freq=250,
        **kwargs,
    ):
        # Set TD7-specific defaults - always enable checkpointing
        td7_kwargs = {
            "n_step": 1,
            "target_network_update_tau": 0,
            "prioritized_replay": True,
            "prioritized_replay_beta0": 0,
            "prioritized_replay_eps": 0,
            "use_checkpointing": True,  # TD7 always uses checkpointing
            **kwargs,
        }

        super().__init__(env_builder, model_builder_maker, **td7_kwargs)

        self.name = "TD7"
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
            self.preproc,
            self.encoder,
            self.action_encoder,
            self.actor,
            self.critic,
            self.encoder_params,
            self.policy_params,
            self.critic_params,
        ) = model_builder(next(self.key_seq), print_model=True)
        self.fixed_encoder_params = deepcopy(self.encoder_params)
        self.fixed_encoder_target_params = deepcopy(self.encoder_params)
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

        self.encoder_opt_state = self.optimizer.init(self.encoder_params)
        self.opt_policy_state = self.optimizer.init(self.policy_params)
        self.opt_critic_state = self.optimizer.init(self.critic_params)
        self._get_actions = jax.jit(self._get_actions)
        self._train_step = jax.jit(self._train_step)

    def checkpoint_params(self):
        return TD7CheckpointParams(
            encoder_params=self.encoder_params,
            policy_params=self.policy_params,
            critic_params=self.critic_params,
            fixed_encoder_params=self.fixed_encoder_params,
            fixed_encoder_target_params=self.fixed_encoder_target_params,
            target_policy_params=self.target_policy_params,
            target_critic_params=self.target_critic_params,
        )

    def load_checkpoint_params(self, bundle):
        self.encoder_params = bundle.encoder_params
        self.policy_params = bundle.policy_params
        self.critic_params = bundle.critic_params
        self.fixed_encoder_params = bundle.fixed_encoder_params
        self.fixed_encoder_target_params = bundle.fixed_encoder_target_params
        self.target_policy_params = bundle.target_policy_params
        self.target_critic_params = bundle.target_critic_params

    def _get_actions(self, encoder_params, policy_params, obses, key=None) -> jnp.ndarray:
        feature = self.preproc(encoder_params, key, convert_jax(obses))
        zs = self.encoder(encoder_params, key, feature)
        return self.actor(policy_params, key, feature, zs)

    def _select_action_state(self, eval, steps):
        if eval and self.use_checkpointing and self.eval_snapshot is not None:
            return self.eval_snapshot
        return self.get_behavior_state()

    def _policy_action_from_state(self, state, obs, eval, steps):
        return np.asarray(self._get_actions(state["encoder"], state["policy"], obs, None))

    def _apply_action_noise(self, actions, steps, eval):
        if eval:
            return actions
        return np.clip(
            actions
            + self.action_noise
            * np.random.normal(0, 1, size=(self.worker_size, self.action_size[0])),
            -1,
            1,
        )

    def _train_on_batch(self, data, context):
        (
            self.encoder_params,
            self.policy_params,
            self.critic_params,
            self.fixed_encoder_params,
            self.fixed_encoder_target_params,
            self.target_policy_params,
            self.target_critic_params,
            self.encoder_opt_state,
            self.opt_policy_state,
            self.opt_critic_state,
            repr_loss,
            loss,
            t_mean,
            new_priorities,
        ) = self._train_step(
            self.encoder_params,
            self.policy_params,
            self.critic_params,
            self.fixed_encoder_params,
            self.fixed_encoder_target_params,
            self.target_policy_params,
            self.target_critic_params,
            self.encoder_opt_state,
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

    def _aggregate_train_reports(self, reports):
        mean_repr_loss = jnp.mean(
            jnp.array([report.metrics["loss/encoder_loss"] for report in reports])
        )
        mean_loss = jnp.mean(jnp.array([report.loss for report in reports]))
        mean_target = jnp.mean(jnp.array([report.target for report in reports]))
        return DPGTrainReport(
            loss=mean_loss,
            target=mean_target,
            metrics={
                "loss/encoder_loss": mean_repr_loss,
                "loss/min_value": self.critic_params["values"]["min_value"],
                "loss/max_value": self.critic_params["values"]["max_value"],
            },
        )

    def _train_step(
        self,
        encoder_params,
        policy_params,
        critic_params,
        fixed_encoder_params,
        fixed_encoder_target_params,
        target_policy_params,
        target_critic_params,
        encoder_opt_state,
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
        obses = convert_jax(obses)
        nxtobses = convert_jax(nxtobses)
        not_terminateds = 1.0 - terminateds

        repr_loss, grad = jax.value_and_grad(self._encoder_loss)(
            encoder_params, obses, nxtobses, actions, key
        )
        updates, encoder_opt_state = self.optimizer.update(
            grad, encoder_opt_state, params=encoder_params
        )
        encoder_params = optax.apply_updates(encoder_params, updates)

        targets = self._target(
            fixed_encoder_target_params,
            target_policy_params,
            target_critic_params,
            rewards,
            nxtobses,
            not_terminateds,
            key,
        )
        critic_params["values"]["min_value"] = jnp.minimum(
            jnp.min(targets), critic_params["values"]["min_value"]
        )
        critic_params["values"]["max_value"] = jnp.maximum(
            jnp.max(targets), critic_params["values"]["max_value"]
        )

        feature, zs = self.feature_and_zs(fixed_encoder_params, obses, key)

        (critic_loss, priority), grad = jax.value_and_grad(self._critic_loss, has_aux=True)(
            critic_params, fixed_encoder_params, feature, zs, actions, targets, key
        )
        updates, opt_critic_state = self.optimizer.update(
            grad, opt_critic_state, params=critic_params
        )
        critic_params = optax.apply_updates(critic_params, updates)

        def _opt_actor(policy_params, opt_policy_state, key):
            grad = jax.grad(self._actor_loss)(
                policy_params, critic_params, fixed_encoder_params, feature, zs, key
            )
            updates, opt_policy_state = self.optimizer.update(
                grad, opt_policy_state, params=policy_params
            )
            policy_params = optax.apply_updates(policy_params, updates)
            return policy_params, opt_policy_state, key

        policy_params, opt_policy_state, key = jax.lax.cond(
            step % self.policy_delay == 0,
            lambda x: _opt_actor(*x),
            lambda x: x,
            (policy_params, opt_policy_state, key),
        )

        target_policy_params = hard_update(
            policy_params, target_policy_params, step, self.target_network_update_freq
        )
        target_critic_params = hard_update(
            critic_params, target_critic_params, step, self.target_network_update_freq
        )
        fixed_encoder_target_params = hard_update(
            fixed_encoder_params,
            fixed_encoder_target_params,
            step,
            self.target_network_update_freq,
        )
        fixed_encoder_params = hard_update(
            encoder_params, fixed_encoder_params, step, self.target_network_update_freq
        )
        if self.scaled_by_reset:
            policy_params = scaled_by_reset(
                policy_params,
                opt_policy_state,
                self.optimizer,
                key,
                step,
                self.reset_freq,
                0.1,  # tau = 0.1 is softreset, but original paper uses 1.0
            )
            critic_params = scaled_by_reset(
                critic_params,
                opt_critic_state,
                self.optimizer,
                key,
                step,
                self.reset_freq,
                0.1,  # tau = 0.1 is softreset, but original paper uses 1.0
            )
        return (
            encoder_params,
            policy_params,
            critic_params,
            fixed_encoder_params,
            fixed_encoder_target_params,
            target_policy_params,
            target_critic_params,
            encoder_opt_state,
            opt_policy_state,
            opt_critic_state,
            repr_loss,
            critic_loss,
            jnp.mean(targets),
            priority,
        )

    def feature_and_zs(self, encoder_params, obses, key):
        feature = self.preproc(encoder_params, key, convert_jax(obses))
        zs = self.encoder(encoder_params, key, feature)
        return feature, zs

    def _encoder_loss(self, encoder_params, obses, next_obses, actions, key):
        next_zs = jax.lax.stop_gradient(
            self.encoder(encoder_params, key, self.preproc(encoder_params, key, next_obses))
        )
        zs = self.encoder(encoder_params, key, self.preproc(encoder_params, key, obses))
        pred_zs = self.action_encoder(encoder_params, key, zs, actions)
        loss = jnp.mean(jnp.square(next_zs - pred_zs))
        return loss

    def _actor_loss(self, policy_params, critic_params, fixed_encoder_params, feature, zs, key):
        actions = self.actor(policy_params, key, feature, zs)
        zsa = self.action_encoder(fixed_encoder_params, key, zs, actions)
        q1, q2 = self.critic(critic_params, key, feature, zs, zsa, actions)
        return -jnp.mean(jnp.minimum(q1, q2))

    def _critic_loss(self, critic_params, fixed_encoder_params, feature, zs, actions, targets, key):
        zsa = self.action_encoder(fixed_encoder_params, key, zs, actions)

        q1, q2 = self.critic(critic_params, key, feature, zs, zsa, actions)
        error1 = jnp.squeeze(q1 - targets)
        error2 = jnp.squeeze(q2 - targets)
        critic_loss = jnp.mean(hubberloss(error1, 1.0)) + jnp.mean(hubberloss(error2, 1.0))

        priority = jnp.maximum(jnp.maximum(jnp.abs(error1), jnp.abs(error2)), 1.0)
        return critic_loss, priority

    def _target(
        self,
        fixed_encoder_target_params,
        target_policy_params,
        target_critic_params,
        rewards,
        nxtobses,
        not_terminateds,
        key,
    ):
        next_feature = self.preproc(fixed_encoder_target_params, key, nxtobses)
        fixed_target_zs = self.encoder(fixed_encoder_target_params, key, next_feature)

        next_action = jnp.clip(
            self.actor(target_policy_params, key, next_feature, fixed_target_zs)
            + jnp.clip(
                self.target_action_noise
                * jax.random.normal(key, (self.batch_size, self.action_size[0])),
                -self.action_noise_clamp,
                self.action_noise_clamp,
            ),
            -1.0,
            1.0,
        )

        fixed_target_zsa = self.action_encoder(
            fixed_encoder_target_params, key, fixed_target_zs, next_action
        )

        q1, q2 = self.critic(
            target_critic_params,
            key,
            next_feature,
            fixed_target_zs,
            fixed_target_zsa,
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

        description += f"loss : {np.mean(self.lossque):.3f}"
        description += self._rollout_pbar_suffix()
        return description

    def run_name_update(self, run_name):
        if self.simba:
            run_name = "Simba_" + run_name
        if self.n_step_method:
            run_name = "{}Step_".format(self.n_step) + run_name
        return run_name

    def learn(
        self,
        total_timesteps,
        callback=None,
        log_interval=1000,
        experiment_name="TD7",
        run_name="TD7",
        eval_num=100,
    ):
        super().learn(
            total_timesteps,
            callback,
            log_interval,
            experiment_name,
            run_name,
            eval_num,
        )
