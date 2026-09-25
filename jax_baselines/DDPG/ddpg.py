from collections.abc import Callable
from copy import deepcopy
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import struct

from jax_baselines.DDPG.base_class import Deteministic_Policy_Gradient_Family
from jax_baselines.DDPG.metrics import critic_metrics
from jax_baselines.DDPG.ou_noise import ou_step
from jax_baselines.math.jax_utils import convert_normalized_obs
from jax_baselines.math.param_updates import scaled_by_reset, soft_update
from jax_baselines.optim import optimizer_metrics


@struct.dataclass
class DDPGCheckpointParams:
    policy_params: Any
    critic_params: Any
    target_policy_params: Any
    target_critic_params: Any


class DDPG(Deteministic_Policy_Gradient_Family):
    _run_name = "DDPG"

    def __init__(
        self,
        env_builder: Callable,
        model_builder_maker,
        exploration_fraction=0.3,
        exploration_final_eps=0.02,
        exploration_initial_eps=1.0,
        **kwargs,
    ):

        self.exploration_final_eps = exploration_final_eps
        self.exploration_initial_eps = exploration_initial_eps
        self.exploration_fraction = exploration_fraction

        super().__init__(env_builder, model_builder_maker, **kwargs)

        with jax.default_device(self.memory_device):
            self._ou_noise = 0.2 * jax.random.normal(
                next(self.key_seq), (self.worker_size, self.action_size[0])
            )

    def setup_model(self):
        model_builder = self.model_builder_maker(
            self.observation_space,
            self.action_size,
            self.policy_kwargs,
        )
        (
            self.actor,
            self.critic,
            self.policy_params,
            self.critic_params,
        ) = model_builder(next(self.key_seq), print_model=True)
        self.target_policy_params = deepcopy(self.policy_params)
        self.target_critic_params = deepcopy(self.critic_params)

        self.opt_policy_state = self.optimizer.init(self.policy_params)
        self.opt_critic_state = self.optimizer.init(self.critic_params)

    def checkpoint_params(self):
        return DDPGCheckpointParams(
            policy_params=self.policy_params,
            critic_params=self.critic_params,
            target_policy_params=self.target_policy_params,
            target_critic_params=self.target_critic_params,
        )

    def load_checkpoint_params(self, bundle):
        self.policy_params = bundle.policy_params
        self.critic_params = bundle.critic_params
        self.target_policy_params = bundle.target_policy_params
        self.target_critic_params = bundle.target_critic_params

    def _get_eval_actions(self, state, obses):
        return self.actor(state["policy"], None, convert_normalized_obs(obses))

    def _get_actions(self, state, obses, key, noise, step, exploration_steps):
        key, noise_key = jax.random.split(key)
        noise = ou_step(noise, noise_key)
        step = step + self.worker_size
        progress = jnp.where(
            exploration_steps > 0, jnp.clip(step / exploration_steps, 0.0, 1.0), 0.0
        )
        epsilon = self.exploration_initial_eps + progress * (
            self.exploration_final_eps - self.exploration_initial_eps
        )
        actions = self._get_eval_actions(state, obses)
        return jnp.clip(actions + noise * epsilon, -1, 1), key, noise, step, epsilon

    def description(self, eval_result=None):
        description = ""
        if eval_result is not None:
            for k, v in eval_result.items():
                description += f"{k} : {v:8.2f}, "

        description += f"loss : {np.mean(jax.device_get(tuple(self.lossque))):.3f}"
        description += f", epsilon : {jax.device_get(self.epsilon):.3f}"
        description += self._rollout_pbar_suffix()
        return description

    def _policy_action_from_state(self, state, obs, eval, steps):
        if eval:
            return self._compiled_eval_actions(state, obs)
        (
            actions,
            self._action_key,
            self._ou_noise,
            self._exploration_step,
            self.epsilon,
        ) = self._compiled_actions(
            state,
            obs,
            self._action_key,
            self._ou_noise,
            self._exploration_step,
            self._exploration_steps,
        )
        return actions

    def prepare_run(self, total_timesteps):
        # Rollouts advance `steps` by worker_size per action call and act from the policy
        # after learning_starts, so the schedule step is carried on device from here.
        self._exploration_steps, self._exploration_step = jax.device_put(
            (
                np.int32(int(self.exploration_fraction * total_timesteps)),
                np.int32(self.learning_starts // self.worker_size * self.worker_size),
            ),
            self.memory_device,
        )
        self.epsilon = self.exploration_initial_eps

    def test_action(self, obs):
        return self.actions(obs, np.inf, eval=True)

    @property
    def _train_state(self):
        return (
            self.policy_params,
            self.critic_params,
            self.target_policy_params,
            self.target_critic_params,
            self.opt_policy_state,
            self.opt_critic_state,
        )

    @_train_state.setter
    def _train_state(self, state):
        (
            self.policy_params,
            self.critic_params,
            self.target_policy_params,
            self.target_critic_params,
            self.opt_policy_state,
            self.opt_critic_state,
        ) = state

    def _train_step(
        self, state, key, step, flags, obses, actions, rewards, nxtobses, terminateds, weights=1
    ):
        (
            policy_params,
            critic_params,
            target_policy_params,
            target_critic_params,
            opt_policy_state,
            opt_critic_state,
        ) = state
        obses = convert_normalized_obs(obses)
        nxtobses = convert_normalized_obs(nxtobses)
        not_terminateds = 1.0 - terminateds

        targets = self._target(
            target_policy_params,
            target_critic_params,
            rewards,
            nxtobses,
            not_terminateds,
            key,
        )
        (critic_loss, (abs_error, metrics)), grad = jax.value_and_grad(
            self._critic_loss, has_aux=True
        )(critic_params, policy_params, obses, actions, targets, weights, key, flags.diagnostics)
        updates, opt_critic_state = self.optimizer.update(
            grad, opt_critic_state, params=critic_params, diagnostics=flags.diagnostics
        )
        critic_params = optax.apply_updates(critic_params, updates)

        actor_loss, grad = jax.value_and_grad(self._actor_loss)(
            policy_params, critic_params, obses, key
        )
        updates, opt_policy_state = self.optimizer.update(
            grad, opt_policy_state, params=policy_params, diagnostics=flags.diagnostics
        )
        policy_params = optax.apply_updates(policy_params, updates)
        metrics["loss/qloss"] = critic_loss
        if flags.diagnostics:
            metrics.update(optimizer_metrics(opt_critic_state, "critic"))
            metrics.update(optimizer_metrics(opt_policy_state, "actor"))
            metrics.update(
                {
                    "loss/actor_loss": actor_loss,
                    "loss/actor_q_mean": -actor_loss,
                    "loss/targets": jnp.mean(targets),
                }
            )
        metric_counts = {name: jnp.asarray(1) for name in metrics}

        target_critic_params = soft_update(
            critic_params, target_critic_params, self.target_network_update_tau
        )
        target_policy_params = soft_update(
            policy_params, target_policy_params, self.target_network_update_tau
        )
        policy_params, opt_policy_state = scaled_by_reset(
            policy_params,
            opt_policy_state,
            self.optimizer,
            key,
            flags.reset,
            0.1,  # tau = 0.1 is softreset, but original paper uses 1.0
        )
        critic_params, opt_critic_state = scaled_by_reset(
            critic_params,
            opt_critic_state,
            self.optimizer,
            key,
            flags.reset,
            0.1,  # tau = 0.1 is softreset, but original paper uses 1.0
        )
        return (
            policy_params,
            critic_params,
            target_policy_params,
            target_critic_params,
            opt_policy_state,
            opt_critic_state,
        ), (abs_error if self.prioritized_replay else None, metrics, metric_counts)

    def _critic_loss(
        self, critic_params, policy_params, obses, actions, targets, weights, key, diagnostics
    ):
        vals = self.critic(critic_params, policy_params, key, obses, actions)
        error = jnp.squeeze(vals - targets)
        critic_loss = jnp.mean(weights * jnp.square(error))
        metrics = (
            critic_metrics(
                (vals,),
                targets,
                (jnp.square(error),),
                weights,
                jnp.abs(error) if self.prioritized_replay else None,
            )
            if diagnostics
            else {}
        )
        return critic_loss, (jnp.abs(error), metrics)

    def _actor_loss(self, policy_params, critic_params, obses, key):
        actions = self.actor(policy_params, key, obses)
        q = self.critic(critic_params, policy_params, key, obses, actions)
        return -jnp.mean(q)

    def _target(
        self,
        target_policy_params,
        target_critic_params,
        rewards,
        nxtobses,
        not_terminateds,
        key,
    ):
        next_action = self.actor(target_policy_params, key, nxtobses)
        next_q = self.critic(target_critic_params, target_policy_params, key, nxtobses, next_action)
        return (not_terminateds * next_q * self._gamma) + rewards
