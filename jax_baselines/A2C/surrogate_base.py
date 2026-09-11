import jax
import jax.numpy as jnp
import numpy as np
import optax

from jax_baselines.A2C.base_class import Actor_Critic_Policy_Gradient_Family
from jax_baselines.math.jax_utils import convert_normalized_obs
from jax_baselines.math.returns import (
    get_gaes,
    normalize_advantage,
    validate_advantage_normalize_scope,
)


class SurrogatePolicyGradient(Actor_Critic_Policy_Gradient_Family):
    """Shared minibatch-epoch surrogate policy-gradient machinery for PPO and SPO.

    PPO and SPO share identical rollout preprocessing (GAE) and the
    minibatch/epoch optimization loop; they differ only in the per-sample actor
    loss supplied through ``_actor_loss_discrete``/``_actor_loss_continuous``
    (wired to ``self._actor_loss`` by ``Actor_Critic_Policy_Gradient_Family``).
    """

    def __init__(
        self,
        env_builder,
        model_builder_maker,
        lamda=0.95,
        gae_normalize=False,
        gae_normalize_scope="batch",
        minibatch_size=32,
        epoch_num=4,
        ppo_eps=0.2,
        value_clip=2.0,
        **kwargs,
    ):
        self.lamda = lamda
        self.gae_normalize = gae_normalize
        self.gae_normalize_scope = validate_advantage_normalize_scope(gae_normalize_scope)
        self.ppo_eps = ppo_eps
        self.value_clip = value_clip
        self.minibatch_size = minibatch_size
        self.epoch_num = epoch_num

        super().__init__(env_builder, model_builder_maker, **kwargs)

        self.batch_size = int(
            np.ceil(kwargs.get("batch_size", 256) * self.worker_size / minibatch_size)
            * minibatch_size
            / self.worker_size
        )
        self.get_memory_setup()

    def setup_model(self):
        self.model_builder = self.model_builder_maker(
            self.observation_space, self.action_size, self.action_type, self.policy_kwargs
        )

        self.actor, self.critic, self.actor_params, self.critic_params = self.model_builder(
            next(self.key_seq), print_model=True
        )
        self.actor_opt_state = self.optimizer.init(self.actor_params)
        self.critic_opt_state = self.optimizer.init(self.critic_params)

        self._get_actions = jax.jit(self._get_actions)
        self._preprocess = jax.jit(self._preprocess)
        self._train_step = jax.jit(self._train_step)

    def train_step(self, steps, logger_run=None):
        data = self.buffer.get_buffer()

        (
            self.actor_params,
            self.critic_params,
            self.actor_opt_state,
            self.critic_opt_state,
            critic_loss,
            actor_loss,
            entropy_loss,
            targets,
        ) = self._train_step(
            self.actor_params,
            self.critic_params,
            self.actor_opt_state,
            self.critic_opt_state,
            next(self.key_seq),
            **data,
        )

        if logger_run:
            logger_run.log_metric("loss/critic_loss", critic_loss, steps)
            logger_run.log_metric("loss/actor_loss", actor_loss, steps)
            logger_run.log_metric("loss/entropy_loss", entropy_loss, steps)
            logger_run.log_metric("loss/mean_target", targets, steps)

        return critic_loss

    def _preprocess(
        self,
        actor_params,
        critic_params,
        key,
        obses,
        actions,
        rewards,
        nxtobses,
        terminateds,
        truncateds,
    ):
        obses = convert_normalized_obs(obses)
        nxtobses = convert_normalized_obs(nxtobses)
        value = jax.vmap(self.critic, in_axes=(None, None, None, 0))(
            critic_params, actor_params, key, obses
        )
        next_value = jax.vmap(self.critic, in_axes=(None, None, None, 0))(
            critic_params, actor_params, key, nxtobses
        )
        pi_prob = jax.vmap(self.get_logprob, in_axes=(0, 0, None))(
            jax.vmap(self.actor, in_axes=(None, None, 0))(actor_params, key, obses),
            actions,
            key,
        )
        adv = jax.vmap(get_gaes, in_axes=(0, 0, 0, 0, 0, None, None))(
            rewards, terminateds, truncateds, value, next_value, self.gamma, self.lamda
        )
        obses = {key: jnp.vstack(value) for key, value in obses.items()}
        actions = jnp.vstack(actions)
        value = jnp.vstack(value)
        pi_prob = jnp.vstack(pi_prob)
        adv = jnp.vstack(adv)
        targets = value + adv
        if self.gae_normalize and self.gae_normalize_scope == "batch":
            adv = normalize_advantage(adv)
        return obses, actions, value, targets, pi_prob, adv

    def _train_step(
        self,
        actor_params,
        critic_params,
        actor_opt_state,
        critic_opt_state,
        key,
        obses,
        actions,
        rewards,
        nxtobses,
        terminateds,
        truncateds,
    ):
        obses, actions, old_values, targets, act_prob, adv = self._preprocess(
            actor_params,
            critic_params,
            key,
            obses,
            actions,
            rewards,
            nxtobses,
            terminateds,
            truncateds,
        )

        def i_f(idx, vals):
            (
                actor_params,
                critic_params,
                actor_opt_state,
                critic_opt_state,
                key,
                critic_loss,
                actor_loss,
                entropy_loss,
            ) = vals
            use_key, key = jax.random.split(key)
            batch_idxes = jax.random.permutation(use_key, jnp.arange(targets.shape[0])).reshape(
                -1, self.minibatch_size
            )
            obses_batch = {key: value[batch_idxes] for key, value in obses.items()}
            actions_batch = actions[batch_idxes]
            old_values_batch = old_values[batch_idxes]
            targets_batch = targets[batch_idxes]
            act_prob_batch = act_prob[batch_idxes]
            adv_batch = adv[batch_idxes]

            def f(updates, input):
                actor_params, critic_params, actor_opt_state, critic_opt_state, key = updates
                obs, act, oldv, target, act_prob, adv = input
                if self.gae_normalize and self.gae_normalize_scope == "minibatch":
                    adv = normalize_advantage(adv)
                use_key, key = jax.random.split(key)
                (_, (a_loss, entropy_loss)), actor_grad = jax.value_and_grad(
                    self._actor_loss, has_aux=True
                )(actor_params, obs, act, act_prob, adv, use_key)
                c_loss, critic_grad = jax.value_and_grad(self._critic_loss)(
                    critic_params, actor_params, obs, oldv, target, use_key
                )
                actor_updates, actor_opt_state = self.optimizer.update(
                    actor_grad, actor_opt_state, params=actor_params
                )
                critic_updates, critic_opt_state = self.optimizer.update(
                    critic_grad, critic_opt_state, params=critic_params
                )
                actor_params = optax.apply_updates(actor_params, actor_updates)
                critic_params = optax.apply_updates(critic_params, critic_updates)
                return (actor_params, critic_params, actor_opt_state, critic_opt_state, key), (
                    c_loss,
                    a_loss,
                    entropy_loss,
                )

            updates, losses = jax.lax.scan(
                f,
                (actor_params, critic_params, actor_opt_state, critic_opt_state, key),
                (
                    obses_batch,
                    actions_batch,
                    old_values_batch,
                    targets_batch,
                    act_prob_batch,
                    adv_batch,
                ),
            )
            actor_params, critic_params, actor_opt_state, critic_opt_state, key = updates
            cl, al, el = losses
            critic_loss += jnp.mean(cl)
            actor_loss += jnp.mean(al)
            entropy_loss += jnp.mean(el)
            return (
                actor_params,
                critic_params,
                actor_opt_state,
                critic_opt_state,
                key,
                critic_loss,
                actor_loss,
                entropy_loss,
            )

        val = jax.lax.fori_loop(
            0,
            self.epoch_num,
            i_f,
            (actor_params, critic_params, actor_opt_state, critic_opt_state, key, 0.0, 0.0, 0.0),
        )
        (
            actor_params,
            critic_params,
            actor_opt_state,
            critic_opt_state,
            key,
            critic_loss,
            actor_loss,
            entropy_loss,
        ) = val
        return (
            actor_params,
            critic_params,
            actor_opt_state,
            critic_opt_state,
            critic_loss / self.epoch_num,
            actor_loss / self.epoch_num,
            entropy_loss / self.epoch_num,
            jnp.mean(targets),
        )

    def _critic_loss(self, critic_params, actor_params, obses, old_value, targets, key):
        values = self.critic(critic_params, actor_params, key, obses)
        clipped_values = old_value + jnp.clip(values - old_value, -self.value_clip, self.value_clip)
        return jnp.mean(
            jnp.maximum(jnp.square(values - targets), jnp.square(clipped_values - targets))
        )
