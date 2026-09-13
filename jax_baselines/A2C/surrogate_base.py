import jax
import jax.numpy as jnp
import numpy as np
import optax

from jax_baselines.A2C.base_class import Actor_Critic_Policy_Gradient_Family
from jax_baselines.math.jax_utils import convert_normalized_obs
from jax_baselines.math.metrics import rollout_metrics
from jax_baselines.math.returns import (
    get_gaes,
    normalize_advantage,
    validate_advantage_normalize_scope,
)
from jax_baselines.optim import optimizer_metrics


class SurrogatePolicyGradient(Actor_Critic_Policy_Gradient_Family):
    """Shared minibatch-epoch surrogate policy-gradient machinery for PPO and SPO.

    PPO and SPO share identical rollout preprocessing (GAE) and the
    minibatch/epoch optimization loop; they differ only in the per-sample actor
    loss supplied through ``_actor_loss_discrete``/``_actor_loss_continuous``
    (wired to ``self._actor_loss`` by ``Actor_Critic_Policy_Gradient_Family``).
    """

    _store_old_policy = True

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
        batch_size=256,
        learning_rate=3e-4,
        lr_annealing=False,
        desired_kl: float | None = None,
        **kwargs,
    ):
        if desired_kl is not None:
            if not np.isfinite(desired_kl) or desired_kl <= 0:
                raise ValueError("desired_kl must be finite and positive")
            if lr_annealing or callable(learning_rate):
                raise ValueError("desired_kl requires a scalar learning rate without lr_annealing")
            if not np.isfinite(learning_rate) or learning_rate <= 0:
                raise ValueError("desired_kl requires a finite positive learning rate")
        self.desired_kl = desired_kl
        self.lamda = lamda
        self.gae_normalize = gae_normalize
        self.gae_normalize_scope = validate_advantage_normalize_scope(gae_normalize_scope)
        self.ppo_eps = ppo_eps
        self.value_clip = value_clip
        self.minibatch_size = minibatch_size
        self.epoch_num = epoch_num

        super().__init__(
            env_builder,
            model_builder_maker,
            batch_size=batch_size,
            learning_rate=learning_rate,
            lr_annealing=lr_annealing,
            **kwargs,
        )

        self.batch_size = int(
            np.ceil(batch_size * self.worker_size / minibatch_size)
            * minibatch_size
            / self.worker_size
        )
        self.get_memory_setup()

    def _make_optimizer(self, learning_rate):
        if self.desired_kl is not None:
            return optax.inject_hyperparams(super()._make_optimizer)(learning_rate=learning_rate)
        return super()._make_optimizer(learning_rate)

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
            metrics,
        ) = self._train_step(
            self.actor_params,
            self.critic_params,
            self.actor_opt_state,
            self.critic_opt_state,
            next(self.key_seq),
            **data,
        )

        if logger_run:
            for name, value in metrics.items():
                logger_run.log_metric(name, value, steps)

        return metrics["loss/critic_loss"]

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
        old_policy,
    ):
        obses = convert_normalized_obs(obses)
        nxtobses = convert_normalized_obs(nxtobses)
        value = jax.vmap(self.critic, in_axes=(None, None, None, 0))(
            critic_params, actor_params, key, obses
        )
        next_value = jax.vmap(self.critic, in_axes=(None, None, None, 0))(
            critic_params, actor_params, key, nxtobses
        )
        pi_prob, old_distribution = old_policy
        if self.action_type == "continuous":
            old_mu, old_log_std = old_distribution
            old_distribution = (old_mu, jnp.exp(old_log_std))
        adv = jax.vmap(get_gaes, in_axes=(0, 0, 0, 0, 0, None, None))(
            rewards, terminateds, truncateds, value, next_value, self.gamma, self.lamda
        )
        obses = {key: jnp.vstack(value) for key, value in obses.items()}
        actions = jnp.vstack(actions)
        value = jnp.vstack(value)
        old_policy = (jnp.vstack(pi_prob), jax.tree.map(jnp.vstack, old_distribution))
        adv = jnp.vstack(adv)
        targets = value + adv
        metrics = rollout_metrics(value, targets, adv)
        if self.gae_normalize and self.gae_normalize_scope == "batch":
            adv = normalize_advantage(adv)
        return obses, actions, value, targets, old_policy, adv, metrics

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
        old_policy,
    ):
        obses, actions, old_values, targets, old_policy, adv, metrics = self._preprocess(
            actor_params,
            critic_params,
            key,
            obses,
            actions,
            rewards,
            nxtobses,
            terminateds,
            truncateds,
            old_policy,
        )

        def i_f(vals, _):
            actor_params, critic_params, actor_opt_state, critic_opt_state, key = vals
            use_key, key = jax.random.split(key)
            batch_idxes = jax.random.permutation(use_key, jnp.arange(targets.shape[0])).reshape(
                -1, self.minibatch_size
            )
            obses_batch = {key: value[batch_idxes] for key, value in obses.items()}
            actions_batch = actions[batch_idxes]
            old_values_batch = old_values[batch_idxes]
            targets_batch = targets[batch_idxes]
            old_policy_batch = jax.tree.map(lambda value: value[batch_idxes], old_policy)
            adv_batch = adv[batch_idxes]

            def f(updates, input):
                actor_params, critic_params, actor_opt_state, critic_opt_state, key = updates
                obs, act, oldv, target, old_policy, adv = input
                if self.gae_normalize and self.gae_normalize_scope == "minibatch":
                    adv = normalize_advantage(adv)
                use_key, key = jax.random.split(key)
                (actor_objective, batch_metrics), actor_grad = jax.value_and_grad(
                    self._actor_loss, has_aux=True
                )(actor_params, obs, act, old_policy, adv, use_key)
                c_loss, critic_grad = jax.value_and_grad(self._critic_loss)(
                    critic_params, actor_params, obs, oldv, target, use_key
                )
                if self.desired_kl is not None:
                    kl = batch_metrics["loss/kl_divergence"]
                    learning_rate = actor_opt_state.hyperparams["learning_rate"]
                    learning_rate = jnp.where(
                        kl > 2.0 * self.desired_kl,
                        jnp.maximum(1e-5, learning_rate / 1.5),
                        jnp.where(
                            (kl > 0.0) & (kl < self.desired_kl / 2.0),
                            jnp.minimum(1e-2, learning_rate * 1.5),
                            learning_rate,
                        ),
                    )
                    actor_opt_state = actor_opt_state._replace(
                        hyperparams={**actor_opt_state.hyperparams, "learning_rate": learning_rate}
                    )
                    critic_opt_state = critic_opt_state._replace(
                        hyperparams={**critic_opt_state.hyperparams, "learning_rate": learning_rate}
                    )
                actor_updates, actor_opt_state = self.optimizer.update(
                    actor_grad, actor_opt_state, params=actor_params
                )
                critic_updates, critic_opt_state = self.optimizer.update(
                    critic_grad, critic_opt_state, params=critic_params
                )
                actor_params = optax.apply_updates(actor_params, actor_updates)
                critic_params = optax.apply_updates(critic_params, critic_updates)
                batch_metrics["loss/critic_loss"] = c_loss
                batch_metrics["loss/actor_objective"] = actor_objective
                batch_metrics.update(optimizer_metrics(actor_opt_state, "actor"))
                batch_metrics.update(optimizer_metrics(critic_opt_state, "critic"))
                return (
                    actor_params,
                    critic_params,
                    actor_opt_state,
                    critic_opt_state,
                    key,
                ), batch_metrics

            updates, losses = jax.lax.scan(
                f,
                (actor_params, critic_params, actor_opt_state, critic_opt_state, key),
                (
                    obses_batch,
                    actions_batch,
                    old_values_batch,
                    targets_batch,
                    old_policy_batch,
                    adv_batch,
                ),
            )
            return updates, jax.tree.map(jnp.mean, losses)

        updates, epoch_metrics = jax.lax.scan(
            i_f,
            (actor_params, critic_params, actor_opt_state, critic_opt_state, key),
            None,
            length=self.epoch_num,
        )
        actor_params, critic_params, actor_opt_state, critic_opt_state, key = updates
        metrics.update(jax.tree.map(jnp.mean, epoch_metrics))
        return (
            actor_params,
            critic_params,
            actor_opt_state,
            critic_opt_state,
            metrics,
        )

    def _critic_loss(self, critic_params, actor_params, obses, old_value, targets, key):
        values = self.critic(critic_params, actor_params, key, obses)
        clipped_values = old_value + jnp.clip(values - old_value, -self.value_clip, self.value_clip)
        return jnp.mean(
            jnp.maximum(jnp.square(values - targets), jnp.square(clipped_values - targets))
        )
