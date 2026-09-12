import jax
import jax.numpy as jnp
import numpy as np
import optax

from jax_baselines.A2C.base_class import Actor_Critic_Policy_Gradient_Family
from jax_baselines.math.jax_utils import convert_normalized_obs
from jax_baselines.math.policy_math import (
    kl_divergence_continuous,
    kl_divergence_discrete,
)
from jax_baselines.math.returns import (
    get_gaes,
    normalize_advantage,
    validate_advantage_normalize_scope,
)


class TPPO(Actor_Critic_Policy_Gradient_Family):
    _run_name = "TPPO"

    def __init__(
        self,
        env_builder,
        model_builder_maker,
        lamda=0.95,
        gae_normalize=False,
        gae_normalize_scope="batch",
        minibatch_size=32,
        epoch_num=4,
        kl_range=0.008,
        kl_coef=5,
        value_clip=2.0,
        **kwargs,
    ):
        super().__init__(env_builder, model_builder_maker, **kwargs)

        self.lamda = lamda
        self.gae_normalize = gae_normalize
        self.gae_normalize_scope = validate_advantage_normalize_scope(gae_normalize_scope)
        self.value_clip = value_clip
        self.kl_range = kl_range
        self.kl_coef = kl_coef
        self.minibatch_size = minibatch_size
        self.batch_size = int(
            np.ceil(kwargs.get("batch_size", 256) * self.worker_size / minibatch_size)
            * minibatch_size
            / self.worker_size
        )
        self.epoch_num = epoch_num

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
            kls,
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
            logger_run.log_metric("loss/kl_divergence", kls, steps)

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
        prob, pi_prob = jax.vmap(self.get_logprob, in_axes=(0, 0, None, None))(
            jax.vmap(self.actor, in_axes=(None, None, 0))(actor_params, key, obses),
            actions,
            key,
            True,
        )
        adv = jax.vmap(get_gaes, in_axes=(0, 0, 0, 0, 0, None, None))(
            rewards, terminateds, truncateds, value, next_value, self.gamma, self.lamda
        )
        obses = {key: jnp.vstack(value) for key, value in obses.items()}
        actions = jnp.vstack(actions)
        value = jnp.vstack(value)
        if self.action_type == "continuous":
            mu, log_std = prob
            prob = (mu, jnp.broadcast_to(log_std, mu.shape))
        prob = jax.tree.map(jnp.vstack, prob)
        pi_prob = jnp.vstack(pi_prob)
        adv = jnp.vstack(adv)
        targets = value + adv
        if self.gae_normalize and self.gae_normalize_scope == "batch":
            adv = normalize_advantage(adv)
        return obses, actions, value, targets, prob, pi_prob, adv

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
        obses, actions, old_value, targets, old_prob, old_act_prob, adv = self._preprocess(
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
                kls,
            ) = vals
            use_key, key = jax.random.split(key)
            batch_idxes = jax.random.permutation(use_key, jnp.arange(targets.shape[0])).reshape(
                -1, self.minibatch_size
            )
            obses_batch = {key: value[batch_idxes] for key, value in obses.items()}
            actions_batch = actions[batch_idxes]
            old_value_batch = old_value[batch_idxes]
            targets_batch = targets[batch_idxes]
            old_prob_batch = jax.tree.map(lambda p: p[batch_idxes], old_prob)
            old_act_prob_batch = old_act_prob[batch_idxes]
            adv_batch = adv[batch_idxes]

            def f(updates, input):
                actor_params, critic_params, actor_opt_state, critic_opt_state, key = updates
                obs, act, old_value, target, old_prob, old_act_prob, adv = input
                if self.gae_normalize and self.gae_normalize_scope == "minibatch":
                    adv = normalize_advantage(adv)
                use_key, key = jax.random.split(key)
                (_, (a_loss, entropy_loss, kl)), actor_grad = jax.value_and_grad(
                    self._actor_loss, has_aux=True
                )(
                    actor_params,
                    obs,
                    act,
                    old_prob,
                    old_act_prob,
                    adv,
                    use_key,
                )
                c_loss, critic_grad = jax.value_and_grad(self._critic_loss)(
                    critic_params, actor_params, obs, old_value, target, use_key
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
                    kl,
                )

            updates, losses = jax.lax.scan(
                f,
                (actor_params, critic_params, actor_opt_state, critic_opt_state, key),
                (
                    obses_batch,
                    actions_batch,
                    old_value_batch,
                    targets_batch,
                    old_prob_batch,
                    old_act_prob_batch,
                    adv_batch,
                ),
            )
            actor_params, critic_params, actor_opt_state, critic_opt_state, key = updates
            cl, al, el, kl = losses
            critic_loss += jnp.mean(cl)
            actor_loss += jnp.mean(al)
            entropy_loss += jnp.mean(el)
            kls += jnp.mean(kl)
            return (
                actor_params,
                critic_params,
                actor_opt_state,
                critic_opt_state,
                key,
                critic_loss,
                actor_loss,
                entropy_loss,
                kls,
            )

        val = jax.lax.fori_loop(
            0,
            self.epoch_num,
            i_f,
            (
                actor_params,
                critic_params,
                actor_opt_state,
                critic_opt_state,
                key,
                0.0,
                0.0,
                0.0,
                0.0,
            ),
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
            kls,
        ) = val
        return (
            actor_params,
            critic_params,
            actor_opt_state,
            critic_opt_state,
            critic_loss / self.epoch_num,
            actor_loss / self.epoch_num,
            entropy_loss / self.epoch_num,
            kls / self.epoch_num,
            jnp.mean(targets),
        )

    def _critic_loss(self, critic_params, actor_params, obses, old_value, targets, key):
        values = self.critic(critic_params, actor_params, key, obses)
        clipped_values = old_value + jnp.clip(values - old_value, -self.value_clip, self.value_clip)
        return jnp.mean(
            jnp.maximum(jnp.square(values - targets), jnp.square(clipped_values - targets))
        )

    def _actor_loss_discrete(
        self,
        actor_params,
        obses,
        actions,
        old_prob,
        old_act_prob,
        adv,
        key,
    ):
        prob, log_prob = self.get_logprob(
            self.actor(actor_params, key, obses), actions, key, out_prob=True
        )
        # Paper's entropy: H = -sum(p * log(p)) >= 0
        entropy_h = -jnp.sum(prob * jnp.log(jnp.maximum(prob, 1e-8)), axis=-1, keepdims=True)
        if self.use_entropy_adv_shaping:
            # Paper's shaping: psi(H) = min(alpha * H, |A| / kappa) >= 0
            psi_h = jnp.minimum(
                self.ent_coef * entropy_h, jnp.abs(adv) / self.entropy_adv_shaping_kappa
            )
            adv += psi_h
        adv = jax.lax.stop_gradient(adv)

        ratio = jnp.exp(log_prob - old_act_prob)
        kl = jax.vmap(kl_divergence_discrete)(old_prob, prob)
        actor_loss = -jnp.mean(
            adv * ratio
            - self.kl_coef
            * jnp.where(
                (kl >= self.kl_range) & (ratio * adv >= adv),
                kl,
                self.kl_range,
            )
        )
        entropy_loss = -jnp.mean(entropy_h)
        if self.use_entropy_adv_shaping:
            actor_objective = actor_loss
        else:
            actor_objective = actor_loss + self.ent_coef * entropy_loss
        return actor_objective, (actor_loss, entropy_loss, jnp.mean(kl))

    def _actor_loss_continuous(
        self,
        actor_params,
        obses,
        actions,
        old_prob,
        old_act_prob,
        adv,
        key,
    ):
        prob, log_prob = self.get_logprob(
            self.actor(actor_params, key, obses), actions, key, out_prob=True
        )
        mu, log_std = prob
        std = jnp.broadcast_to(jnp.exp(log_std), mu.shape)
        prob_std = (mu, std)
        old_std = jnp.exp(jnp.array(old_prob[1]))
        old_prob_std = (old_prob[0], old_std)
        # Paper's Gaussian entropy: H = sum(log(sigma)) + 0.5*d*(1+log(2*pi))
        dim = mu.shape[-1]
        entropy_h = jnp.sum(log_std, axis=-1, keepdims=True) + 0.5 * dim * (
            1.0 + jnp.log(2.0 * jnp.pi)
        )
        if self.use_entropy_adv_shaping:
            # Differential entropy can be negative; shaping must preserve the advantage sign.
            psi_h = jnp.minimum(
                self.ent_coef * jnp.maximum(entropy_h, 0.0),
                jnp.abs(adv) / self.entropy_adv_shaping_kappa,
            )
            adv += psi_h
        adv = jax.lax.stop_gradient(adv)

        ratio = jnp.exp(log_prob - old_act_prob)
        kl = jax.vmap(kl_divergence_continuous)(old_prob_std, prob_std)
        actor_loss = -jnp.mean(
            adv * ratio
            - self.kl_coef
            * jnp.where(
                (kl >= self.kl_range) & (ratio * adv >= adv),
                kl,
                self.kl_range,
            )
        )
        entropy_loss = -jnp.mean(entropy_h)
        if self.use_entropy_adv_shaping:
            actor_objective = actor_loss
        else:
            actor_objective = actor_loss + self.ent_coef * entropy_loss
        return actor_objective, (actor_loss, entropy_loss, jnp.mean(kl))
