import jax
import jax.numpy as jnp
import numpy as np
import optax

from jax_baselines.A2C.base_class import Actor_Critic_Policy_Gradient_Family
from jax_baselines.common.utils import (
    convert_jax,
    get_gaes,
    kl_divergence_continuous,
    kl_divergence_discrete,
)


class TPPO(Actor_Critic_Policy_Gradient_Family):
    def __init__(
        self,
        env_builder,
        model_builder_maker,
        lamda=0.95,
        gae_normalize=False,
        minibatch_size=32,
        epoch_num=4,
        kl_range=0.0008,
        kl_coef=20,
        value_clip=0.5,
        **kwargs
    ):
        super().__init__(env_builder, model_builder_maker, **kwargs)

        self.name = "TPPO"
        self.lamda = lamda
        self.gae_normalize = gae_normalize
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

        self.preproc, self.actor, self.critic, self.params = self.model_builder(
            next(self.key_seq), print_model=True
        )
        self.opt_state = self.optimizer.init(self.params)

        self._get_actions = jax.jit(self._get_actions)
        self._preprocess = jax.jit(self._preprocess)
        self._train_step = jax.jit(self._train_step)

    def train_step(self, steps):
        # Sample a batch from the replay buffer
        data = self.buffer.get_buffer()

        (
            self.params,
            self.opt_state,
            critic_loss,
            actor_loss,
            entropy_loss,
            kls,
            targets,
        ) = self._train_step(self.params, self.opt_state, next(self.key_seq), **data)

        if self.logger_run:
            self.logger_run.log_metric("loss/critic_loss", critic_loss, steps)
            self.logger_run.log_metric("loss/actor_loss", actor_loss, steps)
            self.logger_run.log_metric("loss/entropy_loss", entropy_loss, steps)
            self.logger_run.log_metric("loss/mean_target", targets, steps)
            self.logger_run.log_metric("loss/kl_divergense", kls, steps)

        return critic_loss

    def _preprocess(self, params, key, obses, actions, rewards, nxtobses, terminateds, truncateds):
        obses = [jnp.stack(zo) for zo in zip(*obses)]
        nxtobses = [jnp.stack(zo) for zo in zip(*nxtobses)]
        actions = jnp.stack(actions)
        rewards = jnp.stack(rewards)
        terminateds = jnp.stack(terminateds)
        truncateds = jnp.stack(truncateds)
        obses = convert_jax(obses)
        nxtobses = convert_jax(nxtobses)
        feature = jax.vmap(self.preproc, in_axes=(None, None, 0))(params, key, obses)
        value = jax.vmap(self.critic, in_axes=(None, None, 0))(params, key, feature)
        next_value = jax.vmap(self.critic, in_axes=(None, None, 0))(
            params,
            key,
            jax.vmap(self.preproc, in_axes=(None, None, 0))(params, key, nxtobses),
        )
        prob, pi_prob = jax.vmap(self.get_logprob, in_axes=(0, 0, None, None))(
            jax.vmap(self.actor, in_axes=(None, None, 0))(params, key, feature),
            actions,
            key,
            True,
        )
        adv = jax.vmap(get_gaes, in_axes=(0, 0, 0, 0, 0, None, None))(
            rewards, terminateds, truncateds, value, next_value, self.gamma, self.lamda
        )
        obses = [jnp.vstack(o) for o in obses]
        actions = jnp.vstack(actions)
        value = jnp.vstack(value)
        prob = jnp.vstack(prob)
        pi_prob = jnp.vstack(pi_prob)
        adv = jnp.vstack(adv)
        targets = value + adv
        if self.gae_normalize:
            adv = (adv - jnp.mean(adv, keepdims=True)) / (jnp.std(adv, keepdims=True) + 1e-6)
        return obses, actions, value, targets, prob, pi_prob, adv

    def _train_step(
        self,
        params,
        opt_state,
        key,
        obses,
        actions,
        rewards,
        nxtobses,
        terminateds,
        truncateds,
    ):
        obses, actions, old_value, targets, old_prob, old_act_prob, adv = self._preprocess(
            params, key, obses, actions, rewards, nxtobses, terminateds, truncateds
        )

        def i_f(idx, vals):
            params, opt_state, key, critic_loss, actor_loss, entropy_loss, kls = vals
            use_key, key = jax.random.split(key)
            batch_idxes = jax.random.permutation(use_key, jnp.arange(targets.shape[0])).reshape(
                -1, self.minibatch_size
            )
            obses_batch = [o[batch_idxes] for o in obses]
            actions_batch = actions[batch_idxes]
            old_value_batch = old_value[batch_idxes]
            targets_batch = targets[batch_idxes]
            old_prob_batch = old_prob[batch_idxes]
            old_act_prob_batch = old_act_prob[batch_idxes]
            adv_batch = adv[batch_idxes]

            def f(updates, input):
                params, opt_state, key = updates
                obs, act, old_value, target, old_prob, old_act_prob, adv = input
                use_key, key = jax.random.split(key)
                (total_loss, (c_loss, a_loss, entropy_loss, kl),), grad = jax.value_and_grad(
                    self._loss, has_aux=True
                )(params, obs, act, old_value, target, old_prob, old_act_prob, adv, use_key)
                updates, opt_state = self.optimizer.update(grad, opt_state, params=params)
                params = optax.apply_updates(params, updates)
                return (params, opt_state, key), (c_loss, a_loss, entropy_loss, kl)

            updates, losses = jax.lax.scan(
                f,
                (params, opt_state, key),
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
            params, opt_state, key = updates
            cl, al, el, kl = losses
            critic_loss += jnp.mean(cl)
            actor_loss += jnp.mean(al)
            entropy_loss += jnp.mean(el)
            kls += jnp.mean(kl)
            return params, opt_state, key, critic_loss, actor_loss, entropy_loss, kls

        val = jax.lax.fori_loop(
            0, self.epoch_num, i_f, (params, opt_state, key, 0.0, 0.0, 0.0, 0.0)
        )
        params, opt_state, key, critic_loss, actor_loss, entropy_loss, kls = val
        return (
            params,
            opt_state,
            critic_loss / self.epoch_num,
            actor_loss / self.epoch_num,
            entropy_loss / self.epoch_num,
            kls / self.epoch_num,
            jnp.mean(targets),
        )

    def _loss_discrete(
        self, params, obses, actions, old_value, targets, old_prob, old_act_prob, adv, key
    ):
        feature = self.preproc(params, key, obses)
        vals = self.critic(params, key, feature)
        vals_clip = old_value + jnp.clip(vals - old_value, -self.value_clip, self.value_clip)
        vf1 = jnp.square(vals - targets)
        vf2 = jnp.square(vals_clip - targets)
        critic_loss = jnp.mean(jnp.maximum(vf1, vf2))

        prob, log_prob = self.get_logprob(
            self.actor(params, key, feature), actions, key, out_prob=True
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
                (kl >= self.kl_range) & (ratio > 1.0),
                kl,
                self.kl_range,
            )
        )
        entropy_loss = -jnp.mean(entropy_h)
        if self.use_entropy_adv_shaping:
            total_loss = self.val_coef * critic_loss + actor_loss
        else:
            total_loss = self.val_coef * critic_loss + actor_loss + self.ent_coef * entropy_loss
        return total_loss, (critic_loss, actor_loss, entropy_loss, jnp.mean(kl))

    def _loss_continuous(
        self, params, obses, actions, old_value, targets, old_prob, old_act_prob, adv, key
    ):
        feature = self.preproc(params, key, obses)
        vals = self.critic(params, key, feature)
        vals_clip = old_value + jnp.clip(vals - old_value, -self.value_clip, self.value_clip)
        vf1 = jnp.square(vals - targets)
        vf2 = jnp.square(vals_clip - targets)
        critic_loss = jnp.mean(jnp.maximum(vf1, vf2))

        prob, log_prob = self.get_logprob(
            self.actor(params, key, feature), actions, key, out_prob=True
        )
        mu, log_std = prob
        # Paper's Gaussian entropy: H = sum(log(sigma)) + 0.5*d*(1+log(2*pi))
        dim = mu.shape[-1]
        entropy_h = jnp.sum(log_std, axis=-1, keepdims=True) + 0.5 * dim * (
            1.0 + jnp.log(2.0 * jnp.pi)
        )
        if self.use_entropy_adv_shaping:
            # Paper's shaping: psi(H) = min(alpha * H, |A| / kappa) >= 0
            psi_h = jnp.minimum(
                self.ent_coef * entropy_h, jnp.abs(adv) / self.entropy_adv_shaping_kappa
            )
            adv += psi_h
        adv = jax.lax.stop_gradient(adv)

        ratio = jnp.exp(log_prob - old_act_prob)
        kl = jax.vmap(kl_divergence_continuous)(old_prob, prob)
        actor_loss = -jnp.mean(
            adv * ratio
            - self.kl_coef
            * jnp.where(
                (kl >= self.kl_range) & (ratio > 1.0),
                kl,
                self.kl_range,
            )
        )
        entropy_loss = -jnp.mean(entropy_h)
        if self.use_entropy_adv_shaping:
            total_loss = self.val_coef * critic_loss + actor_loss
        else:
            total_loss = self.val_coef * critic_loss + actor_loss + self.ent_coef * entropy_loss
        return total_loss, (critic_loss, actor_loss, entropy_loss, jnp.mean(kl))

    def learn(
        self,
        total_timesteps,
        callback=None,
        log_interval=1000,
        run_name="TPPO",
        experiment_name="TPPO",
    ):
        super().learn(
            total_timesteps,
            callback,
            log_interval,
            experiment_name,
            run_name,
        )
