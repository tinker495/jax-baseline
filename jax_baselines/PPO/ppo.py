import jax
import jax.numpy as jnp
import numpy as np

from jax_baselines.A2C.surrogate_base import SurrogatePolicyGradient
from jax_baselines.math.returns import validate_advantage_normalize_scope


class PPO(SurrogatePolicyGradient):
    _run_name = "PPO"
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
        # batch_size depends on worker_size which is set in super(); postpone adjustment until after
        self._post_init_minibatch_size = minibatch_size
        self.epoch_num = epoch_num

        super().__init__(env_builder, model_builder_maker, **kwargs)

        # Now that worker_size is known, adjust batch_size
        self.batch_size = int(
            np.ceil(
                kwargs.get("batch_size", 256) * self.worker_size / self._post_init_minibatch_size
            )
            * self._post_init_minibatch_size
            / self.worker_size
        )

        self.get_memory_setup()

    def _loss_discrete(self, params, obses, actions, old_value, targets, old_prob, adv, key):
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

        ratio = jnp.exp(log_prob - old_prob)
        cross_entropy1 = -adv * ratio
        cross_entropy2 = -adv * jnp.clip(ratio, 1.0 - self.ppo_eps, 1.0 + self.ppo_eps)
        actor_loss = jnp.mean(jnp.maximum(cross_entropy1, cross_entropy2))
        entropy_loss = -jnp.mean(entropy_h)
        if self.use_entropy_adv_shaping:
            total_loss = self.val_coef * critic_loss + actor_loss
        else:
            total_loss = self.val_coef * critic_loss + actor_loss + self.ent_coef * entropy_loss
        return total_loss, (critic_loss, actor_loss, entropy_loss)

    def _loss_continuous(self, params, obses, actions, old_value, targets, old_prob, adv, key):
        feature = self.preproc(params, key, obses)
        vals = self.critic(params, key, feature)
        vals_clip = old_value + jnp.clip(vals - old_value, -self.value_clip, self.value_clip)
        vf1 = jnp.square(jnp.squeeze(vals - targets))
        vf2 = jnp.square(jnp.squeeze(vals_clip - targets))
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

        ratio = jnp.exp(log_prob - old_prob)
        cross_entropy1 = -adv * ratio
        cross_entropy2 = -adv * jnp.clip(ratio, 1.0 - self.ppo_eps, 1.0 + self.ppo_eps)
        actor_loss = jnp.mean(jnp.maximum(cross_entropy1, cross_entropy2))
        entropy_loss = -jnp.mean(entropy_h)
        if self.use_entropy_adv_shaping:
            total_loss = self.val_coef * critic_loss + actor_loss
        else:
            total_loss = self.val_coef * critic_loss + actor_loss + self.ent_coef * entropy_loss
        return total_loss, (critic_loss, actor_loss, entropy_loss)
