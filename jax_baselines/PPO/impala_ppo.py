import jax
import jax.numpy as jnp

from jax_baselines.IMPALA.surrogate_base import SurrogateIMPALA


class IMPALA_PPO(SurrogateIMPALA):
    _run_name = "IMPALA_PPO"
    _learn_log_interval = 10

    def _loss_discrete(self, params, obses, actions, vs, mu_prob, pi_prob, adv, key):
        feature = self.preproc(params, key, obses)
        vals = self.critic(params, key, feature)
        critic_loss = jnp.mean(jnp.square(jnp.squeeze(vals - vs)))

        logit = self.actor(params, key, feature)
        prob, log_prob = self.get_logprob(logit, actions, key, out_prob=True)
        # Paper's entropy: H = -sum(p * log(p)) >= 0
        entropy_h = -jnp.sum(prob * jnp.log(jnp.maximum(prob, 1e-8)), axis=-1, keepdims=True)
        if self.use_entropy_adv_shaping:
            # Paper's shaping: psi(H) = min(alpha * H, |A| / kappa) >= 0
            psi_h = jnp.minimum(
                self.ent_coef * entropy_h, jnp.abs(adv) / self.entropy_adv_shaping_kappa
            )
            adv += psi_h
        adv = jax.lax.stop_gradient(adv)

        is_ratio = jnp.clip(jnp.exp(mu_prob - pi_prob), 0.0, 2.0)
        ratio = is_ratio * jnp.exp(log_prob - mu_prob)
        cross_entropy1 = -adv * ratio
        cross_entropy2 = -adv * jnp.clip(ratio, 1.0 - self.ppo_eps, 1.0 + self.ppo_eps)
        actor_loss = jnp.mean(jnp.maximum(cross_entropy1, cross_entropy2))
        entropy_loss = -jnp.mean(entropy_h)
        if self.use_entropy_adv_shaping:
            total_loss = self.val_coef * critic_loss + actor_loss
        else:
            total_loss = self.val_coef * critic_loss + actor_loss + self.ent_coef * entropy_loss
        return total_loss, (critic_loss, actor_loss, entropy_loss)

    def _loss_continuous(self, params, obses, actions, vs, mu_prob, pi_prob, adv, key):
        # pi_prob is accepted for a uniform scan-call signature with _loss_discrete;
        # the continuous IS ratio uses mu_prob only.
        feature = self.preproc(params, key, obses)
        vals = self.critic(params, key, feature)
        critic_loss = jnp.mean(jnp.square(jnp.squeeze(vals - vs)))

        prob = self.actor(params, key, feature)
        prob, log_prob = self.get_logprob(prob, actions, key, out_prob=True)
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

        ratio = jnp.exp(log_prob - mu_prob)
        cross_entropy1 = -adv * ratio
        cross_entropy2 = -adv * jnp.clip(ratio, 1.0 - self.ppo_eps, 1.0 + self.ppo_eps)
        actor_loss = jnp.mean(jnp.maximum(cross_entropy1, cross_entropy2))
        entropy_loss = -jnp.mean(entropy_h)
        if self.use_entropy_adv_shaping:
            total_loss = self.val_coef * critic_loss + actor_loss
        else:
            total_loss = self.val_coef * critic_loss + actor_loss + self.ent_coef * entropy_loss
        return total_loss, (critic_loss, actor_loss, entropy_loss)
