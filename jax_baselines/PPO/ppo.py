import jax
import jax.numpy as jnp

from jax_baselines.A2C.surrogate_base import SurrogatePolicyGradient
from jax_baselines.math.metrics import gaussian_metrics, policy_ratio_metrics
from jax_baselines.math.policy_math import (
    kl_divergence_continuous,
    kl_divergence_discrete,
)


class PPO(SurrogatePolicyGradient):
    _run_name = "PPO"

    def _actor_loss_discrete(self, actor_params, obses, actions, old_policy, adv, key):
        old_prob, old_distribution = old_policy
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

        ratio = jnp.exp(log_prob - old_prob)
        cross_entropy1 = -adv * ratio
        cross_entropy2 = -adv * jnp.clip(ratio, 1.0 - self.ppo_eps, 1.0 + self.ppo_eps)
        actor_loss = jnp.mean(jnp.maximum(cross_entropy1, cross_entropy2))
        entropy_loss = -jnp.mean(entropy_h)
        if self.use_entropy_adv_shaping:
            actor_objective = actor_loss
        else:
            actor_objective = actor_loss + self.ent_coef * entropy_loss
        return actor_objective, {
            "loss/actor_loss": actor_loss,
            "loss/entropy_loss": entropy_loss,
            "loss/kl_divergence": jnp.mean(
                jax.vmap(kl_divergence_discrete, in_axes=(0, 0, None))(old_distribution, prob, 0.0)
            ),
            **policy_ratio_metrics(log_prob, old_prob, self.ppo_eps),
        }

    def _actor_loss_continuous(self, actor_params, obses, actions, old_policy, adv, key):
        old_prob, old_distribution = old_policy
        prob, log_prob = self.get_logprob(
            self.actor(actor_params, key, obses), actions, key, out_prob=True
        )
        mu, log_std = prob
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

        ratio = jnp.exp(log_prob - old_prob)
        cross_entropy1 = -adv * ratio
        cross_entropy2 = -adv * jnp.clip(ratio, 1.0 - self.ppo_eps, 1.0 + self.ppo_eps)
        actor_loss = jnp.mean(jnp.maximum(cross_entropy1, cross_entropy2))
        entropy_loss = -jnp.mean(entropy_h)
        if self.use_entropy_adv_shaping:
            actor_objective = actor_loss
        else:
            actor_objective = actor_loss + self.ent_coef * entropy_loss
        return actor_objective, {
            "loss/actor_loss": actor_loss,
            "loss/entropy_loss": entropy_loss,
            "loss/kl_divergence": jnp.mean(
                kl_divergence_continuous(old_distribution, (mu, jnp.exp(log_std)))
            ),
            **policy_ratio_metrics(log_prob, old_prob, self.ppo_eps),
            **gaussian_metrics(log_std),
        }
