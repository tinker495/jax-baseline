import jax
import jax.numpy as jnp
import numpy as np

from jax_baselines.A2C.surrogate_base import SurrogatePolicyGradient
from jax_baselines.math.returns import validate_advantage_normalize_scope


class SPO(SurrogatePolicyGradient):
    def __init__(
        self,
        env_builder,
        model_builder_maker,
        lamda=0.95,
        gae_normalize=False,
        gae_normalize_scope="batch",
        minibatch_size=32,
        epoch_num=4,
        value_clip=2.0,
        ppo_eps=0.2,
        **kwargs,
    ):

        self.lamda = lamda
        self.gae_normalize = gae_normalize
        self.gae_normalize_scope = validate_advantage_normalize_scope(gae_normalize_scope)
        self.value_clip = value_clip
        self.ppo_eps = ppo_eps
        self.minibatch_size = minibatch_size
        self._post_init_minibatch_size = minibatch_size
        self.epoch_num = epoch_num

        super().__init__(env_builder, model_builder_maker, **kwargs)

        # Adjust batch_size after worker_size is known
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
        # SPO loss: -E{r_t(θ)·Â(s_t,a_t) - |Â(s_t,a_t)|/(2ε)·[r_t(θ) - 1]²}
        spo_term1 = ratio * adv
        spo_term2 = jnp.abs(adv) / (2 * self.ppo_eps) * jnp.square(ratio - 1)
        actor_loss = jnp.mean(-spo_term1 + spo_term2)
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

        ratio = jnp.exp(log_prob - old_prob)
        # SPO loss: -E{r_t(θ)·Â(s_t,a_t) - |Â(s_t,a_t)|/(2ε)·[r_t(θ) - 1]²}
        spo_term1 = ratio * adv
        spo_term2 = jnp.abs(adv) / (2 * self.ppo_eps) * jnp.square(ratio - 1)
        actor_loss = jnp.mean(-spo_term1 + spo_term2)
        entropy_loss = -jnp.mean(entropy_h)
        if self.use_entropy_adv_shaping:
            total_loss = self.val_coef * critic_loss + actor_loss
        else:
            total_loss = self.val_coef * critic_loss + actor_loss + self.ent_coef * entropy_loss
        return total_loss, (critic_loss, actor_loss, entropy_loss)

    def learn(
        self,
        total_timesteps,
        callback=None,
        log_interval=1000,
        experiment_name="SPO",
        run_name="SPO",
        eval_num=100,
        **kwargs,
    ):
        return super().learn(
            total_timesteps,
            callback,
            log_interval,
            experiment_name,
            run_name,
            eval_num,
            **kwargs,
        )
