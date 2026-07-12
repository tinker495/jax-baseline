import jax
import jax.numpy as jnp

from jax_baselines.IMPALA.surrogate_base import SurrogateIMPALA


class IMPALA_SPO(SurrogateIMPALA):
    _run_name = "IMPALA_SPO"
    _learn_log_interval = 10

    def __init__(
        self,
        workers,
        model_builder_maker,
        runtime,
        ppo_eps=0.2,
        epoch_num=3,
        buffer_size=0,
        gamma=0.995,
        lamda=0.95,
        learning_rate=3e-4,
        update_freq=100,
        batch_size=1024,
        sample_size=1,
        val_coef=0.2,
        ent_coef=0.01,
        use_entropy_adv_shaping=True,
        entropy_adv_shaping_kappa=2.0,
        rho_max=1.0,
        log_interval=1,
        log_dir=None,
        _init_setup_model=True,
        policy_kwargs=None,
        seed=None,
        optimizer_factory=None,
        worker_replay_factory=None,
        checkpoint_store=None,
    ):

        self.minibatch_size = 256
        self.epoch_num = epoch_num
        self.ppo_eps = ppo_eps

        super().__init__(
            workers,
            model_builder_maker,
            runtime=runtime,
            buffer_size=buffer_size,
            gamma=gamma,
            lamda=lamda,
            learning_rate=learning_rate,
            update_freq=update_freq,
            batch_size=batch_size,
            sample_size=sample_size,
            val_coef=val_coef,
            ent_coef=ent_coef,
            use_entropy_adv_shaping=use_entropy_adv_shaping,
            entropy_adv_shaping_kappa=entropy_adv_shaping_kappa,
            rho_max=rho_max,
            log_interval=log_interval,
            log_dir=log_dir,
            _init_setup_model=_init_setup_model,
            policy_kwargs=policy_kwargs,
            seed=seed,
            optimizer_factory=optimizer_factory,
            worker_replay_factory=worker_replay_factory,
            checkpoint_store=checkpoint_store,
        )

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
