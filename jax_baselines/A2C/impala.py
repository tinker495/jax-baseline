import jax
import jax.numpy as jnp
import optax

from jax_baselines.IMPALA.base_class import IMPALA_Family
from jax_baselines.math.jax_utils import convert_jax


class IMPALA(IMPALA_Family):
    _run_name = "IMPALA_AC"
    _learn_log_interval = 10

    def __init__(
        self,
        workers,
        model_builder_maker,
        runtime,
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

    def setup_model(self):
        self.model_builder = self.model_builder_maker(
            self.observation_space, self.action_size, self.action_type, self.policy_kwargs
        )
        self.actor_builder = self.get_actor_builder()

        self.preproc, self.actor, self.critic, self.params = self.model_builder(
            next(self.key_seq), print_model=True
        )
        self.opt_state = self.optimizer.init(self.params)

        self._train_step = jax.jit(self._train_step)
        self.preprocess = jax.jit(self.preprocess)
        self._loss = (
            jax.jit(self._loss_discrete)
            if self.action_type == "discrete"
            else jax.jit(self._loss_continuous)
        )

    def train_step(self, steps):
        data = self.buffer.sample()

        (
            self.params,
            self.opt_state,
            critic_loss,
            actor_loss,
            entropy_loss,
            rho,
            targets,
        ) = self._train_step(
            self.params,
            self.opt_state,
            next(self.key_seq),
            data[0],
            data[1],
            data[2],
            data[3],
            data[4],
            data[5],
            data[6],
        )

        if steps % self.log_interval == 0:
            log_dict = {
                "loss/critic_loss": float(critic_loss),
                "loss/actor_loss": float(actor_loss),
                "loss/entropy_loss": float(entropy_loss),
                "loss/mean_rho": float(rho),
                "loss/mean_target": float(targets),
            }
            self.logger_server.log_trainer(steps, log_dict)
        return critic_loss, float(rho)

    def preprocess(
        self,
        params,
        key,
        obses,
        actions,
        mu_log_prob,
        rewards,
        nxtobses,
        terminateds,
        truncateds,
    ):
        # ((b x h x w x c), (b x n)) x x -> (x x b x h x w x c), (x x b x n)
        obses = [jnp.stack(zo) for zo in zip(*obses)]
        nxtobses = [jnp.stack(zo) for zo in zip(*nxtobses)]
        actions = jnp.stack(actions)
        mu_log_prob = jnp.stack(mu_log_prob)
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
        pi_prob = jax.vmap(self.get_logprob, in_axes=(0, 0, None))(
            jax.vmap(self.actor, in_axes=(None, None, 0))(params, key, feature),
            actions,
            key,
        )
        vs, rho, adv = self._compute_vtrace(
            pi_prob, mu_log_prob, rewards, terminateds, truncateds, value, next_value
        )
        obses = [jnp.vstack(o) for o in obses]
        actions = jnp.vstack(actions)
        vs = jnp.vstack(vs)
        rho = jnp.vstack(rho)
        adv = jnp.vstack(adv)
        return obses, actions, vs, rho, adv

    def _train_step(
        self,
        params,
        opt_state,
        key,
        obses,
        actions,
        mu_log_prob,
        rewards,
        nxtobses,
        terminateds,
        truncateds,
    ):
        obses, actions, vs, rho, adv = self.preprocess(
            params,
            key,
            obses,
            actions,
            mu_log_prob,
            rewards,
            nxtobses,
            terminateds,
            truncateds,
        )
        (total_loss, (critic_loss, actor_loss, entropy_loss),), grad = jax.value_and_grad(
            self._loss, has_aux=True
        )(params, obses, actions, vs, adv, key)
        updates, opt_state = self.optimizer.update(grad, opt_state, params=params)
        params = optax.apply_updates(params, updates)
        return (
            params,
            opt_state,
            critic_loss,
            actor_loss,
            entropy_loss,
            jnp.mean(rho),
            jnp.mean(vs),
        )

    def _loss_discrete(self, params, obses, actions, vs, adv, key):
        feature = self.preproc(params, key, obses)
        vals = self.critic(params, key, feature)
        critic_loss = jnp.mean(jnp.square(vs - vals))

        logit = self.actor(params, key, feature)
        prob, log_prob = self.get_logprob(logit, actions, key, out_prob=True)
        entropy_h = -jnp.sum(prob * jnp.log(jnp.maximum(prob, 1e-8)), axis=-1, keepdims=True)
        if self.use_entropy_adv_shaping:
            psi_h = jnp.minimum(
                self.ent_coef * entropy_h, jnp.abs(adv) / self.entropy_adv_shaping_kappa
            )
            adv += psi_h
        adv = jax.lax.stop_gradient(adv)
        actor_loss = -jnp.mean(log_prob * adv)

        entropy_loss = -jnp.mean(entropy_h)
        if self.use_entropy_adv_shaping:
            total_loss = self.val_coef * critic_loss + actor_loss
        else:
            total_loss = self.val_coef * critic_loss + actor_loss + self.ent_coef * entropy_loss
        return total_loss, (critic_loss, actor_loss, entropy_loss)

    def _loss_continuous(self, params, obses, actions, vs, adv, key):
        feature = self.preproc(params, key, obses)
        vals = self.critic(params, key, feature)
        critic_loss = jnp.mean(jnp.square(vs - vals))

        prob = self.actor(params, key, feature)
        prob, log_prob = self.get_logprob(prob, actions, key, out_prob=True)
        mu, log_std = prob
        # Paper's Gaussian entropy: H = sum(log(sigma)) + 0.5*d*(1+log(2*pi))
        dim = mu.shape[-1]
        entropy_h = jnp.sum(log_std, axis=-1, keepdims=True) + 0.5 * dim * (
            1.0 + jnp.log(2.0 * jnp.pi)
        )
        if self.use_entropy_adv_shaping:
            psi_h = jnp.minimum(
                self.ent_coef * entropy_h, jnp.abs(adv) / self.entropy_adv_shaping_kappa
            )
            adv += psi_h
        adv = jax.lax.stop_gradient(adv)
        actor_loss = -jnp.mean(log_prob * adv)
        entropy_loss = -jnp.mean(entropy_h)
        if self.use_entropy_adv_shaping:
            total_loss = self.val_coef * critic_loss + actor_loss
        else:
            total_loss = self.val_coef * critic_loss + actor_loss + self.ent_coef * entropy_loss
        return total_loss, (critic_loss, actor_loss, entropy_loss)
