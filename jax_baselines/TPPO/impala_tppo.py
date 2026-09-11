import jax
import jax.numpy as jnp
import optax

from jax_baselines.IMPALA.base_class import IMPALA_Family
from jax_baselines.math.jax_utils import convert_normalized_obs
from jax_baselines.math.policy_math import (
    kl_divergence_continuous,
    kl_divergence_discrete,
)


class IMPALA_TPPO(IMPALA_Family):
    _run_name = "IMPALA_TPPO"
    _learn_log_interval = 10

    def __init__(
        self,
        workers,
        model_builder_maker,
        runtime,
        kl_range=0.05,
        kl_coef=5,
        mu_ratio=0.0,
        epoch_num=3,
        buffer_size=0,
        gamma=0.995,
        lamda=0.95,
        learning_rate=3e-4,
        update_freq=100,
        batch_size=1024,
        sample_size=1,
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

        self.mu_ratio = mu_ratio
        self.minibatch_size = 256
        self.epoch_num = epoch_num
        self.kl_range = kl_range
        self.kl_coef = kl_coef
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

        self.actor, self.critic, self.actor_params, self.critic_params = self.model_builder(
            next(self.key_seq), print_model=True
        )
        self.actor_opt_state = self.optimizer.init(self.actor_params)
        self.critic_opt_state = self.optimizer.init(self.critic_params)

        self._train_step = jax.jit(self._train_step)
        self.preprocess = jax.jit(self.preprocess)
        self._actor_loss = (
            jax.jit(self._actor_loss_discrete)
            if self.action_type == "discrete"
            else jax.jit(self._actor_loss_continuous)
        )

    def train_step(self, steps):
        data = self.buffer.sample()

        (
            self.actor_params,
            self.critic_params,
            self.actor_opt_state,
            self.critic_opt_state,
            critic_loss,
            actor_loss,
            entropy_loss,
            rho,
            targets,
        ) = self._train_step(
            self.actor_params,
            self.critic_params,
            self.actor_opt_state,
            self.critic_opt_state,
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
                "loss/critic_loss": critic_loss,
                "loss/actor_loss": actor_loss,
                "loss/entropy_loss": entropy_loss,
                "loss/mean_rho": rho,
                "loss/mean_target": targets,
            }
            self.logger_server.log_trainer(
                steps, {key: float(value) for key, value in jax.device_get(log_dict).items()}
            )
        return critic_loss, rho

    def preprocess(
        self,
        actor_params,
        critic_params,
        key,
        obses,
        actions,
        mu_log_prob,
        rewards,
        nxtobses,
        terminateds,
        truncateds,
    ):
        # ((b x h x w x c), (b x n)) x worker -> (worker x b x h x w x c), (worker x b x n)
        obses = jax.tree.map(lambda *values: jnp.stack(values), *obses)
        nxtobses = jax.tree.map(lambda *values: jnp.stack(values), *nxtobses)
        actions = jnp.stack(actions)
        mu_log_prob = jnp.stack(mu_log_prob)
        rewards = jnp.stack(rewards)
        terminateds = jnp.stack(terminateds)
        truncateds = jnp.stack(truncateds)
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
        vs, rho, adv = self._compute_vtrace(
            pi_prob, mu_log_prob, rewards, terminateds, truncateds, value, next_value
        )
        obses = {key: jnp.vstack(value) for key, value in obses.items()}
        actions = jnp.vstack(actions)
        vs = jnp.vstack(vs)
        if self.action_type == "continuous":
            mu, log_std = prob
            prob = (mu, jnp.broadcast_to(log_std, mu.shape))
        prob = jax.tree.map(jnp.vstack, prob)
        pi_prob = jnp.vstack(pi_prob)
        rho = jnp.vstack(rho)
        adv = jnp.vstack(adv)
        if self.mu_ratio != 0.0:
            mu_prob = jnp.vstack(mu_log_prob)
            out_prob = jnp.log(
                self.mu_ratio * jnp.exp(mu_prob) + (1.0 - self.mu_ratio) * jnp.exp(pi_prob)
            )
            return obses, actions, vs, prob, out_prob, rho, adv
        else:
            return obses, actions, vs, prob, pi_prob, rho, adv

    def _train_step(
        self,
        actor_params,
        critic_params,
        actor_opt_state,
        critic_opt_state,
        key,
        obses,
        actions,
        mu_log_prob,
        rewards,
        nxtobses,
        terminateds,
        truncateds,
    ):
        obses, actions, vs, old_prob, old_act_prob, rho, adv = self.preprocess(
            actor_params,
            critic_params,
            key,
            obses,
            actions,
            mu_log_prob,
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
            batch_idxes = jax.random.permutation(use_key, jnp.arange(vs.shape[0])).reshape(
                -1, self.minibatch_size
            )
            obses_batch = {key: value[batch_idxes] for key, value in obses.items()}
            actions_batch = actions[batch_idxes]
            vs_batch = vs[batch_idxes]
            old_prob_batch = jax.tree.map(lambda p: p[batch_idxes], old_prob)
            old_act_prob_batch = old_act_prob[batch_idxes]
            adv_batch = adv[batch_idxes]

            def f(updates, input):
                actor_params, critic_params, actor_opt_state, critic_opt_state, key = updates
                obs, act, vs, old_prob, old_act_prob, adv = input
                use_key, key = jax.random.split(key)
                (_, (actor_loss, entropy_loss)), actor_grad = jax.value_and_grad(
                    self._actor_loss, has_aux=True
                )(actor_params, obs, act, old_prob, old_act_prob, adv, use_key)
                critic_loss, critic_grad = jax.value_and_grad(self._critic_loss)(
                    critic_params, actor_params, obs, vs, use_key
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
                    critic_loss,
                    actor_loss,
                    entropy_loss,
                )

            updates, losses = jax.lax.scan(
                f,
                (actor_params, critic_params, actor_opt_state, critic_opt_state, key),
                (
                    obses_batch,
                    actions_batch,
                    vs_batch,
                    old_prob_batch,
                    old_act_prob_batch,
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
            jnp.mean(rho),
            jnp.mean(vs),
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
            jnp.where(
                (kl >= self.kl_range) & (adv * (ratio - 1.0) > 0.0),
                adv * ratio - self.kl_coef * kl,
                adv * ratio,
            )
        )
        entropy_loss = -jnp.mean(entropy_h)
        if self.use_entropy_adv_shaping:
            actor_objective = actor_loss
        else:
            actor_objective = actor_loss + self.ent_coef * entropy_loss
        return actor_objective, (actor_loss, entropy_loss)

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
        prob_std = (mu, jnp.broadcast_to(jnp.exp(log_std), mu.shape))
        old_prob_std = (old_prob[0], jnp.exp(old_prob[1]))
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
            jnp.where(
                (kl >= self.kl_range) & (adv * (ratio - 1.0) > 0.0),
                adv * ratio - self.kl_coef * kl,
                adv * ratio,
            )
        )
        entropy_loss = -jnp.mean(entropy_h)
        if self.use_entropy_adv_shaping:
            actor_objective = actor_loss
        else:
            actor_objective = actor_loss + self.ent_coef * entropy_loss
        return actor_objective, (actor_loss, entropy_loss)

    def run_name_update(self, run_name):
        if self.mu_ratio != 0.0:
            run_name += f"({self.mu_ratio:.2f})"
        return super().run_name_update(run_name)
