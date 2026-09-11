import jax
import jax.numpy as jnp
import optax

from jax_baselines.IMPALA.base_class import IMPALA_Family
from jax_baselines.math.jax_utils import convert_normalized_obs


class IMPALA(IMPALA_Family):
    _run_name = "IMPALA_AC"
    _learn_log_interval = 10

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
        self._loss = (
            jax.jit(self._loss_discrete)
            if self.action_type == "discrete"
            else jax.jit(self._loss_continuous)
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
        # ((b x h x w x c), (b x n)) x x -> (x x b x h x w x c), (x x b x n)
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
        pi_prob = jax.vmap(self.get_logprob, in_axes=(0, 0, None))(
            jax.vmap(self.actor, in_axes=(None, None, 0))(actor_params, key, obses),
            actions,
            key,
        )
        vs, rho, adv = self._compute_vtrace(
            pi_prob, mu_log_prob, rewards, terminateds, truncateds, value, next_value
        )
        obses = {key: jnp.vstack(value) for key, value in obses.items()}
        actions = jnp.vstack(actions)
        vs = jnp.vstack(vs)
        rho = jnp.vstack(rho)
        adv = jnp.vstack(adv)
        return obses, actions, vs, rho, adv

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
        obses, actions, vs, rho, adv = self.preprocess(
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
        (
            (
                _total_loss,
                (critic_loss, actor_loss, entropy_loss),
            ),
            (actor_grad, critic_grad),
        ) = jax.value_and_grad(self._loss, argnums=(0, 1), has_aux=True)(
            actor_params, critic_params, obses, actions, vs, adv, key
        )
        actor_updates, actor_opt_state = self.optimizer.update(
            actor_grad, actor_opt_state, params=actor_params
        )
        critic_updates, critic_opt_state = self.optimizer.update(
            critic_grad, critic_opt_state, params=critic_params
        )
        actor_params = optax.apply_updates(actor_params, actor_updates)
        critic_params = optax.apply_updates(critic_params, critic_updates)
        return (
            actor_params,
            critic_params,
            actor_opt_state,
            critic_opt_state,
            critic_loss,
            actor_loss,
            entropy_loss,
            jnp.mean(rho),
            jnp.mean(vs),
        )

    def _loss_discrete(self, actor_params, critic_params, obses, actions, vs, adv, key):
        vals = self.critic(critic_params, actor_params, key, obses)
        critic_loss = jnp.mean(jnp.square(vs - vals))

        logit = self.actor(actor_params, key, obses)
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

    def _loss_continuous(self, actor_params, critic_params, obses, actions, vs, adv, key):
        vals = self.critic(critic_params, actor_params, key, obses)
        critic_loss = jnp.mean(jnp.square(vs - vals))

        prob = self.actor(actor_params, key, obses)
        prob, log_prob = self.get_logprob(prob, actions, key, out_prob=True)
        mu, log_std = prob
        # Paper's Gaussian entropy: H = sum(log(sigma)) + 0.5*d*(1+log(2*pi))
        dim = mu.shape[-1]
        entropy_h = jnp.sum(log_std, axis=-1, keepdims=True) + 0.5 * dim * (
            1.0 + jnp.log(2.0 * jnp.pi)
        )
        if self.use_entropy_adv_shaping:
            psi_h = jnp.minimum(
                self.ent_coef * jnp.maximum(entropy_h, 0.0),
                jnp.abs(adv) / self.entropy_adv_shaping_kappa,
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
