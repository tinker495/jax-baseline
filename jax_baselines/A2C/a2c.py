import jax
import jax.numpy as jnp
import optax

from jax_baselines.A2C.base_class import Actor_Critic_Policy_Gradient_Family
from jax_baselines.math.jax_utils import convert_normalized_obs
from jax_baselines.math.returns import discount_with_terminated


class A2C(Actor_Critic_Policy_Gradient_Family):
    def __init__(self, env_builder, model_builder_maker, **kwargs):

        super().__init__(env_builder, model_builder_maker, **kwargs)
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
            targets,
        ) = self._train_step(
            self.actor_params,
            self.critic_params,
            self.actor_opt_state,
            self.critic_opt_state,
            None,
            **data,
        )

        if logger_run:
            logger_run.log_metric("loss/critic_loss", critic_loss, steps)
            logger_run.log_metric("loss/actor_loss", actor_loss, steps)
            logger_run.log_metric("loss/entropy_loss", entropy_loss, steps)
            logger_run.log_metric("loss/mean_target", targets, steps)

        return critic_loss

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
        obses = convert_normalized_obs(obses)
        nxtobses = convert_normalized_obs(nxtobses)
        value = jax.vmap(self.critic, in_axes=(None, None, 0))(critic_params, key, obses)
        next_value = jax.vmap(self.critic, in_axes=(None, None, 0))(critic_params, key, nxtobses)
        targets = jax.vmap(discount_with_terminated, in_axes=(0, 0, 0, 0, None))(
            rewards, terminateds, truncateds, next_value, self.gamma
        )
        obses = {key: jnp.vstack(value) for key, value in obses.items()}
        actions = jnp.vstack(actions)
        value = jnp.vstack(value)
        targets = jnp.vstack(targets)
        adv = targets - value
        (_total_loss, (critic_loss, actor_loss, entropy_loss)), (actor_grad, critic_grad) = (
            jax.value_and_grad(self._loss, argnums=(0, 1), has_aux=True)(
                actor_params, critic_params, obses, actions, targets, adv, key
            )
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
            jnp.mean(targets),
        )

    def _loss_discrete(self, actor_params, critic_params, obses, actions, targets, adv, key):
        vals = self.critic(critic_params, key, obses)
        critic_loss = jnp.mean(jnp.square(jnp.squeeze(targets - vals)))

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

        actor_loss = -jnp.mean(log_prob * jax.lax.stop_gradient(adv))
        entropy_loss = -jnp.mean(entropy_h)
        if self.use_entropy_adv_shaping:
            total_loss = self.val_coef * critic_loss + actor_loss
        else:
            total_loss = self.val_coef * critic_loss + actor_loss + self.ent_coef * entropy_loss
        return total_loss, (critic_loss, actor_loss, entropy_loss)

    def _loss_continuous(self, actor_params, critic_params, obses, actions, targets, adv, key):
        vals = self.critic(critic_params, key, obses)
        critic_loss = jnp.mean(jnp.square(jnp.squeeze(targets - vals)))

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

        actor_loss = -jnp.mean(log_prob * jax.lax.stop_gradient(adv))
        entropy_loss = -jnp.mean(entropy_h)
        if self.use_entropy_adv_shaping:
            total_loss = self.val_coef * critic_loss + actor_loss
        else:
            total_loss = self.val_coef * critic_loss + actor_loss + self.ent_coef * entropy_loss
        return total_loss, (critic_loss, actor_loss, entropy_loss)
