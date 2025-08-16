import jax
import jax.numpy as jnp
import optax

from jax_baselines.A2C.base_class import Actor_Critic_Policy_Gradient_Family
from jax_baselines.common.utils import convert_jax, discount_with_terminated


class A2C(Actor_Critic_Policy_Gradient_Family):
    def __init__(self, env_builder, model_builder_maker, **kwargs):
        super().__init__(env_builder, model_builder_maker, **kwargs)

        self.name = "A2C"
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
            targets,
        ) = self._train_step(self.params, self.opt_state, None, **data)

        if self.logger_run:
            self.logger_run.log_metric("loss/critic_loss", critic_loss, steps)
            self.logger_run.log_metric("loss/actor_loss", actor_loss, steps)
            self.logger_run.log_metric("loss/entropy_loss", entropy_loss, steps)
            self.logger_run.log_metric("loss/mean_target", targets, steps)

        return critic_loss

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
        obses = [jnp.stack(zo) for zo in zip(*obses)]
        nxtobses = [jnp.stack(zo) for zo in zip(*nxtobses)]
        actions = jnp.stack(actions)
        rewards = jnp.stack(rewards)
        terminateds = jnp.stack(terminateds)
        truncateds = jnp.stack(truncateds)
        obses = convert_jax(obses)
        nxtobses = convert_jax(nxtobses)
        value = jax.vmap(self.critic, in_axes=(None, None, 0))(
            params,
            key,
            jax.vmap(self.preproc, in_axes=(None, None, 0))(params, key, obses),
        )
        next_value = jax.vmap(self.critic, in_axes=(None, None, 0))(
            params,
            key,
            jax.vmap(self.preproc, in_axes=(None, None, 0))(params, key, nxtobses),
        )
        targets = jax.vmap(discount_with_terminated, in_axes=(0, 0, 0, 0, None))(
            rewards, terminateds, truncateds, next_value, self.gamma
        )
        obses = [jnp.vstack(o) for o in obses]
        actions = jnp.vstack(actions)
        value = jnp.vstack(value)
        targets = jnp.vstack(targets)
        adv = targets - value
        (total_loss, (critic_loss, actor_loss, entropy_loss)), grad = jax.value_and_grad(
            self._loss, has_aux=True
        )(params, obses, actions, targets, adv, key)
        updates, opt_state = self.optimizer.update(grad, opt_state, params=params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, critic_loss, actor_loss, entropy_loss, jnp.mean(targets)

    def _loss_discrete(self, params, obses, actions, targets, adv, key):
        feature = self.preproc(params, key, obses)
        vals = self.critic(params, key, feature)
        critic_loss = jnp.mean(jnp.square(jnp.squeeze(targets - vals)))

        prob, log_prob = self.get_logprob(
            self.actor(params, key, feature), actions, key, out_prob=True
        )
        actor_loss = -jnp.mean(log_prob * jax.lax.stop_gradient(adv))
        entropy = prob * jnp.log(prob)
        entropy_loss = jnp.mean(entropy)
        total_loss = self.val_coef * critic_loss + actor_loss + self.ent_coef * entropy_loss
        return total_loss, (critic_loss, actor_loss, entropy_loss)

    def _loss_continuous(self, params, obses, actions, targets, adv, key):
        feature = self.preproc(params, key, obses)
        vals = self.critic(params, key, feature)
        critic_loss = jnp.mean(jnp.square(jnp.squeeze(targets - vals)))

        prob, log_prob = self.get_logprob(
            self.actor(params, key, feature), actions, key, out_prob=True
        )
        actor_loss = -jnp.mean(log_prob * jax.lax.stop_gradient(adv))
        mu, log_std = prob
        entropy_loss = jnp.mean(jnp.square(mu) - log_std)
        total_loss = self.val_coef * critic_loss + actor_loss + self.ent_coef * entropy_loss
        return total_loss, (critic_loss, actor_loss, entropy_loss)

    def _value_loss(self, params, obses, targets, key):
        feature = self.preproc(params, key, obses)
        vals = self.critic(params, key, feature)
        critic_loss = jnp.mean(jnp.square(jnp.squeeze(targets - vals)))
        return critic_loss

    def learn(
        self,
        total_timesteps,
        callback=None,
        log_interval=1000,
        experiment_name="A2C",
        run_name="A2C",
    ):
        super().learn(
            total_timesteps,
            callback,
            log_interval,
            experiment_name,
            run_name,
        )
