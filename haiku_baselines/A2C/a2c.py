import jax
import jax.numpy as jnp
import haiku as hk
import numpy as np
import optax

from haiku_baselines.A2C.base_class import Actor_Critic_Policy_Gradient_Family
from haiku_baselines.A2C.network import Actor, Critic
from haiku_baselines.common.Module import PreProcess
from haiku_baselines.common.utils import (
    convert_jax,
    discount_with_terminal,
    print_param,
)


class A2C(Actor_Critic_Policy_Gradient_Family):
    def __init__(
        self,
        env,
        gamma=0.995,
        learning_rate=3e-4,
        batch_size=32,
        val_coef=0.2,
        ent_coef=0.5,
        log_interval=200,
        tensorboard_log=None,
        _init_setup_model=True,
        policy_kwargs=None,
        full_tensorboard_log=False,
        seed=None,
        optimizer="rmsprop",
    ):
        super(A2C, self).__init__(
            env,
            gamma,
            learning_rate,
            batch_size,
            val_coef,
            ent_coef,
            log_interval,
            tensorboard_log,
            _init_setup_model,
            policy_kwargs,
            full_tensorboard_log,
            seed,
            optimizer,
        )

        self.name = "A2C"
        self.get_memory_setup()

        if _init_setup_model:
            self.setup_model()

    def setup_model(self):
        self.policy_kwargs = {} if self.policy_kwargs is None else self.policy_kwargs
        if "cnn_mode" in self.policy_kwargs.keys():
            cnn_mode = self.policy_kwargs["cnn_mode"]
            del self.policy_kwargs["cnn_mode"]
        self.preproc = hk.transform(
            lambda x: PreProcess(self.observation_space, cnn_mode=cnn_mode)(x)
        )
        self.actor = hk.transform(
            lambda x: Actor(self.action_size, self.action_type, **self.policy_kwargs)(x)
        )
        self.critic = hk.transform(lambda x: Critic(**self.policy_kwargs)(x))
        pre_param = self.preproc.init(
            next(self.key_seq),
            [np.zeros((1, *o), dtype=np.float32) for o in self.observation_space],
        )
        feature = self.preproc.apply(
            pre_param,
            None,
            [np.zeros((1, *o), dtype=np.float32) for o in self.observation_space],
        )
        actor_param = self.actor.init(next(self.key_seq), feature)
        critic_param = self.critic.init(next(self.key_seq), feature)
        self.params = hk.data_structures.merge(pre_param, actor_param, critic_param)

        self.opt_state = self.optimizer.init(self.params)

        print("----------------------model----------------------")
        print_param("preprocess", pre_param)
        print_param("actor", actor_param)
        print_param("critic", critic_param)
        print("-------------------------------------------------")

        self._get_actions = jax.jit(self._get_actions)
        self._train_step = jax.jit(self._train_step)

    def discription(self):
        return "score : {:.3f}, loss : {:.3f} |".format(
            np.mean(self.scoreque), np.mean(self.lossque)
        )

        return np.random.normal(np.array(mu), np.array(std))

    def train_step(self, steps):
        # Sample a batch from the replay buffer
        data = self.buffer.get_buffer()

        self.params, self.opt_state, critic_loss, actor_loss = self._train_step(
            self.params, self.opt_state, None, **data
        )

        if self.summary:
            self.summary.add_scalar("loss/critic_loss", critic_loss, steps)
            self.summary.add_scalar("loss/actor_loss", actor_loss, steps)

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
        dones,
        terminals,
    ):
        obses = [jnp.stack(zo) for zo in zip(*obses)]
        nxtobses = [jnp.stack(zo) for zo in zip(*nxtobses)]
        actions = jnp.stack(actions)
        rewards = jnp.stack(rewards)
        dones = jnp.stack(dones)
        terminals = jnp.stack(terminals)
        obses = jax.vmap(convert_jax)(obses)
        nxtobses = jax.vmap(convert_jax)(nxtobses)
        value = jax.vmap(self.critic.apply, in_axes=(None, None, 0))(
            params,
            key,
            jax.vmap(self.preproc.apply, in_axes=(None, None, 0))(params, key, obses),
        )
        next_value = jax.vmap(self.critic.apply, in_axes=(None, None, 0))(
            params,
            key,
            jax.vmap(self.preproc.apply, in_axes=(None, None, 0))(params, key, nxtobses),
        )
        targets = jax.vmap(discount_with_terminal, in_axes=(0, 0, 0, 0))(
            rewards, dones, terminals, next_value
        )
        obses = [jnp.vstack(o) for o in obses]
        actions = jnp.vstack(actions)
        value = jnp.vstack(value)
        targets = jnp.vstack(targets)
        adv = targets - value
        (total_loss, (critic_loss, actor_loss)), grad = jax.value_and_grad(
            self._loss, has_aux=True
        )(params, obses, actions, targets, adv, key)
        updates, opt_state = self.optimizer.update(grad, opt_state, params=params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, critic_loss, actor_loss

    def _loss_discrete(self, params, obses, actions, targets, adv, key):
        feature = self.preproc.apply(params, key, obses)
        vals = self.critic.apply(params, key, feature)
        critic_loss = jnp.mean(jnp.square(jnp.squeeze(targets - vals)))

        prob, log_prob = self.get_logprob(
            self.actor.apply(params, key, feature), actions, key, out_prob=True
        )
        actor_loss = -jnp.mean(log_prob * jax.lax.stop_gradient(adv))
        entropy = prob * jnp.log(prob)
        entropy_loss = jnp.mean(entropy)
        total_loss = self.val_coef * critic_loss + actor_loss + self.ent_coef * entropy_loss
        return total_loss, (critic_loss, actor_loss)

    def _loss_continuous(self, params, obses, actions, targets, adv, key):
        feature = self.preproc.apply(params, key, obses)
        vals = self.critic.apply(params, key, feature)
        critic_loss = jnp.mean(jnp.square(jnp.squeeze(targets - vals)))

        prob, log_prob = self.get_logprob(
            self.actor.apply(params, key, feature), actions, key, out_prob=True
        )
        actor_loss = -jnp.mean(log_prob * jax.lax.stop_gradient(adv))
        mu, log_std = prob
        entropy_loss = jnp.mean(jnp.square(mu) - log_std)
        total_loss = self.val_coef * critic_loss + actor_loss + self.ent_coef * entropy_loss
        return total_loss, (critic_loss, actor_loss)

    def learn(
        self,
        total_timesteps,
        callback=None,
        log_interval=100,
        tb_log_name="A2C",
        reset_num_timesteps=True,
        replay_wrapper=None,
    ):
        super().learn(
            total_timesteps,
            callback,
            log_interval,
            tb_log_name,
            reset_num_timesteps,
            replay_wrapper,
        )
