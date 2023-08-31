import jax
import jax.numpy as jnp
import haiku as hk
import numpy as np
import optax
from itertools import repeat

from jax_baselines.IMPALA.base_class import IMPALA_Family
from jax_baselines.A2C.network import Actor, Critic
from jax_baselines.common.Module import PreProcess
from jax_baselines.common.utils import convert_jax, get_vtrace, print_param
from jax_baselines.common.losses import hubberloss


class IMPALA(IMPALA_Family):
    def __init__(
        self,
        workers,
        manager=None,
        buffer_size=0,
        gamma=0.995,
        lamda=0.95,
        learning_rate=0.0003,
        update_freq=100,
        batch_size=1024,
        sample_size=1,
        val_coef=0.2,
        ent_coef=0.01,
        rho_max=1.0,
        log_interval=1,
        tensorboard_log=None,
        _init_setup_model=True,
        policy_kwargs=None,
        full_tensorboard_log=False,
        seed=None,
        optimizer="adamw",
    ):
        super().__init__(
            workers,
            manager,
            buffer_size,
            gamma,
            lamda,
            learning_rate,
            update_freq,
            batch_size,
            sample_size,
            val_coef,
            ent_coef,
            rho_max,
            log_interval,
            tensorboard_log,
            _init_setup_model,
            policy_kwargs,
            full_tensorboard_log,
            seed,
            optimizer,
        )

        self.name = "IMPALA_AC"

        if _init_setup_model:
            self.setup_model()

    def setup_model(self):
        self.policy_kwargs = {} if self.policy_kwargs is None else self.policy_kwargs
        if "cnn_mode" in self.policy_kwargs.keys():
            cnn_mode = self.policy_kwargs["cnn_mode"]
            del self.policy_kwargs["cnn_mode"]

        def network_builder(observation_space, cnn_mode, action_size, action_type, **kwargs):
            def builder():
                preproc = hk.transform(
                    lambda x: PreProcess(observation_space, cnn_mode=cnn_mode)(x)
                )
                actor = hk.transform(lambda x: Actor(action_size, action_type, **kwargs)(x))
                critic = hk.transform(lambda x: Critic(**kwargs)(x))
                return preproc, actor, critic

            return builder

        self.network_builder = network_builder(
            self.observation_space,
            cnn_mode,
            self.action_size,
            self.action_type,
            **self.policy_kwargs
        )
        self.actor_builder = self.get_actor_builder()

        self.preproc, self.actor, self.critic = self.network_builder()
        pre_param = self.preproc.init(
            next(self.key_seq),
            [np.zeros((1, *o), dtype=np.float32) for o in self.observation_space],
        )
        actor_param = self.actor.init(
            next(self.key_seq),
            self.preproc.apply(
                pre_param,
                None,
                [np.zeros((1, *o), dtype=np.float32) for o in self.observation_space],
            ),
        )
        critic_param = self.critic.init(
            next(self.key_seq),
            self.preproc.apply(
                pre_param,
                None,
                [np.zeros((1, *o), dtype=np.float32) for o in self.observation_space],
            ),
        )
        self.params = hk.data_structures.merge(pre_param, actor_param, critic_param)

        self.opt_state = self.optimizer.init(self.params)

        print("----------------------model----------------------")
        print_param("preprocess", pre_param)
        print_param("actor", actor_param)
        print_param("critic", critic_param)
        print("-------------------------------------------------")

        self._train_step = jax.jit(self._train_step)
        self.preprocess = jax.jit(self.preprocess)
        self._loss = (
            jax.jit(self._loss_discrete)
            if self.action_type == "discrete"
            else jax.jit(self._loss_continuous)
        )

    def get_logprob_discrete(self, prob, action, key, out_prob=False):
        prob = jnp.clip(jax.nn.softmax(prob), 1e-5, 1.0)
        action = action.astype(jnp.int32)
        if out_prob:
            return prob, jnp.log(jnp.take_along_axis(prob, action, axis=1))
        else:
            return jnp.log(jnp.take_along_axis(prob, action, axis=1))

    def get_logprob_continuous(self, prob, action, key, out_prob=False):
        mu, log_std = prob
        std = jnp.exp(log_std)
        if out_prob:
            return prob, -(
                0.5 * jnp.sum(jnp.square((action - mu) / (std + 1e-7)), axis=-1, keepdims=True)
                + jnp.sum(log_std, axis=-1, keepdims=True)
                + 0.5 * jnp.log(2 * np.pi) * jnp.asarray(action.shape[-1], dtype=jnp.float32)
            )
        else:
            return -(
                0.5 * jnp.sum(jnp.square((action - mu) / (std + 1e-7)), axis=-1, keepdims=True)
                + jnp.sum(log_std, axis=-1, keepdims=True)
                + 0.5 * jnp.log(2 * np.pi) * jnp.asarray(action.shape[-1], dtype=jnp.float32)
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
            self.logger_server.log_trainer.remote(steps, log_dict)
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
        dones,
        terminals,
    ):
        # ((b x h x w x c), (b x n)) x x -> (x x b x h x w x c), (x x b x n)
        obses = [jnp.stack(zo) for zo in zip(*obses)]
        nxtobses = [jnp.stack(zo) for zo in zip(*nxtobses)]
        actions = jnp.stack(actions)
        mu_log_prob = jnp.stack(mu_log_prob)
        rewards = jnp.stack(rewards)
        dones = jnp.stack(dones)
        terminals = jnp.stack(terminals)
        obses = jax.vmap(convert_jax)(obses)
        nxtobses = jax.vmap(convert_jax)(nxtobses)
        feature = jax.vmap(self.preproc.apply, in_axes=(None, None, 0))(params, key, obses)
        value = jax.vmap(self.critic.apply, in_axes=(None, None, 0))(params, key, feature)
        next_value = jax.vmap(self.critic.apply, in_axes=(None, None, 0))(
            params,
            key,
            jax.vmap(self.preproc.apply, in_axes=(None, None, 0))(params, key, nxtobses),
        )
        pi_prob = jax.vmap(self.get_logprob, in_axes=(0, 0, None))(
            jax.vmap(self.actor.apply, in_axes=(None, None, 0))(params, key, feature),
            actions,
            key,
        )
        rho_raw = jnp.exp(pi_prob - mu_log_prob)
        rho = jnp.minimum(rho_raw, self.rho_max)
        c_t = self.lamda * jnp.minimum(rho, self.cut_max)
        vs = jax.vmap(get_vtrace, in_axes=(0, 0, 0, 0, 0, 0, 0, None))(
            rewards, rho, c_t, dones, terminals, value, next_value, self.gamma
        )
        vs_t_plus_1 = jax.vmap(
            lambda v, nv, t: jnp.where(
                t == 1, nv, jnp.concatenate([v[1:], jnp.expand_dims(nv[-1], axis=-1)])
            ),
            in_axes=(0, 0, 0),
        )(vs, next_value, terminals)
        adv = rewards + self.gamma * (1.0 - dones) * vs_t_plus_1 - value
        # adv = (adv - jnp.mean(adv,keepdims=True)) / (jnp.std(adv,keepdims=True) + 1e-6)
        adv = rho * adv
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
        dones,
        terminals,
    ):
        obses, actions, vs, rho, adv = self.preprocess(
            params,
            key,
            obses,
            actions,
            mu_log_prob,
            rewards,
            nxtobses,
            dones,
            terminals,
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
        feature = self.preproc.apply(params, key, obses)
        vals = self.critic.apply(params, key, feature)
        critic_loss = jnp.mean(jnp.square(vs - vals))

        logit = self.actor.apply(params, key, feature)
        prob, log_prob = self.get_logprob(logit, actions, key, out_prob=True)
        actor_loss = -jnp.mean(adv * log_prob)
        entropy = prob * jnp.log(prob)
        entropy_loss = jnp.mean(entropy)
        total_loss = self.val_coef * critic_loss + actor_loss + self.ent_coef * entropy_loss
        return total_loss, (critic_loss, actor_loss, entropy_loss)

    def _loss_continuous(self, params, obses, actions, vs, adv, key):
        feature = self.preproc.apply(params, key, obses)
        vals = self.critic.apply(params, key, feature)
        critic_loss = jnp.mean(jnp.square(vs - vals))

        prob = self.actor.apply(params, key, feature)
        prob, log_prob = self.get_logprob(prob, actions, key, out_prob=True)
        actor_loss = -jnp.mean(adv * log_prob)
        mu, log_std = prob
        entropy_loss = jnp.mean(jnp.square(mu) - log_std)
        total_loss = self.val_coef * critic_loss + actor_loss + self.ent_coef * entropy_loss
        return total_loss, (critic_loss, actor_loss, entropy_loss)

    def learn(
        self,
        total_trainstep,
        callback=None,
        log_interval=10,
        tb_log_name="IMPALA_AC",
        reset_num_timesteps=True,
        replay_wrapper=None,
    ):
        super().learn(
            total_trainstep,
            callback,
            log_interval,
            tb_log_name,
            reset_num_timesteps,
            replay_wrapper,
        )
