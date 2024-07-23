import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

from model_builder.flax.apply import get_apply_fn_flax_module
from model_builder.flax.initializers import clip_uniform_initializers
from model_builder.flax.layers import Dense
from model_builder.flax.Module import PreProcess
from model_builder.utils import print_param
SIGMA_INIT = 0.5

class Actor(nn.Module):
    action_size: tuple
    node: int = 256
    hidden_n: int = 2
    layer: nn.Module = Dense

    def setup(self) -> None:
        self.pi_mtx = jax.lax.stop_gradient(
            jnp.expand_dims(jnp.pi * (jnp.arange(0, 128, dtype=np.float32) + 1), axis=(0, 2))
        )  # [ 1 x 128 x 1]

    @nn.compact
    def __call__(self, feature: jnp.ndarray, taus: jnp.ndarray) -> jnp.ndarray:
        """
        Actor network

        Args:
            feature: feature from the observation (batch_size x feature size)
            taus: tau values for the quantile function (batch_size x action size x n_tau)
        """
        feature = nn.Sequential([
            self.layer(self.node),
            jax.nn.relu
        ])(feature)
        init_gru_hidden = self.param(
            "init_gru_hidden", jax.nn.initializers.constant(SIGMA_INIT/jnp.sqrt(self.node)), (self.node,), jnp.float32
        )
        def run_tau_gru(carry, tau):
            last_action, gru_hidden = carry
            last_action = jnp.expand_dims(last_action, axis=1) # [batch x 1]
            last_action_cos = jnp.cos(last_action * self.pi_mtx) # [batch x 128]
            last_action_feature = nn.Sequential([self.layer(self.node), jax.nn.relu])(
                last_action_cos
            )  # [ batch x feature ]
            feature_concat = jnp.concatenate([feature, last_action_feature], axis=1)
            gru_hidden = nn.GRUCell(self.node)(gru_hidden, feature_concat)
            tau = jnp.expand_dims(tau, axis=1) # [batch x 1]
            tau_cos = jnp.cos(tau * self.pi_mtx) # [batch x 128]
            quantile_embedding = nn.Sequential([self.layer(self.node), jax.nn.relu])(tau_cos)
            mul_embedding = feature * quantile_embedding
            action = nn.Sequential(
                [self.layer(self.node) if i % 2 == 0 else jax.nn.relu for i in range(2 * self.hidden_n)]
                + [
                    self.layer(self.action_size[0], kernel_init=clip_uniform_initializers(-0.03, 0.03)),
                    jax.nn.tanh,
                ]
            )(mul_embedding)

            return (action, gru_hidden), action
        taus = jnp.transpose(taus, (1, 0, 2)) # [action size x batch size x n_tau]
        _, action = jax.lax.scan(run_tau_gru, (jnp.zeros((feature.shape[0], self.action_size[0]), dtype=jnp.float32), init_gru_hidden), taus)
        return action

class Critic(nn.Module):
    node: int = 256
    hidden_n: int = 2
    layer: nn.Module = Dense

    @nn.compact
    def __call__(self, feature: jnp.ndarray, actions: jnp.ndarray) -> jnp.ndarray:
        concat = jnp.concatenate([feature, actions], axis=1)
        q_net = nn.Sequential(
            [self.layer(self.node) if i % 2 == 0 else jax.nn.relu for i in range(2 * self.hidden_n)]
            + [self.layer(1, kernel_init=clip_uniform_initializers(-0.03, 0.03))]
        )(concat)
        return q_net

def model_builder_maker(observation_space, action_size, policy_kwargs):
    policy_kwargs = {} if policy_kwargs is None else policy_kwargs
    if "embedding_mode" in policy_kwargs.keys():
        embedding_mode = policy_kwargs["embedding_mode"]
        del policy_kwargs["embedding_mode"]
    else:
        embedding_mode = "normal"

    def model_builder(key=None, print_model=False):
        class Merged_Actor(nn.Module):
            def setup(self):
                self.preproc = PreProcess(observation_space, embedding_mode=embedding_mode)
                self.act = Actor(action_size, **policy_kwargs)

            def __call__(self, x):
                feature = self.preprocess(x)
                action = self.actor(feature)
                return action

            def preprocess(self, x):
                x = self.preproc(x)
                return x

            def actor(self, x):
                return self.act(x)
            
        class Merged_Critics(nn.Module):
            def setup(self):
                self.crit1 = Critic(**policy_kwargs)
                self.crit2 = Critic(**policy_kwargs)

            def __call__(self, x, a):
                q1 = self.crit1(x, a)
                q2 = self.crit2(x, a)
                return q1, q2

        model_actor = Merged_Actor()
        preproc_fn = get_apply_fn_flax_module(model_actor, model_actor.preprocess)
        actor_fn = get_apply_fn_flax_module(model_actor, model_actor.actor)
        model_critic = Merged_Critics()
        critic_fn = get_apply_fn_flax_module(model_critic)
        if key is not None:
            policy_params = model_actor.init(
                key,
                [np.zeros((1, *o), dtype=np.float32) for o in observation_space],
            )
            critic_params = model_critic.init(
                key,
                preproc_fn(policy_params, key, [np.zeros((1, *o), dtype=np.float32) for o in observation_space]),
                np.zeros((1, *action_size), dtype=np.float32),
            )
            if print_model:
                print("------------------build-flax-model--------------------")
                print_param("", policy_params)
                print_param("", critic_params)
                print("------------------------------------------------------")
            return preproc_fn, actor_fn, critic_fn, policy_params, critic_params
        else:
            return preproc_fn, actor_fn, critic_fn

    return model_builder
