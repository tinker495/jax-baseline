from copy import deepcopy

import jax
import jax.numpy as jnp
import numpy as np

from jax_baselines.math.losses import QuantileHuberLosses
from jax_baselines.math.policy_math import truncated_mixture
from jax_baselines.SAC.sac import SAC, SACCheckpointParams

TQCCheckpointParams = SACCheckpointParams


class TQC(SAC):
    _run_name = "TQC"
    supports_bulk_training = True

    def __init__(
        self,
        env_builder: callable,
        model_builder_maker,
        ent_coef="auto",
        n_support=25,
        delta=1.0,
        critic_num=2,
        quantile_drop=0.05,
        mixture_type="truncated",
        **kwargs,
    ):

        self.n_support = n_support
        self.delta = delta
        self.critic_num = critic_num
        self.quantile_drop = int(max(np.round(critic_num * n_support * quantile_drop), 1))
        if mixture_type not in ("truncated", "min"):
            raise ValueError(
                f"Invalid mixture_type '{mixture_type}', expected 'truncated' or 'min'"
            )
        self.mixture_type = mixture_type

        super().__init__(env_builder, model_builder_maker, ent_coef=ent_coef, **kwargs)

    def setup_model(self):
        model_builder = self.model_builder_maker(
            self.observation_space,
            self.action_size,
            self.n_support,
            self.policy_kwargs,
        )
        (
            self.preproc,
            self.actor,
            self.critic,
            self.policy_params,
            self.critic_params,
        ) = model_builder(next(self.key_seq), print_model=True)
        self.target_critic_params = deepcopy(self.critic_params)
        self.opt_policy_state = self.optimizer.init(self.policy_params)
        self.opt_critic_state = self.optimizer.init(self.critic_params)

        self._setup_entropy_coef()

        self.quantile = (
            jnp.linspace(0.0, 1.0, self.n_support + 1, dtype=jnp.float32)[1:]
            + jnp.linspace(0.0, 1.0, self.n_support + 1, dtype=jnp.float32)[:-1]
        ) / 2.0  # [support]
        self.quantile = jax.device_put(jnp.expand_dims(self.quantile, axis=(0, 1))).astype(
            jnp.float32
        )  # [1 x 1 x support]

        self._get_actions = jax.jit(self._get_actions)
        self._get_eval_actions = jax.jit(self._get_eval_actions)
        self._train_step = jax.jit(self._train_step)
        self._train_ent_coef = jax.jit(self._train_ent_coef)
        self._bulk_scan = jax.jit(self._bulk_scan)

    def _critic_loss(self, critic_params, policy_params, obses, actions, targets, weights, key):
        feature = self.preproc(policy_params, key, obses)
        qnets = self.critic(critic_params, key, feature, actions)
        logit_valid_tile = jnp.expand_dims(targets, axis=2)  # batch x support x 1
        huber0 = QuantileHuberLosses(
            logit_valid_tile,
            jnp.expand_dims(qnets[0], axis=1),
            self.quantile,
            self.delta,
        )
        critic_loss = jnp.mean(weights * huber0)
        for q in qnets[1:]:
            critic_loss += jnp.mean(
                weights
                * QuantileHuberLosses(
                    logit_valid_tile,
                    jnp.expand_dims(q, axis=1),
                    self.quantile,
                    self.delta,
                )
            )
        return critic_loss, huber0

    def _actor_loss(self, policy_params, critic_params, obses, key, ent_coef):
        feature = self.preproc(policy_params, key, obses)
        policy, log_prob = self._get_pi_log_prob(policy_params, feature, key)
        qnets_pi = self.critic(critic_params, key, feature, policy)
        actor_loss = jnp.mean(
            ent_coef * log_prob - jnp.mean(jnp.concatenate(qnets_pi, axis=1), axis=1)
        )
        return actor_loss, log_prob

    def _target(
        self,
        policy_params,
        target_critic_params,
        rewards,
        nxtobses,
        not_terminateds,
        key,
        ent_coef,
    ):
        next_feature = self.preproc(target_critic_params, key, nxtobses)
        policy, log_prob = self._get_pi_log_prob(
            policy_params, self.preproc(policy_params, key, nxtobses), key
        )
        qnets_pi = self.critic(target_critic_params, key, next_feature, policy)
        if self.mixture_type == "min":
            next_q = jnp.min(jnp.stack(qnets_pi, axis=-1), axis=-1) - ent_coef * log_prob
        else:
            next_q = truncated_mixture(qnets_pi, self.quantile_drop) - ent_coef * log_prob
        return (not_terminateds * next_q * self._gamma) + rewards

    def run_name_update(self, run_name):
        run_name += f"({self.n_support:d})"
        run_name += (
            f"_truncated({self.quantile_drop:d})" if self.mixture_type == "truncated" else "_min"
        )
        return super().run_name_update(run_name)
