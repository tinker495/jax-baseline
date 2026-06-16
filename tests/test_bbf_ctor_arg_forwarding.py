import pytest

from jax_baselines.BBF.bbf import BBF
from jax_baselines.BBF.hl_gauss_bbf import HL_GAUSS_BBF
from jax_baselines.DQN.base_class import Q_Network_Family
from jax_baselines.SPR.hl_gauss_spr import HL_GAUSS_SPR
from jax_baselines.SPR.spr import SPR


@pytest.mark.parametrize(
    "cls, extra",
    [
        (BBF, {}),
        (HL_GAUSS_BBF, {}),
        (HL_GAUSS_SPR, {"scaled_by_reset": True}),
    ],
)
def test_spr_family_forwards_clobbered_ctor_args_to_spr_owner(cls, extra, monkeypatch):
    """Regression for the SPR-family constructor clobber.

    BBF/HL_GAUSS_BBF/HL_GAUSS_SPR set off_policy_fix/spr_weight/categorial_*
    (and scaled_by_reset where exposed) on self before super().__init__, but
    omitted them from the forwarded kwargs, so SPR.__init__ reset them to SPR
    defaults (last-write-wins). SPR is now the single owner and non-default
    values must survive construction.
    """
    monkeypatch.setattr(Q_Network_Family, "get_env_setup", lambda self: None)
    monkeypatch.setattr(SPR, "get_memory_setup", lambda self: None)

    agent = cls(
        env_builder=lambda *args, **kwargs: None,
        model_builder_maker=lambda *args, **kwargs: None,
        optimizer_factory=lambda learning_rate: None,
        off_policy_fix=True,
        spr_weight=2.5,
        categorial_bar_n=41,
        categorial_max=123,
        categorial_min=-77,
        _init_setup_model=False,
        **extra,
    )

    assert agent.off_policy_fix is True
    assert agent.spr_weight == 2.5
    assert agent.categorial_bar_n == 41
    assert agent.categorial_max == 123.0
    assert agent.categorial_min == -77.0
    # categorial bounds are coerced to float by SPR, the single owner.
    assert isinstance(agent.categorial_max, float)
    assert isinstance(agent.categorial_min, float)
    for key, value in extra.items():
        assert getattr(agent, key) == value
