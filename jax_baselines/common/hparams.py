def get_hyper_params(agent):
    return dict(
        [
            (attr, getattr(agent, attr))
            for attr in dir(agent)
            if not callable(getattr(agent, attr))
            and not attr.startswith("__")
            and not attr.startswith("_")
            and isinstance(getattr(agent, attr), (int, float, str, bool))
        ]
    )


def add_hparams(agent, tensorboardrun):
    hparam_dict = get_hyper_params(agent)
    tensorboardrun.log_param(hparam_dict)
