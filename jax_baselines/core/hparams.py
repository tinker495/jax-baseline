def get_hyper_params(agent):
    params = {
        attr: value
        for attr in dir(agent)
        if not attr.startswith("_")
        for value in [getattr(agent, attr)]
        if not callable(value) and isinstance(value, (int, float, str, bool))
    }

    return params


def add_hparams(agent, tensorboardrun):
    hparam_dict = get_hyper_params(agent)
    tensorboardrun.log_param(hparam_dict)
