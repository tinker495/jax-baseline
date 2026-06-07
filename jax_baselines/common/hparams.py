def get_hyper_params(agent):
    params = {
        attr: value
        for attr in dir(agent)
        if not attr.startswith("_")
        for value in [getattr(agent, attr)]
        if not callable(value) and isinstance(value, (int, float, str, bool))
    }

    # Merge held hparam providers: any non-underscore attribute exposing a
    # callable ``hparams()`` returning a dict contributes its config (e.g. the
    # checkpoint scaffold). Kept generic so future deep handles log for free.
    for attr in dir(agent):
        if attr.startswith("_"):
            continue
        value = getattr(agent, attr)
        provider = getattr(value, "hparams", None)
        if callable(provider):
            provided = provider()
            if isinstance(provided, dict):
                params.update(provided)

    return params


def add_hparams(agent, tensorboardrun):
    hparam_dict = get_hyper_params(agent)
    tensorboardrun.log_param(hparam_dict)
