def worker_epsilons(initial_eps, decay, worker_num):
    if worker_num == 1:
        return [initial_eps]
    return [initial_eps ** (1 + decay * idx / (worker_num - 1)) for idx in range(worker_num)]
