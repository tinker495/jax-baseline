from collections import namedtuple

batch = namedtuple(
    "batch_tuple",
    ["obses", "actions", "mu_log_prob", "rewards", "nxtobses", "terminateds", "truncateds"],
)
