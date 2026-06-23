"""Schedules that evolve over time during training (e.g. exploration epsilon)."""


class ConstantSchedule:
    """Value remains constant over time.

    :param value: (float) Constant value of the schedule
    """

    def __init__(self, value):
        self._value = value

    def value(self, step):
        return self._value


class LinearSchedule:
    """Linear interpolation between initial_p and final_p over schedule_timesteps. After this many timesteps pass
    final_p is returned.

    :param schedule_timesteps: (int) Number of timesteps for which to linearly anneal initial_p to final_p
    :param initial_p: (float) initial output value
    :param final_p: (float) final output value
    """

    def __init__(self, schedule_timesteps, final_p, initial_p=1.0):
        self.schedule_timesteps = schedule_timesteps
        self.final_p = final_p
        self.initial_p = initial_p

    def value(self, step):
        fraction = min(float(step) / self.schedule_timesteps, 1.0)
        return self.initial_p + fraction * (self.final_p - self.initial_p)
