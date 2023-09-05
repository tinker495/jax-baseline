from abc import ABC, abstractmethod


class Model_builder(ABC):
    def __init__(self, model_name, env, **kwargs) -> None:
        super().__init__()
        self.model_name = model_name
        self.env = env
        self.kwargs = kwargs

    def build(self):
        return self._build()

    @abstractmethod
    def _build(self):
        pass


class Model(ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def forward(self, x):
        pass

    @abstractmethod
    def get_weights(self):
        pass
