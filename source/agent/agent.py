from abc import (ABCMeta,
                 abstractmethod)


class Agent(metaclass=ABCMeta):
    def __init__(self):
        pass

    @abstractmethod
    def q_value(self, state, action):
        pass

    @abstractmethod
    def Q(self, state, actions):
        pass
