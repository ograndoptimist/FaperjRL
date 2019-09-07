import abc

class Agent(abc.ABC):
    @abc.abstractmethod
    def q_value(self, state, action):
        """
          Obtains the q-value corresponding to the action specified.
        """

    @abc.abstractmethod
    def Q(self, state, actions):
        """
          Obtains all q-values to the all actions on the specified state.
        """
