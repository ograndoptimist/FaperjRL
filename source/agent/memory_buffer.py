from collections import deque
from random import shuffle


class MemoryReplay(object):
    """
        Implements a refreshing memory.
    """
    def __init__(self,
                 buffer_size=50):
        self.__deque = deque(maxlen=buffer_size)
        self.__max_len = buffer_size

    def __refresh_memory(self):
        assert len(self) > 0
        self.__deque.popleft()

    def __insert_item(self, state, action, reinforcement):
        return self.__deque.append(
            {
                'state': state,
                'action': action,
                'reinforcement': reinforcement
            }
        )

    def add_item(self, state, action, reinforcement):
        if len(self) >= self.__max_len:
            self.__refresh_memory()
        self.__insert_item(state, action, reinforcement)

    @staticmethod
    def __rebuild_memory(list_items):
        new_memory = deque()
        [new_memory.append(item) for item in list_items]
        return new_memory

    def shuffle_items(self):
        shuffled_items = [item for item in self.__deque]
        shuffle(shuffled_items)
        self.__deque = MemoryReplay.__rebuild_memory(shuffled_items)
        del shuffled_items
   
    def to_numpy(self, max_len):
        states = np.zeros((len(self), max_len))
        actions = np.zeros((len(self), max_len))
        reinforcements = np.zeros((len(self), 1))
        for cont, memory in zip(range(max_len),
                                self.__deque):
            states[cont] = memory['state']
            actions[cont] = memory['action']
            reinforcements[cont] = memory['reinforcement']
        return states, actions, reinforcements

    def __len__(self):
        return len(self.__deque)

    def __str__(self):
        if len(self) == 0:
            return "[]"
        string = ""
        for item in self.__deque:
            string += "{0}\n".format(item)
        return string    
