import numpy as np

from source.agent.memory_buffer import MemoryReplay


def test_memory_buffer():
    memory_buffer = MemoryReplay(buffer_size=5)

    for k in range(5):
        memory_buffer.add_item(state='state_' + str(k),
                               action='action_' + str(k),
                               reinforcement='reinforcement_' + str(k))

    return memory_buffer


if __name__ == '__main__':
    memory_replay = test_memory_buffer()
    print(memory_replay)
    print(len(memory_replay))

    memory_replay.shuffle_items()
    print(memory_replay)
