import re

from textos import ESTADOS as est
from textos import ACOES as act

from source.preprocessing.utils import (list_to_dictionary,
                                        save_content)


def states_words():
    words = []
    for state_description in est.values():
        for word in re.findall(r"[\w']+|[-.,!?;]", state_description):
            if word not in words:
                words.append(word)
    return words


def actions_words():
    words = []
    for state_description in act.values():
        for list_words in state_description:
            for word in re.findall(r"[\w']+|[-.,!?;]", list_words):
                if word not in words:
                    words.append(word)
    return words


def build_structure():
    states = states_words()
    actions = actions_words()
    final_words = list(set().union(states, actions))
    final_words += " "
    return final_words


def build_vocabulary(path):
    final_words = build_structure()
    write_vocabulary(path=path,
                     vocabulary=final_words)


def build_lookup_table(path):
    final_words = build_structure()
    lookup_table = list_to_dictionary(final_words)
    save_content(path=path,
                     vocabulary=lookup_table)
