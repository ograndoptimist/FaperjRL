from keras import Input
from keras.layers import (Embedding,
                          LSTM,
                          Dropout,
                          Dense,
                          concatenate)
from keras.optimizers import RMSprop
from keras.models import Model

from source.agent.agent import Agent


class DQNAgent(Agent):
    """
        Implements a Reinforcement Learning Agent based on Deep Q-Learning.
    """
    def __init__(self):
        super(self).__init__()
        self.__model = None

    def build_network(dim_embedding=32,
                      dim_lstm=32,
                      dropout_rate=0.5,
                      dim_corpus=272):
        """
            Implementa a Rede Neural que dá o Q(s,a) para todas possíveis ações no estado considerado.
        """
        assert self.__model == None
    
        state_input = Input(batch_shape=(None, None))
        action_input = Input(batch_shape=(None, None))

        embedding_layer = Embedding(input_dim=dim_corpus,
                                    output_dim=dim_embedding)

        embedding_output_state = embedding_layer(state_input)
        embedding_output_action = embedding_layer(action_input)

        lstm_shared = LSTM(units=dim_lstm)

        lstm_output_state = lstm_shared(embedding_output_state)
        lstm_output_action = lstm_shared(embedding_output_action)

        merged = concatenate([lstm_output_state, lstm_output_action], axis=-1)

        dense_1 = Dense(units=5, activation='relu')
        dense_1_output = dense_1(merged)

        dropout = Dropout(rate=dropout_rate)
        dropout_output = dropout(dense_1_output)

        dense_output = Dense(1, activation='relu')
        output = dense_output(dropout_output)

        model = Model([state_input, action_input], output)

        optimizer = RMSprop(lr=0.0001)

        model.compile(optimizer=optimizer, loss='mse', metrics=['acc'])

        self.__model = model        

    def q_value(self, state, action):
        """
            Retorna o Q(s, a) calculado pela rede neural.
                ::estado:
                        Um vetor de inteiros.
                ::acao:
                        Um vetor de inteiros
                ::retorna:                        
        """
        return self.model.predict([state, action])[0][0]  

    def Q(self, state, actions):
        """
            Calcula o Q-value para todas ações do estado.
            ::estado:
                    Um vetor de inteiros.
            ::acoes:
                    Um vetor de inteiros.            
            ::retorna::                    
        """
        if isinstance(actions[0], int):
            return [self.q_value(np.array(state), np.array(actions))]
        else:
            return [self.q_value(np.array(state), np.array(action)) for action in actions] 
