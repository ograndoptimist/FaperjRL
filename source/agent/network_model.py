from keras import Input
from keras.layers import Embedding
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import Dense
from keras.layers import concatenate
from keras.optimizers import RMSprop
from keras.models import Model


def cria_modelo(dimensoes_embedding=32,
                dimensoes_lstm=32,
                dropout_rate=0.5,
                numero_palavras=272):
    """
        Implementa a Rede Neural que dá o Q(s,a) para todas possíveis ações no estado considerado.
    """
    estado_input = Input(batch_shape=(None, None))
    acao_input = Input(batch_shape=(None, None))

    embedding_layer = Embedding(input_dim=numero_palavras,
                                output_dim=dimensoes_embedding)

    embedding_output_estado = embedding_layer(estado_input)
    embedding_output_acao = embedding_layer(acao_input)

    lstm_compartilhada = LSTM(units=dimensoes_lstm)

    lstm_output_estado = lstm_compartilhada(embedding_output_estado)
    lstm_output_acao = lstm_compartilhada(embedding_output_acao)

    fusao = concatenate([lstm_output_estado, lstm_output_acao], axis=-1)

    dense_1 = Dense(units=5, activation='relu')
    dense_1_output = dense_1(fusao)

    dropout = Dropout(rate=dropout_rate)
    dropout_output = dropout(dense_1_output)

    dense_output = Dense(1, activation='relu')
    output = dense_output(dropout_output)

    modelo = Model([estado_input, acao_input], output)

    optimizer = RMSprop(lr=0.0001)

    modelo.compile(optimizer=optimizer, loss='mse', metrics=['acc'])

    return modelo
