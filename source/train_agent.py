import numpy as np
import random
from keras.preprocessing.sequence import pad_sequences

from source.emulador.emulador import AdmiravelMundoNovo

from source.agent.memory_buffer import MemoryReplay

from source.agent.dqn_agent import DQNAgent

from source.preprocessing.preprocessing import vectorizer 


def treino(episodios=256,
           batch_size=64,
           epsilon=1.0,
           epsilon_decay=0.99,
           gamma=0.95):
    """
        Realiza o treinamento do Agente em batch.
        ::batch_size:
                    Números de experiências que serão usados para o treinamento (cada uma é executada apenas uma vez).
        ::epsilon:
                    Probabilidade de escolher aleatoriamente uma ação.
        ::epsilon_decay:
                    Taxa na qual o epsilon decai (a cada episódio, eps *= epsilon_decay).
        ::gamma:
                    Fator de desconto (gamma alto ~~ reforços futuros valem mais que os reforços imediatos).
        ::retorna:
                    Retorna um dicionário onde {episódio:reforco_acumulado}.
    """
    eps = epsilon

    memory_replay = MemoryReplay(buffer_size=50)

    dqn_agent = DQNAgent()

    for episodio in range(1, episodios + 1):
        reforco_acumulado = 0

        jogo = AdmiravelMundoNovo()

        for passo in range(batch_size):
            # Obtém as descrições de estado e ação do jogo.
            state_text, action_text, dimensao_acao = jogo.read()

            estado_ = vectorizer(state_text)
            acoes_ = vectorizer(action_text)

            # Seleciona uma ação, escolhida de forma aleatória ou através de max(Q(s, a)).
            if np.random.random() < eps:
                escolha = np.random.randint(0, dimensao_acao)
            else:
                escolha = np.argmax(self.Q(estado_, acoes_))

            # Executa a ação no emulador e observa o reforço e o próximo estado.
            proximo_estado_texto, proximas_acoes_texto, prox_dimensao_acao, prox_reforco, terminado = jogo.emulador(escolha)

            proximo_estado = vectorizer(proximo_estado_texto)
            proximas_acoes = vectorizer(proximas_acoes_texto)

            if not terminado:
                # Q(s, a) = r' +  γ  * Q(s', a')
                target = prox_reforco + gamma * max(self.Q(proximo_estado, proximas_acoes))
            else:
                target = prox_reforco

            memory_replay.add_item(state=estado,
                                   action=acoes_[escolha],
                                   reinforcement=target)
            
            reforco_acumulado += prox_reforco

            if terminado:
                break

            jogo.transicao_estado(escolha)            

        state_array, action_array, reinforcement_array = memory_replay.to_numpy()

        dqn_agent.fit([state_array, action_array], reinforcement_array, epochs=5, verbose=False)

        print("Episódio {0}: Reforço acumulado de {1}".format(episodio, reforco_acumulado))
        print()

        estat_reforcos.update({episodio: reforco_acumulado})
        eps *= epsilon_decay
