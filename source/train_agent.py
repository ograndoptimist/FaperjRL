import numpy as np
import random
from keras.preprocessing.sequence import pad_sequences

from source.emulador.emulador import AdmiravelMundoNovo


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

    # Dicionário que armazena {episódio:reforco_acumulado}
    estat_reforcos = dict()

    # Tentativa de implementação do replay de experiência.
    memoria_replay = dict()
    vet_estado_acao_target = [[None], [None], [None]]

    numero_iteracoes = 0


    for episodio in range(1, episodios + 1):
        reforco_acumulado = 0

        # O jogo é inicializado no primeiro estado.
        jogo = AdmiravelMundoNovo()

        for passo in range(batch_size):
            # Obtém as descrições de estado e ação do jogo.
            estado_texto, acao_texto, dimensao_acao = jogo.read()

            # Pré-processa os estados e ações, que estão em texto, para vetores de inteiros.
            estado_ = transforma(estado_texto)
            acoes_ = transforma(acao_texto)

            # Seleciona uma ação, escolhida de forma aleatória ou através de max(Q(s, a)).
            if np.random.random() < eps:
                escolha = np.random.randint(0, dimensao_acao)
            else:
                escolha = np.argmax(self.Q(estado_, acoes_))

            # Executa a ação no emulador e observa o reforço e o próximo estado.
            proximo_estado_texto, proximas_acoes_texto, prox_dimensao_acao, prox_reforco, terminado = jogo.emulador(
                escolha)

            # Pré-processa os próximos estados e ações, que estão em texto, para vetores de inteiros.
            proximo_estado = transforma(proximo_estado_texto)
            proximas_acoes = transforma(proximas_acoes_texto)

            if not terminado:
                # Q(s, a) = r' +  γ  * Q(s', a')
                target = prox_reforco + gamma * max(self.Q(proximo_estado, proximas_acoes))
            else:
                target = prox_reforco

            # Para o treinamento ser realizado em batch, é necessário que:
            vet_estado_acao_target[0] = estado_
            vet_estado_acao_target[1] = acoes_ if isinstance(acoes_[0], int) else acoes_[escolha]
            vet_estado_acao_target[2] = [target]

            memoria_replay[numero_iteracoes] = vet_estado_acao_target.copy()

            reforco_acumulado += prox_reforco

            if terminado:
                break

            # Dessa vez, a transição de estado é realizada:
            # estado = novo_estado
            jogo.transicao_estado(escolha)

            numero_iteracoes += 1

        # Torna os dados estatisticamente independentes, já que os dados num problema de RL
        # são fortemente dependentes e não-uniformemente distribuídos.
        amostra = list(memoria_replay.values())
        if episodio in [1, 2, 3, 4, 5, 6]:
            random.shuffle(amostra)
        elif episodio < 50:
            amostra = random.sample(amostra, batch_size // 8)
        else:
            main = random.sample(amostra, batch_size // 4)

        estado = []
        acao = []
        Q_target = []
        for i in range(len(amostra)):
            estado.append(amostra[i][0])
            acao.append(amostra[i][1])
            Q_target.append(amostra[i][2])

        # As sequências precisam ter o mesmo tamanho dentro do mesmo tensor que vai alimentar a rede neural.
        estado = pad_sequences(estado)
        acao = pad_sequences(acao)

        Q_target = np.array(Q_target)

        # Realiza o passo de gradiente de descida de forma a alcançar o mínimo global da função.
        self.modelo.fit([estado, acao], Q_target, epochs=5, verbose=False)

        print("Episódio {0}: Reforço acumulado de {1}".format(episodio, reforco_acumulado))
        print()

        estat_reforcos.update({episodio: reforco_acumulado})
        eps *= epsilon_decay
