import numpy as np

from source.emulador.emulador import AdmiravelMundoNovo

from source.agent.memory_buffer import MemoryReplay

from source.agent.dqn_agent import DQNAgent

from source.preprocessing.preprocessing import vectorizer 

from source.preprocessing.utils import duck_type


def train_agent(episodes=256,
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
    dqn_agent.build_model()

    for episode in range(1, episodes + 1):
        reforco_acumulado = 0

        jogo = AdmiravelMundoNovo()

        for step in range(batch_size):
            state_text, action_text, dimensao_acao = jogo.read()

            state_ = vectorizer(state_text)
            actions_ = vectorizer(action_text)

            if np.random.random() < eps:
                choice = np.random.randint(0, dimensao_acao)
            else:
                choice = np.argmax(self.Q(state_, actions_))

            next_state, next_actions, prox_dimensao_acao, prox_reforco, done = jogo.emulador(choice)

            next_state = vectorizer(next_state)
            next_actions = vectorizer(next_actions)

            if not done:
                # Q(s, a) = r' +  γ  * Q(s', a')
                target = prox_reforco + gamma * max(self.Q(next_state, next_actions))
            else:
                target = prox_reforco
           
            action_ = duck_type(actions_, choice)

            memory_replay.add_item(state=state,
                                   action=action_,
                                   reinforcement=target)
            
            reforco_acumulado += prox_reforco

            if done:
                break

            jogo.transicao_estado(choice)            

        state_array, action_array, reinforcement_array = memory_replay.to_numpy()

        dqn_agent.fit(state_array, action_array, reinforcement_array, epochs=5, verbose=False)

        print("Episódio {0}: Reforço acumulado de {1}\n".format(episode, reforco_acumulado))
        
        estat_reforcos.update({episodio: reforco_acumulado})
        eps *= epsilon_decay
