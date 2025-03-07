import numpy as np
import gym

# Criar ambiente FrozenLake-v1 (4x4)
env = gym.make("FrozenLake-v1", is_slippery=True)  # is_slippery=True para ambiente mais realista

# Inicializar a tabela Q com zeros
num_states = env.observation_space.n  # Número total de estados
num_actions = env.action_space.n  # Número total de ações
Q_table = np.zeros((num_states, num_actions))

# Hiperparâmetros
learning_rate = 0.1  # Taxa de aprendizado (alpha)
discount_factor = 0.99  # Fator de desconto (gamma)
epsilon = 1.0  # Taxa de exploração inicial
epsilon_decay = 0.995  # Decaimento da exploração
epsilon_min = 0.01  # Valor mínimo de epsilon
num_episodes = 5000  # Número de episódios para treinamento

# Treinamento do agente
for episode in range(num_episodes):
    state = env.reset()[0]  # Reset do ambiente
    done = False

    while not done:
        # Escolher ação com política epsilon-greedy
        if np.random.rand() < epsilon:
            action = env.action_space.sample()  # Exploração (ação aleatória)
        else:
            action = np.argmax(Q_table[state, :])  # Exploração (ação ótima)

        # Executar ação no ambiente
        next_state, reward, done, _, _ = env.step(action)

        # Atualizar tabela Q com a equação do Q-Learning
        Q_table[state, action] = Q_table[state, action] + learning_rate * (
                reward + discount_factor * np.max(Q_table[next_state, :]) - Q_table[state, action]
        )

        # Atualizar estado atual
        state = next_state

    # Reduzir epsilon para diminuir exploração ao longo do tempo
    epsilon = max(epsilon_min, epsilon * epsilon_decay)

print("Treinamento concluído!")

# Testando o agente treinado
num_test_episodes = 10
total_rewards = 0

for _ in range(num_test_episodes):
    state = env.reset()[0]
    done = False

    while not done:
        action = np.argmax(Q_table[state, :])  # Escolhe a melhor ação aprendida
        next_state, reward, done, _, _ = env.step(action)
        total_rewards += reward
        state = next_state

print(f"Recompensa média após teste: {total_rewards / num_test_episodes}")

env.close()
