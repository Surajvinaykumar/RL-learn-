# DQN for CartPole
import gymnasium as gym
import numpy as np
import random
from collections import deque
import tensorflow as tf
from tensorflow.keras import layers
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Environment setup
env = gym.make("CartPole-v1")
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

# Hyperparameters
gamma = 0.95 
epsilon = 1.0 
epsilon_min = 0.01
epsilon_decay = 0.999
learning_rate = 0.005
batch_size = 64
train_start = 1000 
episodes = 1000

# Experience replay buffer
memory = deque(maxlen=2000)
scores = []

# Q-Network
def build_model():
    model = tf.keras.Sequential([
        layers.Dense(64, input_shape=(state_size,), activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(32, activation='relu'),
        layers.Dense(action_size, activation='linear')
    ])
    model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate))
    return model

model = build_model()
target_model = build_model()
target_model.set_weights(model.get_weights())

# Training
for e in range(episodes):
    state, _ = env.reset()
    state = np.reshape(state, [1, state_size])
    total_reward = 0

    for time in range(500):
        # Epsilon-greedy action
        if np.random.rand() <= epsilon:
            action = random.randrange(action_size)
        else:
            q_values = model.predict(state, verbose=0)
            action = np.argmax(q_values[0])

        # Take action
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        next_state = np.reshape(next_state, [1, state_size])

        # Store experience
        memory.append((state, action, reward, next_state, done))
        state = next_state
        total_reward += reward

        if done:
            scores.append(total_reward)
            print(f"Episode {e+1}/{episodes} - Score: {total_reward}, Epsilon: {epsilon:.2f}")
            break

        # Train the model
        if len(memory) > train_start:
            minibatch = random.sample(memory, batch_size)
            states = np.array([m[0][0] for m in minibatch])
            next_states = np.array([m[3][0] for m in minibatch])
            q_targets = model.predict(states, verbose=0)
            q_next = target_model.predict(next_states, verbose=0)

            for i, (_, action, reward, _, done) in enumerate(minibatch):
                if done:
                    q_targets[i][action] = reward
                else:
                    q_targets[i][action] = reward + gamma * np.amax(q_next[i])

            model.fit(states, q_targets, epochs=1, verbose=0)

    # Update target model every 10 episodes
    if e % 10 == 0:
        target_model.set_weights(model.get_weights())

    # Decay epsilon
    if epsilon > epsilon_min:
        epsilon *= epsilon_decay

env.close()

# 📈 Visualizing the training performance
sns.set(style="darkgrid")
df = pd.DataFrame(scores, columns=["score"])
df["episode"] = df.index + 1
df["avg_score"] = df["score"].rolling(window=50).mean()

plt.figure(figsize=(14, 6))
sns.lineplot(data=df, x="episode", y="score", label="Score per Episode")
sns.lineplot(data=df, x="episode", y="avg_score", label="50-Episode Moving Average")
plt.xlabel("Episode")
plt.ylabel("Score (Steps Survived)")
plt.title("CartPole Training Progress with Target DQN")
plt.legend()
plt.show()

# 🎮 Visualizing the Agent after training
env = gym.make("CartPole-v1", render_mode="human")
state, _ = env.reset()
state = np.reshape(state, [1, state_size])

for _ in range(500):
    q_values = model.predict(state, verbose=0)
    action = np.argmax(q_values[0])
    next_state, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated
    state = np.reshape(next_state, [1, state_size])
    if done:
        break

env.close()
