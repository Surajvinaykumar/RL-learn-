# About me
This summer, I’ve decided to dive into Reinforcement Learning. I'm a project-driven learner — I grasp concepts best by building and experimenting rather than just reading books. So, I’ll be starting with a simple problem and documenting everything along the way: the key observations, the underlying math, and a breakdown of what’s actually happening in the code. As I progress and grow more confident, I’ll move on to tackle more complex problems and environments.

### TrainingRL.py
In the TrainingRL.py, i have tried to solve the CartPole problem, where the agent must balance a pole on the cart by moving it left or right, and the feedback (reward) of +1 will be given everytime the agent balances the pole

## Problem Statement

The CartPole problem is a classic control task in reinforcement learning. An agent must learn to balance a pole on a moving cart by taking actions to move the cart left or right. For every time step the pole stays balanced, the agent receives a reward of +1.

The objective is to maximize the cumulative reward per episode, with a maximum score of 500. An agent that achieves an average score ≥ 475 over 100 episodes is considered to have solved the environment.

## Algorithm: Deep Q-Network (DQN)

This implementation uses a Deep Q-Network (DQN) to approximate the optimal action-value function Q(s, a) with a neural network. It includes:

- Epsilon-Greedy Exploration
- Experience Replay
- Target Network
- Neural Network-based Q-function approximation
- Moving Average Performance Plotting
- Live Environment Simulation after Training

## Hyperparameters

| Parameter        | Value        | Description |
|------------------|--------------|-------------|
| `gamma`          | `0.95`       | Discount factor for future rewards |
| `epsilon`        | `1.0 → 0.01` | Exploration rate (decays each episode) |
| `epsilon_decay`  | `0.999`      | Decay factor for epsilon |
| `learning_rate`  | `0.005`      | Learning rate for the optimizer |
| `batch_size`     | `64`         | Number of experiences used for training at each step |
| `train_start`    | `1000`       | Training begins after collecting these many experiences |
| `episodes`       | `1000`       | Total training episodes |
| `replay_buffer`  | `2000`       | Maximum number of stored experiences |

**[A] HyperParameters -** 

1. Gamma-controls how much future rewards matter.
- gamma → 0, the agent only cares about immediate reward
- gamma → is close to 1, it cares more about long-term rewards.

![Screenshot 2025-05-25 at 5.52.49 PM.png](attachment:51fbcb32-ab21-437f-bf0d-6cc4be0b0590:Screenshot_2025-05-25_at_5.52.49_PM.png)

→ intuition with an example:

with action a and get reward r = 1, with having next state possible reward as 5, consider gamma = 0.95, then,
Q(s,a)=1+0.95⋅5=1+4.75= 5.75

if gamma = 0.5, then,

Q(s,a)=1+0.5⋅5=1+2.5= 3.5

So, Higher gamma = More Patient Agent

, Lower gamma = Short- Sighted Agent

1. Epsilon(e) - controls exploration vs exploitation
    
    This tells the agent how often it should explore random actions instead of choosing the “best” one.
    
    epsilon = 1.0 → 100% random — explore everything
    
    epsilon = 0.1 → 10% random — mostly exploit, rarely explore
    
    → Intuition: In the initial phase of training, , we don’t know anything, so we explore (experimenting with random actions). As we learn more, we start exploiting what we’ve learnt.
    
2. epsilon_decay - This reduces epsilon slightly after each episode, so the agent gradually explores less and exploits more.
    
    epsilon becomes very small(but never zero) — because epsilon_min is set to 0.01.
    
    This is Important because:
    
    Early Training : Explore to gather experience
    
    Later training : Exploit what we’ve learnt
    
    The decay helps the agent automatically shift from beginner to expert mode.
