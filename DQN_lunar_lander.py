import gymnasium as gym
import numpy as np
import random
from collections import namedtuple, deque
from itertools import count
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

device = torch.device("cpu")
env = gym.make("LunarLander-v3", continuous=False, gravity=-10.0,
               enable_wind=False, wind_power=15.0, turbulence_power=1.5)


class DQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(n_observations, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions)
        )

    def forward(self, x):
        Q_values = self.linear_relu_stack(x)
        return Q_values


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


# Training parameters
N_EPOCHES = 5000
BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.02
EPS_DECAY = 0.995
EPS = EPS_START
TAU = 0.005
LR = 0.001

# enviroment parameters
n_actions = env.action_space.n

state, _ = env.reset()
n_observations = len(state)

policy_net = DQN(n_observations=n_observations, n_actions=n_actions)
target_net = DQN(n_observations=n_observations, n_actions=n_actions)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.Adam(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(100000)


def select_action(state, epsilon):
    if random.random() < epsilon:
        action = env.action_space.sample()
        return torch.tensor([[action]], dtype=torch.long)

    with torch.no_grad():
        return policy_net(state).max(1).indices.view(1, 1)


def optimize_model():
    if len(memory) < BATCH_SIZE:
        return

    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                       if s is not None])

    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    Q_values = policy_net(
        state_batch).gather(1, action_batch)

    next_Q_values = torch.zeros(BATCH_SIZE, dtype=torch.float32)
    with torch.no_grad():
        next_Q_values[non_final_mask] = target_net(
            non_final_next_states).max(1).values

    target_values = GAMMA * next_Q_values + reward_batch

    criterion = nn.MSELoss()
    loss = criterion(Q_values, target_values.unsqueeze(1))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


pbar = tqdm(range(N_EPOCHES))

for k in pbar:
    state, _ = env.reset()
    state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

    for t in count():
        action = select_action(state=state, epsilon=EPS)
        next_state, reward, terminated, truncated, _ = env.step(action.item())
        reward = torch.tensor([reward], device=device, dtype=torch.float32)
        done = terminated or truncated

        if terminated:
            next_state = None
        else:
            next_state = torch.tensor(
                next_state, dtype=torch.float32).unsqueeze(0)

        memory.push(state, action, next_state, reward)

        state = next_state

        optimize_model()

        target_net_state_dict = target_net.state_dict()
        policy_net_state_dict = policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key] * \
                TAU + target_net_state_dict[key]*(1-TAU)
        target_net.load_state_dict(target_net_state_dict)

        if done:
            break

    EPS = EPS * EPS_DECAY if EPS > EPS_END else EPS

torch.save(policy_net.state_dict(), 'DQN5000_weights.pth')
