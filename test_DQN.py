import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import numpy as np


device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)
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


policy_net = DQN(8, 4)
policy_net.load_state_dict(torch.load(
    'DQN5000_weights.pth', weights_only=True))
policy_net.eval()


def select_action(state):
    with torch.no_grad():
        return policy_net(state).max(1).indices.view(1, 1)


env = RecordVideo(gym.make("LunarLander-v3", continuous=False, gravity=-10.0,
                           enable_wind=False, wind_power=15.0, turbulence_power=1.5, render_mode='rgb_array'), "./mp4/", name_prefix="5000_epoches_lunar_lander")
total_points = []

# for k in range(100):
#     state, _ = env.reset()
#     total_point = 0
#     for _ in range(1000):
#         state = torch.tensor(state, device=device,
#                              dtype=torch.float32).unsqueeze(0)
#         action = select_action(state=state)
#         state, reward, done, truncated, _ = env.step(action.item())
#         total_point += reward
#         if done or truncated:
#             break
#     total_points.append(total_point)

# total_points = np.array(total_points)
# print(sum(total_points >= 200))
# print(total_points)


state, _ = env.reset()
total_point = 0
for _ in range(1000):
    state = torch.tensor(state, device=device,
                         dtype=torch.float32).unsqueeze(0)
    action = select_action(state=state)
    state, reward, done, truncated, _ = env.step(action.item())
    total_point += reward
    if done or truncated:
        break


env.close()
