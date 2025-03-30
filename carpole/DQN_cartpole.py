import gym
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from collections import deque
import time
import os

# 設定裝置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 建立環境
env = gym.make("CartPole-v1", render_mode="human")
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

# 建立 DQN 模型
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 24)
        self.fc2 = nn.Linear(24, 48)
        self.out = nn.Linear(48, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.out(x)

# 初始化網路與目標網路
policy_net = DQN(state_size, action_size).to(device)
target_net = DQN(state_size, action_size).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

# 優化器與損失函數
optimizer = optim.Adam(policy_net.parameters(), lr=0.001)
criterion = nn.MSELoss()

# Replay Buffer
replay_buffer = deque(maxlen=10000)

# 超參數
gamma = 0.99
epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.995
batch_size = 64
episodes = 500
update_target_every = 10
rewards_per_episode = []

# 儲存資料夾
os.makedirs("checkpoints", exist_ok=True)

# 選擇動作
def choose_action(state):
    if random.random() < epsilon:
        return env.action_space.sample()
    else:
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            q_values = policy_net(state)
        return torch.argmax(q_values).item()

# 訓練一步
def replay():
    if len(replay_buffer) < batch_size:
        return

    batch = random.sample(replay_buffer, batch_size)
    states, actions, rewards, next_states, dones = zip(*batch)

    states = torch.FloatTensor(states).to(device)
    actions = torch.LongTensor(actions).unsqueeze(1).to(device)
    rewards = torch.FloatTensor(rewards).to(device)
    next_states = torch.FloatTensor(next_states).to(device)
    dones = torch.FloatTensor(dones).to(device)

    current_q = policy_net(states).gather(1, actions).squeeze()
    next_q = target_net(next_states).max(1)[0]
    expected_q = rewards + (1 - dones) * gamma * next_q

    loss = criterion(current_q, expected_q.detach())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# 訓練主迴圈
for episode in range(episodes):
    state, _ = env.reset()
    done = False
    total_reward = 0

    while not done:
        action = choose_action(state)
        next_state, reward, done, truncated, _ = env.step(action)
        replay_buffer.append((state, action, reward, next_state, done))
        state = next_state
        total_reward += reward
        replay()

    rewards_per_episode.append(total_reward)

    # 更新 target 網路
    if episode % update_target_every == 0:
        target_net.load_state_dict(policy_net.state_dict())

    # 儲存模型
    if (episode + 1) % 50 == 0:
        path = f"checkpoints/dqn_checkpoint_ep{episode + 1:03d}.pt"
        torch.save(policy_net.state_dict(), path)
        print(f"✅ Saved model to {path}")

    # 更新 epsilon
    epsilon = max(epsilon_min, epsilon * epsilon_decay)

    # 顯示訓練進度
    if (episode + 1) % 10 == 0:
        print(f"Episode {episode+1}, Reward: {total_reward}, Epsilon: {epsilon:.3f}")

# 畫出訓練結果
plt.plot(rewards_per_episode)
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("DQN Training on CartPole-v1")
plt.grid(True)
plt.show()

# 模擬遊玩過程
def play(policy_net, episodes=3):
    for ep in range(episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0
        print(f"\n--- Playing Episode {ep + 1} ---")
        while not done:
            env.render()
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            with torch.no_grad():
                action = torch.argmax(policy_net(state_tensor)).item()
            state, reward, done, truncated, _ = env.step(action)
            total_reward += reward
            time.sleep(0.02)
        print(f"Total Reward: {total_reward}")
    env.close()

# 模擬玩一次
play(policy_net)
