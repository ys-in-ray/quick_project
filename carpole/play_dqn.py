import torch
import torch.nn as nn
import gym
import pygame
import numpy as np
import cv2
import time

# 建立 DQN 架構（與原本訓練一致）
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

# 初始化 Pygame
def init_pygame(width, height):
    pygame.init()
    screen = pygame.display.set_mode((width, height + 60))  # 留一塊顯示按鍵指示
    pygame.display.set_caption("CartPole with DQN Action Visualization")
    return screen

# 將 numpy image 顯示到 pygame 上
def render_frame(screen, frame, action):
    # 畫面轉換成 pygame image
    surf = pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
    screen.blit(surf, (0, 0))

    # 加入指示文字區
    font = pygame.font.SysFont(None, 36)
    white = (255, 255, 255)
    black = (0, 0, 0)
    red   = (255, 0, 0)

    screen.fill(black, (0, frame.shape[0], frame.shape[1], 60))  # 清空下方指示區

    text_left = font.render("[ ← ]", True, white)
    text_right = font.render("[ → ]", True, white)
    screen.blit(text_left, (frame.shape[1] // 4 - 30, frame.shape[0] + 10))
    screen.blit(text_right, (3 * frame.shape[1] // 4 - 30, frame.shape[0] + 10))

    # 畫箭頭指示
    if action == 0:  # left
        pygame.draw.polygon(screen, red, [
            (frame.shape[1] // 4, frame.shape[0] + 50),
            (frame.shape[1] // 4 - 10, frame.shape[0] + 30),
            (frame.shape[1] // 4 + 10, frame.shape[0] + 30),
        ])
    else:  # right
        pygame.draw.polygon(screen, red, [
            (3 * frame.shape[1] // 4, frame.shape[0] + 50),
            (3 * frame.shape[1] // 4 - 10, frame.shape[0] + 30),
            (3 * frame.shape[1] // 4 + 10, frame.shape[0] + 30),
        ])

    pygame.display.flip()

# 主播放函數（帶方向指示）
def play_with_visual(model_path, episodes=3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 建立環境（用 rgb_array 取畫面）
    env = gym.make("CartPole-v1", render_mode="rgb_array")
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    # 載入模型
    model = DQN(state_size, action_size).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    screen = None

    for ep in range(episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0
        print(f"\n--- Playing Episode {ep + 1} ---")

        while not done:
            frame = env.render()
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            with torch.no_grad():
                action = torch.argmax(model(state_tensor)).item()
            state, reward, done, truncated, _ = env.step(action)
            total_reward += reward

            if screen is None:
                h, w, _ = frame.shape
                screen = init_pygame(w, h)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return

            render_frame(screen, frame, action)
            time.sleep(0.03)

        print(f"Total Reward: {total_reward}")

    env.close()
    pygame.quit()

# 🟢 修改這裡指定模型檔名
model_path = "checkpoints/dqn_checkpoint_ep350.pt"
play_with_visual(model_path, episodes=3)
