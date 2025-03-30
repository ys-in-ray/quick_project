# 🧠 CartPole DQN 強化學習專案

本專案使用 PyTorch 搭配 DQN（Deep Q-Network）訓練代理，在 OpenAI Gym 的經典環境 `CartPole-v1` 中學會保持平衡。訓練完成後可使用 Pygame 視覺化模型的行為策略。

---

## 📁 專案結構

```
.
├── DQN_cartpole.py        # 訓練主程式
├── play_dqn.py            # 模型展示與視覺化
├── checkpoints/           # 訓練中儲存的模型
└── README.md              # 本說明文件
```

---

## 🚀 環境安裝

請先安裝所需套件（建議使用虛擬環境）：

```bash
pip install torch gym pygame matplotlib numpy
```

---

## 🎯 功能說明

### 1️⃣ 訓練模型（`DQN_cartpole.py`）

- 使用經典 DQN 架構：
  - 網路結構：2 hidden layers（24 → 48）
  - Replay Buffer、ε-greedy 探索策略
  - 損失函數：MSELoss
  - Optimizer：Adam
- 每 10 輪更新 target network
- 每 50 輪儲存模型於 `checkpoints/` 目錄
- 最後繪製每輪 reward 收斂曲線

執行訓練：
```bash
python DQN_cartpole.py
```

---

### 2️⃣ 視覺化展示（`play_dqn.py`）

- 使用 `pygame` 播放模型行為，每一步用紅色箭頭指示模型決策（向左或向右）
- 模擬 `CartPole-v1` 環境，使用訓練好的 checkpoint 進行遊戲

執行展示：
```bash
python play_dqn.py
```

> 🔧 你可以在 `play_dqn.py` 中修改模型檔案路徑，例如：
```python
model_path = "checkpoints/dqn_checkpoint_ep350.pt"
```

---

## 📈 模型架構（DQN）

```python
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
```

---

## 💡 延伸方向

- 改為 Double DQN 或 Dueling DQN 架構
- 加入 Prioritized Experience Replay
- 嘗試在其他 Gym 環境應用（如 LunarLander、MountainCar）
- 匯出模型為 ONNX 或部署到 Web

---

## 📬 聯絡作者

📧 hsinray.y@gmail.com

---

## 🤖 註記

本專案部分程式碼由 ChatGPT 協助生成與修改。