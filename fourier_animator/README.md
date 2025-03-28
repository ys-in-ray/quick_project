
# 從圖片輪廓生成傅立葉動畫

📄 HackMD 原始記錄連結：[https://hackmd.io/@hsinray/SkCXfam61g](https://hackmd.io/@hsinray/SkCXfam61g)

---

## 前言

大二的時候學了 Fourier 轉換，剛好看到網路上有一些 [Fourier Drawing 動畫](https://www.youtube.com/watch?v=r6sGWTCMz2k)。

那時候用 Matlab 手刻了一個類似的方法，沒有用函式庫。我記得它花了我一整個晚上（till 早上六七點）。程式碼沒有留著，但是有錄影：

👉 [當年用 Matlab 自幹的 Fourier 動畫](https://www.youtube.com/watch?v=5V3DET2KMZc)

---

## 成果展示

輸入圖 hi.png：  
![hi](https://hackmd.io/_uploads/H1HK1ONayl.png)

這次復刻的動畫成品：  
👉 [觀看動畫（YouTube）](https://www.youtube.com/watch?v=PK8avQFDkCM)

---

## 使用說明

### 📂 圖片準備

將你想要畫成傅立葉動畫的圖像命名為 `hi.png`，放在與主程式相同的資料夾中。  
建議圖片為黑白線稿、邊緣清晰的圖像，例如簽名、塗鴉、簡筆畫。

如果不是黑白線稿，可能會出現非預期結果，例如以下這張川普的照片：

![原始圖片](https://hackmd.io/_uploads/B1e7-_Vpyl.jpg)
![處理結果](https://hackmd.io/_uploads/HJK-WdNakx.png)

---

### 🖥️ 執行方式

請先安裝必要的 Python 套件：

```bash
pip install numpy matplotlib opencv-python scipy
```

接著執行主程式（檔案中預設讀取 `hi.png`）：

```bash
python fourier_animator.py
```

程式會自動開啟一個動畫視窗，展示 Epicycle（旋轉向量）如何逐步重建原始輪廓的過程。

---

### 🛠️ 可調參數

你可以在程式最上方調整以下參數，影響畫面細節與動畫效果：

| 參數             | 說明                                       |
|------------------|--------------------------------------------|
| `kernel_size`     | 模糊程度（數字越大越模糊）                 |
| `blockSize`       | 自適應閾值演算法的區塊大小（必須是奇數）   |
| `C`               | 閾值調整用常數（亮度微調）                 |
| `dilate_iter`     | 邊緣擴張次數（可強化輪廓）                 |
| `min_area`        | 最小輪廓面積（過小的會被過濾掉）           |
| `K`               | 保留的傅立葉頻率個數（越多越精細）         |
| `frame_step`      | 動畫速度（越大越快但不夠平滑）             |
| `tail_ratio`      | 畫筆尾巴長度的比例（0.0 ~ 1.0）            |

---

### 🔍 注意事項

- 圖片中最好只有一個主要輪廓或彼此靠近的物件。
- 合併輪廓的邏輯是自動進行的，可打開程式內的 debug 區塊觀察合併過程。

---

## 程式規劃（流程圖）

> 以下流程圖僅在支援 Mermaid 的平台顯示（如 HackMD、Obsidian 等）

```mermaid
graph TD;
A[載入圖片 hi.png<br>→ 灰階 + 雙邊模糊] --> B[自適應二值化 + 閉運算 + 輪廓擷取];
B --> C[篩選有意義的輪廓<br>根據 gradient 與面積];
C --> D[將輪廓合併成單一路徑<br>(貪婪法 + 合併後去重)];
D --> E[平移中心 → 補點平滑<br>B-spline 插值];
E --> F[轉成複數 z(t)<br>→ 傅立葉轉換];
F --> G[取前 K 個頻率分量];
G --> H[畫出 Epicycle 動畫<br>+ Fading 尾巴];
```

---

## 詳細步驟說明

1. **載入與預處理**  
    - 載入灰階圖 `hi.png`  
    - 雙邊濾波去雜訊保邊緣  
    - 自適應閾值 + 閉運算處理，取得初步輪廓  

2. **篩選輪廓**  
    - 利用 Sobel 取 gradient  
    - 過濾掉過小或邊緣過弱的輪廓  

3. **輪廓合併**  
    - 找出最短距離點連接各輪廓  
    - 動態更新群組、合併過程自動記錄  

4. **平滑與中心化**  
    - 移動到中心  
    - B-spline 補點並轉成複數形式 `z(t)`

5. **傅立葉分解**  
    - 使用 FFT 分解複數曲線  
    - 挑出前 K 個高能量頻率分量  

6. **動畫繪製**  
    - 每個頻率對應一個旋轉向量（Epicycle）  
    - 組合向量畫圖，同時繪出 fading 尾巴  

---

如需進一步展示成果或調整參數邏輯，可參考程式內註解。
