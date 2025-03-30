# 🎨 Fourier Draw App

這是一個用於視覺化圖像輪廓的傅立葉重建動畫應用，透過 Gradio 建立簡單的互動式網頁，並已部署至 Hugging Face Spaces。使用者可以上傳任意圖片，從中擷取邊緣輪廓後以傅立葉分量進行還原動畫重建。

📍 線上體驗：[https://huggingface.co/spaces/ysinray/fourier-draw](https://huggingface.co/spaces/ysinray/fourier-draw)

---

## 📦 功能介紹

- 上傳圖片並進行邊緣預處理（Adaptive Threshold + 梯度過濾）
- 自動合併多輪廓為單一路徑
- 使用傅立葉變換進行曲線重建動畫
- 可調參數：
  - `K`：傅立葉分量數量（控制重建精度）
  - `frame_step`：動畫速度
  - `tail_ratio`：軌跡拖尾長度比例
- 顯示處理進度與動畫輸出結果（GIF）

---

## 🚀 執行方式

若欲在本機端運行，請先安裝依賴：

```bash
pip install gradio opencv-python numpy matplotlib scipy
```

然後執行：

```bash
python app.py
```

---

## 💡 部署說明

此應用已部署於 Hugging Face Spaces，無需安裝即可直接於瀏覽器操作：

🔗 https://huggingface.co/spaces/ysinray/fourier-draw

---

## 🧬 參考來源

本專案的演算法邏輯改寫自：  
🔗 [https://github.com/ys-in-ray/quick_project/tree/main/fourier_animator](https://github.com/ys-in-ray/quick_project/tree/main/fourier_animator)

---

## 🤖 註記

本應用部分程式碼由 ChatGPT 協助整合與撰寫。

---

## 📬 聯絡作者

📧 hsinray.y@gmail.com