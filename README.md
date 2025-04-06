# Quick Project: 利用 AI 快速將想法化為工具 🚀

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) 

## 🎯 專案目標 (Project Goal)

本專案 (`quick_project`) 旨在展示如何利用 AI（特別是大型語言模型 LLM）與其他新興技術，將腦中的各種想法快速開發成實用或有趣的小工具。這不僅是一個技術實踐的集合，更是展現個人**快速學習、擁抱新技術 (與時俱進)** 以及**將創意付諸實現 (有趣的點子)** 的證明。

## ✨ 動機與理念 (Motivation & Philosophy)

在快速變化的科技時代，能夠迅速掌握並應用新興技術（例如 AI、物理模擬、Web App 框架）至關重要。此專案的核心理念是：

* **敏捷開發 (Agile Prototyping):** 透過 LLM 輔助與現成工具，縮短從概念到原型的開發週期。
* **創意探索 (Creative Exploration):** 將有趣的、實驗性的想法轉化為可互動、可視化的工具，探索技術的邊界。
* **持續學習 (Continuous Learning):** 保持對 AI、圖形計算、強化學習等領域發展的關注，並實際應用於專案中。
* **展示能力 (Portfolio Showcase):** 作為個人作品集的一部分，具體呈現技術實力、創造力、解決問題的能力與主動性，特別適合證明自己能跟上技術潮流。

## 🛠️ 工具列表 (Tools Included)

這個 Repository 目前包含以下幾個透過 AI 或其他技術快速打造的工具：

1.  **[🗣️ 語音輸入法 (Voice Input IME)](./voice_input_whisper/)**
    * **簡介：** 基於 `Faster Whisper` 和 `Tkinter` 的語音輸入工具，可透過快捷鍵 (`Ctrl+Alt+Z`) 啟動錄音，辨識結果自動轉繁體中文並複製到剪貼簿。
    * **亮點：** 結合語音辨識 AI 與桌面應用，解決特定輸入情境的需求。

2.  **[🍑 3D 屁股物理模擬 (Slap Simulation)](./slap/)**
    * **簡介：** 一個充滿創意的物理模擬專案，使用高效能圖形計算框架 `Taichi` 實現。它透過質點-彈簧系統和 PBD 式約束模擬了一個具彈性的 3D 雙葉橢球體（形似臀部），並允許使用者透過滑鼠進行即時「拍打」互動。
    * **亮點： 透過 `Taichi` 高效能框架，以創意方式展示了穩定的軟體物理模擬、程序化建模及即時互動視覺效果。**
    * **[觀看 Demo 影片](https://www.youtube.com/watch?v=30DuSpVrrgY)** (註: 此工具 README 僅含影片連結)

3.  **[🖌️ 傅立葉繪圖動畫 Web App (Fourier Animator Gradio)](./fourier_animator_gradio/)**
    * **簡介：** 將 [傅立葉繪圖工具](#4---傅立葉繪圖動畫生成器-fourier-animator) 包裝成 `Gradio` Web 應用，並部署於 Hugging Face Spaces。使用者可上傳圖片，調整參數後線上生成傅立葉重建動畫 (GIF)。
    * **亮點：** 快速將本機工具轉為可公開互動的 Web App，展現部署與應用整合能力。([線上體驗](https://huggingface.co/spaces/ysinray/fourier-draw))

4.  **[📈 傅立葉繪圖動畫生成器 (Fourier Animator)](./fourier_animator/)**
    * **簡介：** 從輸入圖片 (`hi.png`) 提取輪廓，自動合併多個輪廓，並使用傅立葉級數重建輪廓的過程，生成 Epicycle 旋轉動畫。靈感來自多年前的個人專案復刻與改進。
    * **亮點：** 結合影像處理、路徑規劃與傅立葉分析，將數學概念視覺化。

5.  **[🤖 CartPole DQN 強化學習](./carpole/)**
    * **簡介：** 使用 `PyTorch` 實現 Deep Q-Network (DQN) 演算法，訓練 AI 代理在 OpenAI Gym 的 `CartPole-v1` 環境中學習平衡竿子。包含訓練腳本與使用 `Pygame` 視覺化模型決策的播放腳本。
    * **亮點：** 實踐經典強化學習演算法，展示 AI 訓練與評估流程。

6.  **[🧑‍💻 AI Vtuber Demo](./ai_vtuber/)**
    * **簡介：** 一個簡易的 AI Vtuber 原型。結合 `SpeechRecognition` (語音轉文字)、`deep-translator` (翻譯)、`Transformers` (情緒分析)、`pyttsx3` (文字轉語音) 與 `Tkinter` (表情動畫)，讓虛擬角色能模仿輸入語音並根據情緒改變表情。
    * **亮點：** 整合多種 AI 與多媒體技術，快速搭建一個互動式的 AI 應用原型。

每個工具的詳細功能、使用方法和背後想法，請參考其各自目錄下的 `README.md` 文件。

## 💻 主要技術棧 (Core Technologies)

這個專案集合運用了多種技術，展現了廣泛的技術涉獵與整合能力：

* **程式語言:** Python
* **AI / 機器學習:**
    * 大型語言模型 (LLM) 應用概念 (雖然此處直接應用不多，但體現快速開發理念)
    * 語音辨識: Faster Whisper, SpeechRecognition (Google API)
    * 自然語言處理: Transformers (Hugging Face), Emotion Classification
    * 強化學習: PyTorch, OpenAI Gym (DQN)
    * 翻譯: deep-translator
* **圖形 / 視覺化:**
    * 桌面應用: Tkinter
    * 物理模擬 / 高效能計算: Taichi
    * Web App: Gradio
    * 繪圖 / 動畫: Matplotlib, Pygame, Pillow
* **影像處理:** OpenCV

## ⭐ 為何關注此專案？ (Why This Matters for Portfolio)

這個專案集合體現了：

* **與時俱進 (Adaptability & Continuous Learning):** 主動學習並應用 AI (語音、NLP、RL)、圖形計算 (Taichi)、Web App (Gradio) 等前沿或實用技術。
* **有趣點子 (Creativity & Ideation):** 將抽象想法（如物理模擬、傅立葉視覺化、AI Vtuber）轉化為具體、有創意的解決方案或互動工具。
* **執行力 (Execution & Prototyping):** 快速將概念落地，使用合適的工具高效完成原型開發，展現實踐能力。
* **廣度與整合 (Breadth & Integration):** 專案涵蓋桌面應用、Web、AI 模型、物理模擬等多個面向，並能將不同技術整合在一起。

---

希望這份 `README.md` 能有效幫助你展現你的能力！你可以直接將這段 Markdown 內容複製貼上到你的 `quick_project` repository 的根目錄下的 `README.md` 檔案中。記得檢查並確認每個工具的目錄連結 (`./tool_directory/`) 是否正確。
