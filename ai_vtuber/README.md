# 🎙️ AI Vtuber Demo

這是一個結合文字、語音、情緒判斷與表情動畫的簡易 AI Vtuber 原型。你可以對著它說話，它會模仿你說的內容、根據情緒變化表情，並同步播放語音與嘴型動畫。

---

## 🧠 功能特色

- 🎤 **中文語音輸入**：錄音後自動轉文字。
- 🌐 **語言翻譯**：將中文翻譯為英文以利情緒分析。
- 😃 **多情緒判斷**：支援 joy / sadness / anger / surprise / neutral。
- 🖼️ **表情圖片對應**：根據情緒切換臉部表情。
- 👄 **嘴巴動畫**：同步 TTS 語音進行嘴型開合。
- 🪟 **簡易介面**：使用 tkinter 顯示圖像與當前情緒。

---

## 🧰 使用技術

| 功能   | 套件 / 模型                                                          |
| ---- | ---------------------------------------------------------------- |
| 語音辨識 | `SpeechRecognition` + Google 語音辨識 API                            |
| 語言翻譯 | `deep-translator`（Google Translate）                              |
| 情緒分析 | `transformers` + `j-hartmann/emotion-english-distilroberta-base` |
| 語音合成 | `pyttsx3`（離線文字轉語音）                                               |
| 圖像顯示 | `Pillow` + `tkinter`                                             |

---

## 🔧 安裝說明

```bash
pip install transformers pyttsx3 pillow SpeechRecognition pyaudio deep-translator
```

> 💡 Windows 使用者如果安裝 `pyaudio` 有問題，可參考 [https://www.lfd.uci.edu/\~gohlke/pythonlibs/#pyaudio](https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyaudio)

---

## 📂 專案結構

```
📁 專案資料夾
├── aivtuber.py               # 主程式
└── face/                    # 表情圖片資料夾
    ├── happy.png
    ├── sad.png
    ├── angry.png
    ├── surprise.png
    ├── neutral.png
    ├── mouth_open.png
    └── mouth_closed.png
```

---

## ▶️ Demo 影片

🎬 [👉 點我觀看 Demo](https://youtube.com/shorts/GX_d2_oBNHU?feature=share)

---

## 📌 備註與後續開發方向

- 加入眨眼 / 點頭動畫 ( 或者更擬真的合成表情方式 )
- 嘴型對音素精準對齊（lip sync）
- 整合 Whisper 模型支援離線語音辨識



---

Made with 💻 by Hsin-Ray Yang

