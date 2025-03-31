# Voice Input IME (語音輸入法)

## 🎯 目標

這是一個基於 Whisper 語音辨識模型與 Tkinter 圖形介面的語音輸入應用程式。目的是打造一套可以透過快捷鍵快速啟動的中文語音輸入法，適合用於記事本、文字編輯器、網頁輸入等場景。

---

## 🧩 系統設計架構

```text
┌────────────┐        ┌────────────┐        ┌──────────────┐
│ GUI (Tk)   │ <───→  │ Controller │ ───→   │ WhisperModel │
└────────────┘        └────────────┘        └──────────────┘
     ↑                         ↓                      ↑
    Log                     Threading             FasterWhisper
     ↓                         ↓                      ↓
┌────────────┐        ┌────────────┐        ┌────────────────┐
│ TextArea   │        │ AudioStream│        │ pyperclip剪貼簿 │
└────────────┘        └────────────┘        └────────────────┘
```

- GUI 使用 `Tkinter` 建構，包含文字輸出欄與 log 訊息欄
- `keyboard` 套件監聽快捷鍵 `Ctrl+Alt+Z` 觸發語音辨識
- 使用 `threading` 確保錄音與 GUI 不阻塞
- 語音經由 `pyaudio` 錄製後送進 `faster-whisper` 處理
- 最終辨識結果複製到剪貼簿，並顯示於 GUI 中

---

## 🔧 核心功能

- 🎙️ 一鍵啟動語音辨識（含快捷鍵 `Ctrl+Alt+Z`）
- 🔁 錄音期間按一次即可停止錄音並轉文字
- 📋 輸出結果會自動複製到剪貼簿
- 📝 GUI 顯示辨識文字與處理 log
- 🖱 右鍵選單支援複製、貼上、全選

---

## 🚦 執行流程與事件管理

### 🎯 如何使用 Thread 確保不阻塞：

- 錄音與辨識操作皆由 `threading.Thread(...).start()` 呼叫，避免阻塞 GUI 主執行緒。
- 非同步錄音使用 `tk.after()` 以 50ms 間隔持續讀取音訊。

### 🔐 錄音與辨識流程中的全域鎖概念：

- 使用全域變數 `recording: bool` 控制錄音狀態（視為一種邏輯鎖）
- 按下按鈕或快捷鍵只會觸發一次開始/停止流程，避免重複觸發錯誤

---

## 🧪 使用方法

1. 安裝依賴：
```bash
pip install faster-whisper pyaudio keyboard pyperclip opencc-python-reimplemented
```

2. 執行程式：
```bash
python voice_input_v4.py
```

3. 按下按鈕或 `Ctrl+Alt+Z` 開始錄音，再按一次即停止並辨識
4. 文字會自動顯示在上方輸出欄，並已複製到剪貼簿

---

## 🎥 DEMO 示意影片

👉 https://youtu.be/4tnFs9gCxW0

---

## 📁 備註

- 輸出預設為**繁體中文**，透過 `opencc` 模組自動轉換
- 如果要支援英文、日文，可在呼叫 `transcribe` 時變更語言設定

---

© 2025 Hsin-Ray Yang. All rights reserved.

