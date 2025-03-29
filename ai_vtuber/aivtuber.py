import tkinter as tk
from PIL import Image, ImageTk
import time
import threading
import pyttsx3
from transformers import pipeline
import os
import speech_recognition as sr
from deep_translator import GoogleTranslator

# 初始化模型 & TTS
emotion_classifier = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    return_all_scores=False
)
engine = pyttsx3.init()

# Tkinter 初始化
root = tk.Tk()
root.title("AI Vtuber")
canvas = tk.Canvas(root, width=512, height=512)
canvas.pack()

# 顯示目前表情的 Label
emotion_label = tk.Label(root, text="目前表情：尚未辨識", font=('Arial', 14))
emotion_label.pack(pady=5)

# 圖片快取
images = {}
IMAGE_DIR = "face"

def load_image(filename):
    path = os.path.join(IMAGE_DIR, filename)
    if path not in images:
        img = Image.open(path).resize((512, 512))
        images[path] = ImageTk.PhotoImage(img)
    return images[path]

# 預設圖層 ID
face_id = None
mouth_id = None

# 嘴巴動畫函式
def speak_and_animate(text):
    global mouth_id
    duration = max(len(text) / 10, 2)
    end_time = time.time() + duration
    while time.time() < end_time:
        canvas.itemconfig(mouth_id, image=load_image("mouth_open.png"))
        root.update()
        time.sleep(0.15)
        canvas.itemconfig(mouth_id, image=load_image("mouth_closed.png"))
        root.update()
        time.sleep(0.15)
    canvas.delete(mouth_id)

# TTS 播放函式
def engine_runner(text):
    engine.say(text)
    engine.runAndWait()

# 主邏輯：輸入文字 → 表情判斷 + 嘴巴動畫
def vtuber_say(text):
    global face_id, mouth_id

    # 中文轉英文（翻譯失敗則保留原文）
    try:
        english_text = GoogleTranslator(source='zh-TW', target='en').translate(text)
    except Exception as e:
        print("⚠️ 翻譯失敗，使用原文：", e)
        english_text = text

    # 情緒分析
    result = emotion_classifier(english_text)[0]
    label = result["label"]

    # 顯示情緒名稱
    emotion_label.config(text=f"目前表情：{label}")

    # 表情圖片對應
    if label == "joy":
        face_img = "happy.png"
    elif label == "sadness":
        face_img = "sad.png"
    elif label == "anger":
        face_img = "angry.png"
    elif label == "surprise":
        face_img = "surprise.png"
    else:
        face_img = "neutral.png"

    # 清除前一張臉與嘴巴
    if face_id:
        canvas.delete(face_id)
    if mouth_id:
        canvas.delete(mouth_id)

    # 顯示新臉與嘴巴（初始閉口）
    face_id = canvas.create_image(0, 0, anchor=tk.NW, image=load_image(face_img))
    mouth_id = canvas.create_image(0, 0, anchor=tk.NW, image=load_image("mouth_closed.png"))

    # 同時啟動聲音與嘴巴動畫
    threading.Thread(target=engine_runner, args=(text,)).start()
    threading.Thread(target=speak_and_animate, args=(text,)).start()

# 鍵盤輸入模式
def run_vtuber():
    text = entry.get()
    vtuber_say(text)

# 錄音 + 語音辨識
def record_and_recognize():
    recognizer = sr.Recognizer()
    mic = sr.Microphone()

    print("🎤 開始錄音...")
    with mic as source:
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)

    try:
        print("🧠 辨識中...")
        text = recognizer.recognize_google(audio, language='zh-TW')
        print("✅ 你說了：", text)
        entry.delete(0, tk.END)
        entry.insert(0, text)
        vtuber_say(text)
    except sr.UnknownValueError:
        print("😅 聽不清楚...")
    except sr.RequestError as e:
        print("❌ Google 辨識出錯：", e)

# 介面元件
entry = tk.Entry(root, width=40, font=('Arial', 16))
entry.pack(pady=10)

btn_text = tk.Button(root, text="用文字讓 Vtuber 說話", command=run_vtuber, font=('Arial', 14))
btn_text.pack(pady=5)

btn_voice = tk.Button(root, text="🎙️ 錄音讓 Vtuber 模仿你", command=lambda: threading.Thread(target=record_and_recognize).start(), font=('Arial', 14))
btn_voice.pack(pady=5)

root.mainloop()
