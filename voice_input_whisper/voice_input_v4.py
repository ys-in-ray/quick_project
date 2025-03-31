from faster_whisper import WhisperModel
import pyaudio
import wave
import threading
import time
import os
import tkinter as tk
from tkinter import scrolledtext
import keyboard
import pyperclip
from opencc import OpenCC

model = WhisperModel("base", device="cpu", compute_type="int8")

FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 1024

frames = []
recording = False
stream = None

p = pyaudio.PyAudio()

# 非阻塞地錄音
def capture_audio(window):
    global stream, frames, recording
    if recording:
        data = stream.read(CHUNK, exception_on_overflow=False)
        frames.append(data)
        window.after(50, lambda: capture_audio(window))

# GUI 初始化
def setup_gui():
    window = tk.Tk()
    window.title("語音輸入法 GUI")
    window.geometry("700x500")
    window.configure(bg="#f5f5f5")

    window.rowconfigure(0, weight=4)
    window.rowconfigure(1, weight=0)
    window.rowconfigure(2, weight=0)
    window.columnconfigure(0, weight=1)

    # 輸出文字區塊
    text_area = scrolledtext.ScrolledText(window, wrap=tk.WORD, font=("Microsoft JhengHei", 14), height=10)
    text_area.grid(row=0, column=0, sticky="nsew", padx=15, pady=(15, 5))

    # log 顯示區塊
    log_area = scrolledtext.ScrolledText(window, wrap=tk.WORD, font=("Consolas", 10), height=3, bg="#eeeeee")
    log_area.grid(row=1, column=0, sticky="ew", padx=15, pady=5)
    log_area.insert(tk.END, "🔧 系統日誌：\n")
    log_area.config(state=tk.DISABLED)

    # 錄音控制按鈕
    button = tk.Button(window, text="🎙️ 開始錄音", font=("Microsoft JhengHei", 14), bg="#4CAF50", fg="white", width=20)
    button.grid(row=2, column=0, pady=10)

    def log(msg):
        log_area.config(state=tk.NORMAL)
        log_area.insert(tk.END, f"{msg}\n")
        log_area.see(tk.END)
        log_area.config(state=tk.DISABLED)

    def toggle_record():
        nonlocal button
        if not recording:
            button.config(text="🛑 停止 (錄音中...)", bg="#f44336")
            threading.Thread(target=start_recording, args=(window, log)).start()
        else:
            button.config(text="🎙️ 開始錄音", bg="#4CAF50")
            threading.Thread(target=stop_and_finalize, args=(text_area, log)).start()

    button.config(command=toggle_record)

    keyboard.add_hotkey("ctrl+alt+z", toggle_record)

    # 加入右鍵選單功能
    def create_context_menu(widget):
        menu = tk.Menu(widget, tearoff=0)
        menu.add_command(label="複製", command=lambda: widget.event_generate("<<Copy>>"))
        menu.add_command(label="貼上", command=lambda: widget.event_generate("<<Paste>>"))
        menu.add_command(label="全選", command=lambda: widget.event_generate("<<SelectAll>>"))

        def popup(event):
            menu.tk_popup(event.x_root, event.y_root)

        widget.bind("<Button-3>", popup)

    create_context_menu(text_area)
    create_context_menu(log_area)

        # 顯示快捷鍵提示文字
    shortcut_label = tk.Label(window, text="快捷鍵：Ctrl + Alt + Z 開始/停止錄音", font=("Microsoft JhengHei", 10), bg="#f5f5f5", fg="#555")
    shortcut_label.grid(row=3, column=0, pady=(0, 10))

    window.mainloop()

# 開始錄音
def start_recording(window, log):
    global stream, frames, recording
    frames = []
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE,
                    input=True, frames_per_buffer=CHUNK)
    recording = True
    log("🎙️ 開始錄音...")
    capture_audio(window)

# 停止錄音並輸出最終文字
def stop_and_finalize(output_widget, log):
    global stream, recording
    recording = False
    stream.stop_stream()
    stream.close()

    log("🛑 錄音結束，輸出最終結果...")
    audio_data = b''.join(frames)

    final_filename = "final_output.wav"
    wf = wave.open(final_filename, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(audio_data)
    wf.close()

    try:
        if os.path.getsize(final_filename) == 0:
            raise ValueError("輸出音訊檔案為空，無法辨識。")
        segments, _ = model.transcribe(final_filename, language="zh")
        text = "".join([segment.text for segment in segments]).strip()

        # 強制轉繁體
        try:
            converter = OpenCC("s2t")
            text_t = converter.convert(text)  # 簡體轉繁體
            if(text_t!=text):
                log(f"繁體轉換：{text}->{text_t}")
                text = text_t
                
        except Exception as conv_err:
            log(f"⚠️ 繁體轉換失敗：{conv_err}")
        output_widget.insert(tk.END, text + "\n")
        pyperclip.copy(text)
        log(f"📝 輸出（已複製）：{text}")
    except Exception as e:
        log(f"❌ 最終辨識錯誤：{e}")
    finally:
        if os.path.exists(final_filename):
            os.remove(final_filename)

# 主程式
if __name__ == "__main__":
    setup_gui()
