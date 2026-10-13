"""
Speech to Text Desktop App
===========================
A fast, accurate, local speech-to-text recorder built on OpenAI Whisper
(via the openai-whisper / PyTorch backend, chosen for its much more
reliable Windows install story than ctranslate2-based alternatives).

Features:
  - Open Records folder / Run (start-stop) recording buttons
  - Near-live transcription (speech is chunked on natural pauses and
    transcribed while you keep talking, instead of waiting until you stop)
  - Auto GPU/CPU detection, with a Fast / Balanced / Accurate quality preset
  - Microphone picker
  - Live input level meter so you can see the mic is actually picking you up
  - Session timer + running word count
  - Each day's speech is appended to records/YYYY-MM-DD.txt (new session =
    new timestamped header inside that day's file)
  - Search across every past transcript from inside the app
  - Remembers your last mic/quality/language choice between runs

Run with:  python speech_to_text_app.py
(see README.md for one-time setup)
"""

import collections
import json
import os
import queue
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import sounddevice as sd
import tkinter as tk
from tkinter import ttk, messagebox

try:
    import torch
    import whisper as openai_whisper
except ImportError:
    torch = None
    openai_whisper = None


# --------------------------------------------------------------------------
# Paths / constants
# --------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
RECORDS_DIR = BASE_DIR / "records"
SETTINGS_PATH = BASE_DIR / "settings.json"

SAMPLE_RATE = 16000
CHANNELS = 1
BLOCK_MS = 50                  # audio callback block size (smaller = snappier detection + smoother waveform)
SILENCE_TIMEOUT_S = 0.55       # pause length that ends an utterance (shorter = text lands faster)
MAX_CHUNK_S = 8                # force-flush a chunk even without a pause (keeps long speech incremental)
TRAILING_PAD_S = 0.2           # trailing silence kept for natural cadence; the rest is trimmed before transcribing
SPEECH_RMS_THRESHOLD = 0.012   # roughly-tuned mic level considered "speech"

WAVEFORM_HISTORY_LEN = 150     # number of columns shown in the live waveform

QUALITY_PRESETS = {
    # name: (english_model, multilingual_model, beam_size)
    "Fast":     ("tiny.en",   "tiny",   1),
    "Balanced": ("base.en",   "base",   1),
    "Accurate": ("small.en",  "small",  5),
    "Max (slow on CPU)": ("medium.en", "medium", 5),
}

LANGUAGES = ["English", "Auto-detect"]


# --------------------------------------------------------------------------
# Settings persistence
# --------------------------------------------------------------------------
def load_settings() -> dict:
    if SETTINGS_PATH.exists():
        try:
            return json.loads(SETTINGS_PATH.read_text())
        except Exception:
            pass
    return {}


def save_settings(data: dict) -> None:
    try:
        SETTINGS_PATH.write_text(json.dumps(data, indent=2))
    except Exception:
        pass


# --------------------------------------------------------------------------
# Audio capture + utterance chunking (silence-based VAD)
# --------------------------------------------------------------------------
class AudioEngine:
    """Captures mic audio and slices it into utterances on natural pauses,
    handing each finished utterance off for transcription."""

    def __init__(self, chunk_callback, level_callback, error_callback, device=None):
        self.chunk_callback = chunk_callback
        self.level_callback = level_callback
        self.error_callback = error_callback
        self.device = device

        self._stream = None
        self._buffer = []          # list of int16 numpy blocks for current utterance
        self._buffer_seconds = 0.0
        self._silence_seconds = 0.0
        self._has_speech = False
        self._running = False
        self._last_speech_block_index = -1   # index into _buffer of the most recent speech block

    @staticmethod
    def list_input_devices():
        devices = []
        try:
            for idx, d in enumerate(sd.query_devices()):
                if d.get("max_input_channels", 0) > 0:
                    devices.append((idx, d["name"]))
        except Exception:
            pass
        return devices

    def start(self):
        if self._running:
            return
        self._buffer = []
        self._buffer_seconds = 0.0
        self._silence_seconds = 0.0
        self._has_speech = False
        self._last_speech_block_index = -1
        self._running = True
        blocksize = int(SAMPLE_RATE * BLOCK_MS / 1000)
        try:
            self._stream = sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=CHANNELS,
                dtype="int16",
                blocksize=blocksize,
                device=self.device,
                callback=self._on_audio,
            )
            self._stream.start()
        except Exception as e:
            self._running = False
            self.error_callback(f"Could not open microphone:\n{e}")

    def stop(self):
        self._running = False
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None
        # flush whatever's left as a final utterance
        self._flush()

    def _flush(self):
        if self._buffer and self._has_speech:
            # Trim excess trailing silence (kept only long enough to hear the
            # pause naturally) so we hand Whisper less dead air to chew
            # through - shorter input means faster transcription per chunk.
            pad_blocks = max(0, round(TRAILING_PAD_S / (BLOCK_MS / 1000)))
            keep_to = min(len(self._buffer), self._last_speech_block_index + 1 + pad_blocks)
            trimmed = self._buffer[:keep_to] if keep_to > 0 else self._buffer
            audio = np.concatenate(trimmed)
            self.chunk_callback(audio)
        self._buffer = []
        self._buffer_seconds = 0.0
        self._silence_seconds = 0.0
        self._has_speech = False
        self._last_speech_block_index = -1

    def _on_audio(self, indata, frames, time_info, status):
        if not self._running:
            return
        block = indata[:, 0].copy()
        block_seconds = len(block) / SAMPLE_RATE

        rms = float(np.sqrt(np.mean((block.astype(np.float32) / 32768.0) ** 2)))
        self.level_callback(rms)

        is_speech = rms > SPEECH_RMS_THRESHOLD

        if is_speech:
            self._has_speech = True
            self._silence_seconds = 0.0
            self._buffer.append(block)
            self._last_speech_block_index = len(self._buffer) - 1
            self._buffer_seconds += block_seconds
        elif self._has_speech:
            # trailing silence after speech - keep a little context, count it
            self._buffer.append(block)
            self._buffer_seconds += block_seconds
            self._silence_seconds += block_seconds

        if self._has_speech and (
            self._silence_seconds >= SILENCE_TIMEOUT_S
            or self._buffer_seconds >= MAX_CHUNK_S
        ):
            self._flush()


# --------------------------------------------------------------------------
# Transcription worker (background thread, own queue)
# --------------------------------------------------------------------------
class Transcriber:
    def __init__(self, result_queue: queue.Queue, status_callback):
        self.result_queue = result_queue
        self.status_callback = status_callback
        self.model = None
        self.model_key = None  # (model_name,) currently loaded
        self.audio_queue = queue.Queue()
        self.language = "en"
        self.beam_size = 1
        self._worker_thread = None
        self._stop = False

    def ensure_model(self, model_name: str):
        if self.model_key == model_name and self.model is not None:
            return
        self.status_callback(f"Loading model '{model_name}' ...")
        try:
            device = "cuda" if (torch is not None and torch.cuda.is_available()) else "cpu"
            self.model = openai_whisper.load_model(model_name, device=device)
            self.model_key = model_name
            self.status_callback(f"Model ready ({device}).")
        except Exception as e:
            self.model = None
            self.model_key = None
            self.status_callback(f"Model load failed: {e}")
            raise

    def start_worker(self):
        self._stop = False
        self._worker_thread = threading.Thread(target=self._run, daemon=True)
        self._worker_thread.start()

    def stop_worker(self):
        self._stop = True
        self.audio_queue.put(None)  # wake the thread so it can exit

    def submit(self, audio_int16: np.ndarray):
        self.audio_queue.put(audio_int16)

    def _run(self):
        while not self._stop:
            item = self.audio_queue.get()
            if item is None:
                continue
            audio_float = item.astype(np.float32) / 32768.0
            try:
                use_fp16 = torch is not None and torch.cuda.is_available()
                result = self.model.transcribe(
                    audio_float,
                    language=self.language,
                    beam_size=self.beam_size,
                    fp16=use_fp16,
                    # each chunk is an independent utterance - don't waste
                    # time conditioning on unrelated prior text
                    condition_on_previous_text=False,
                    without_timestamps=True,
                )
                text = result.get("text", "").strip()
            except Exception as e:
                self.result_queue.put(("error", f"Transcription error: {e}"))
                continue
            if text:
                self.result_queue.put(("text", text))


# --------------------------------------------------------------------------
# Main application
# --------------------------------------------------------------------------
class SpeechToTextApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Speech to Text")
        self.geometry("760x600")
        self.minsize(640, 500)

        RECORDS_DIR.mkdir(exist_ok=True)

        self.settings = load_settings()
        self.recording = False
        self.session_start = None
        self.session_path = None
        self.word_count = 0
        self.timer_job = None

        self.level_queue = queue.Queue()
        self.result_queue = queue.Queue()

        self.transcriber = Transcriber(self.result_queue, self._set_status_threadsafe)
        self.audio_engine = None  # created on Run, needs chosen device

        self._build_ui()
        self._poll_queues()
        # Deferred via `after` rather than called directly: the background
        # warm-up thread touches Tk (status label) through `self.after`,
        # which needs the mainloop already running or it raises
        # "main thread is not in main loop". Scheduling it guarantees
        # mainloop has started by the time it fires.
        self.after(50, self._warm_model_async)

        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.bind("<Control-r>", lambda e: self._toggle_recording())

    # ---------------------------------------------------------- UI layout
    def _build_ui(self):
        pad = {"padx": 8, "pady": 6}

        top = ttk.Frame(self)
        top.pack(fill="x", **pad)

        self.run_btn = ttk.Button(top, text="\u25b6  Run", command=self._toggle_recording, width=12)
        self.run_btn.pack(side="left")

        ttk.Button(top, text="Open Records", command=self._open_records).pack(side="left", padx=(8, 0))

        self.timer_label = ttk.Label(top, text="00:00:00", font=("Consolas", 12))
        self.timer_label.pack(side="right")

        # settings row
        settings = ttk.Frame(self)
        settings.pack(fill="x", **pad)

        ttk.Label(settings, text="Microphone:").grid(row=0, column=0, sticky="w")
        self.mic_var = tk.StringVar()
        self.mic_combo = ttk.Combobox(settings, textvariable=self.mic_var, state="readonly", width=32)
        self.mic_combo.grid(row=0, column=1, sticky="w", padx=(4, 16))
        self._refresh_mics()

        ttk.Label(settings, text="Quality:").grid(row=0, column=2, sticky="w")
        self.quality_var = tk.StringVar(value=self.settings.get("quality", "Balanced"))
        quality_combo = ttk.Combobox(
            settings, textvariable=self.quality_var, state="readonly",
            values=list(QUALITY_PRESETS.keys()), width=18,
        )
        quality_combo.grid(row=0, column=3, sticky="w", padx=(4, 16))
        quality_combo.bind("<<ComboboxSelected>>", lambda e: self._warm_model_async())

        ttk.Label(settings, text="Language:").grid(row=0, column=4, sticky="w")
        self.language_var = tk.StringVar(value=self.settings.get("language", "English"))
        lang_combo = ttk.Combobox(
            settings, textvariable=self.language_var, state="readonly",
            values=LANGUAGES, width=12,
        )
        lang_combo.grid(row=0, column=5, sticky="w")
        lang_combo.bind("<<ComboboxSelected>>", lambda e: self._warm_model_async())

        # level meter
        level_frame = ttk.Frame(self)
        level_frame.pack(fill="x", **pad)
        ttk.Label(level_frame, text="Mic level:").pack(side="left")
        self.level_bar = ttk.Progressbar(level_frame, maximum=100, length=300)
        self.level_bar.pack(side="left", padx=(6, 0), fill="x", expand=True)

        # live waveform
        wave_frame = ttk.Frame(self)
        wave_frame.pack(fill="x", **pad)
        self.level_history = collections.deque([0.0] * WAVEFORM_HISTORY_LEN, maxlen=WAVEFORM_HISTORY_LEN)
        self.wave_canvas = tk.Canvas(wave_frame, height=64, bg="#fbfbfb",
                                      highlightthickness=1, highlightbackground="#cccccc")
        self.wave_canvas.pack(fill="x", expand=True)
        self.wave_canvas.bind("<Configure>", lambda e: self._draw_waveform())

        # status
        self.status_var = tk.StringVar(value="Idle. Choose a mic and press Run to start.")
        ttk.Label(self, textvariable=self.status_var, foreground="#555").pack(fill="x", padx=8)

        # live transcript
        text_frame = ttk.Frame(self)
        text_frame.pack(fill="both", expand=True, **pad)

        self.transcript_box = tk.Text(text_frame, wrap="word", font=("Segoe UI", 11), undo=True)
        scrollbar = ttk.Scrollbar(text_frame, command=self.transcript_box.yview)
        self.transcript_box.configure(yscrollcommand=scrollbar.set)
        self.transcript_box.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # bottom bar: word count + search
        bottom = ttk.Frame(self)
        bottom.pack(fill="x", **pad)

        self.wordcount_var = tk.StringVar(value="0 words")
        ttk.Label(bottom, textvariable=self.wordcount_var).pack(side="left")

        ttk.Label(bottom, text="Search past transcripts:").pack(side="left", padx=(20, 4))
        self.search_var = tk.StringVar()
        search_entry = ttk.Entry(bottom, textvariable=self.search_var, width=24)
        search_entry.pack(side="left")
        search_entry.bind("<Return>", lambda e: self._search_records())
        ttk.Button(bottom, text="Search", command=self._search_records).pack(side="left", padx=(4, 0))

    def _refresh_mics(self):
        devices = AudioEngine.list_input_devices()
        self._mic_devices = devices
        names = [name for _, name in devices]
        self.mic_combo["values"] = names
        saved_name = self.settings.get("mic_name")
        if saved_name and saved_name in names:
            self.mic_var.set(saved_name)
        elif names:
            self.mic_var.set(names[0])

    def _selected_device_index(self):
        name = self.mic_var.get()
        for idx, dname in getattr(self, "_mic_devices", []):
            if dname == name:
                return idx
        return None

    # ------------------------------------------------------ model warm-up
    def _current_model_name(self):
        preset = QUALITY_PRESETS.get(self.quality_var.get(), QUALITY_PRESETS["Balanced"])
        return preset[0] if self.language_var.get() == "English" else preset[1]

    def _warm_model_async(self):
        """Load the currently-selected Whisper model in the background so
        it's already warm by the time Run is pressed, instead of making the
        user wait for it after clicking Run."""
        if openai_whisper is None:
            return
        model_name = self._current_model_name()

        def go():
            try:
                self.transcriber.ensure_model(model_name)
            except Exception:
                pass

        threading.Thread(target=go, daemon=True).start()

    # ------------------------------------------------------- record/stop
    def _toggle_recording(self):
        if self.recording:
            self._stop_recording()
        else:
            self._start_recording()

    def _start_recording(self):
        if openai_whisper is None:
            messagebox.showerror(
                "Missing dependency",
                "openai-whisper / torch is not installed.\nRun: pip install -r requirements.txt",
            )
            return

        preset = QUALITY_PRESETS.get(self.quality_var.get(), QUALITY_PRESETS["Balanced"])
        model_name = self._current_model_name()
        self.transcriber.language = "en" if self.language_var.get() == "English" else None
        self.transcriber.beam_size = preset[2]

        self.run_btn.config(state="disabled")
        self.status_var.set("Loading model, one moment...")

        def load_and_go():
            try:
                # no-op if _warm_model_async already loaded this exact model
                self.transcriber.ensure_model(model_name)
            except Exception:
                self.after(0, lambda: self.run_btn.config(state="normal"))
                return
            self.after(0, self._begin_session)

        threading.Thread(target=load_and_go, daemon=True).start()

    def _begin_session(self):
        self.transcriber.start_worker()

        device_idx = self._selected_device_index()
        self.audio_engine = AudioEngine(
            chunk_callback=self.transcriber.submit,
            level_callback=lambda rms: self.level_queue.put(rms),
            error_callback=self._audio_error_threadsafe,
            device=device_idx,
        )
        self.audio_engine.start()

        self.session_start = datetime.now()
        self.session_path = RECORDS_DIR / f"{self.session_start.strftime('%Y-%m-%d')}.txt"
        header = f"\n--- Session started {self.session_start.strftime('%Y-%m-%d %H:%M:%S')} ---\n"
        with open(self.session_path, "a", encoding="utf-8") as f:
            f.write(header)

        self.transcript_box.insert("end", header)
        self.transcript_box.see("end")

        self.recording = True
        self.run_btn.config(text="\u25a0  Stop", state="normal")
        self.status_var.set(f"Recording -> {self.session_path.name}")
        self._tick_timer()

        # persist chosen settings
        self.settings.update({
            "mic_name": self.mic_var.get(),
            "quality": self.quality_var.get(),
            "language": self.language_var.get(),
        })
        save_settings(self.settings)

    def _stop_recording(self):
        self.recording = False
        if self.audio_engine:
            self.audio_engine.stop()
        self.transcriber.stop_worker()
        self.run_btn.config(text="\u25b6  Run")
        self.status_var.set(f"Stopped. Saved to {self.session_path.name}" if self.session_path else "Stopped.")
        self.level_bar["value"] = 0
        if self.timer_job:
            self.after_cancel(self.timer_job)
            self.timer_job = None

    def _tick_timer(self):
        if not self.recording or self.session_start is None:
            return
        elapsed = int((datetime.now() - self.session_start).total_seconds())
        h, rem = divmod(elapsed, 3600)
        m, s = divmod(rem, 60)
        self.timer_label.config(text=f"{h:02d}:{m:02d}:{s:02d}")
        self.timer_job = self.after(1000, self._tick_timer)

    # --------------------------------------------------------- queue pump
    def _poll_queues(self):
        # level meter + waveform history
        new_levels = []
        try:
            while True:
                new_levels.append(self.level_queue.get_nowait())
        except queue.Empty:
            pass
        if new_levels:
            self.level_history.extend(new_levels)
            pct = min(100, int(new_levels[-1] * 100 / 0.15))  # rough scale to 0-100
            self.level_bar["value"] = pct
            self._draw_waveform()

        # transcription results
        try:
            while True:
                kind, payload = self.result_queue.get_nowait()
                if kind == "text":
                    self._append_transcript(payload)
                elif kind == "error":
                    self.status_var.set(payload)
        except queue.Empty:
            pass

        self.after(80, self._poll_queues)

    def _draw_waveform(self):
        canvas = self.wave_canvas
        canvas.delete("all")
        width = canvas.winfo_width()
        height = canvas.winfo_height()
        if width <= 1 or height <= 1:
            return
        mid_y = height / 2
        history = list(self.level_history)
        n = len(history)
        if n == 0:
            return
        canvas.create_line(0, mid_y, width, mid_y, fill="#dddddd")
        bar_width = width / n
        x = 0.0
        for level in history:
            amp = min(1.0, level / 0.2)  # normalize against a rough loud-speech ceiling
            bar_height = amp * (height / 2 - 2)
            cx = x + bar_width / 2
            canvas.create_line(
                cx, mid_y - bar_height, cx, mid_y + bar_height,
                fill="#4a90d9", width=max(1.0, bar_width * 0.7),
            )
            x += bar_width

    def _append_transcript(self, text: str):
        stamp = datetime.now().strftime("%H:%M:%S")
        line = f"[{stamp}] {text}\n"
        self.transcript_box.insert("end", line)
        self.transcript_box.see("end")
        self.word_count += len(text.split())
        self.wordcount_var.set(f"{self.word_count} words")
        if self.session_path:
            with open(self.session_path, "a", encoding="utf-8") as f:
                f.write(line)

    def _set_status_threadsafe(self, msg: str):
        self.after(0, lambda: self.status_var.set(msg))

    def _audio_error_threadsafe(self, msg: str):
        def show():
            self.status_var.set("Microphone error.")
            messagebox.showerror("Microphone error", msg)
            self._stop_recording()
            self.run_btn.config(state="normal")
        self.after(0, show)

    # -------------------------------------------------------------- misc
    def _open_records(self):
        RECORDS_DIR.mkdir(exist_ok=True)
        try:
            if sys.platform.startswith("win"):
                os.startfile(RECORDS_DIR)  # type: ignore[attr-defined]
            elif sys.platform == "darwin":
                subprocess.Popen(["open", str(RECORDS_DIR)])
            else:
                subprocess.Popen(["xdg-open", str(RECORDS_DIR)])
        except Exception as e:
            messagebox.showerror("Couldn't open folder", str(e))

    def _search_records(self):
        query = self.search_var.get().strip()
        if not query:
            return
        results = []
        for path in sorted(RECORDS_DIR.glob("*.txt")):
            try:
                lines = path.read_text(encoding="utf-8").splitlines()
            except Exception:
                continue
            for i, line in enumerate(lines, start=1):
                if query.lower() in line.lower():
                    results.append((path.name, i, line.strip()))

        win = tk.Toplevel(self)
        win.title(f"Search results: \"{query}\"")
        win.geometry("560x360")

        if not results:
            ttk.Label(win, text="No matches found.").pack(padx=12, pady=12)
            return

        listbox = tk.Listbox(win, font=("Segoe UI", 10))
        listbox.pack(fill="both", expand=True, padx=8, pady=8)
        for fname, lineno, text in results:
            listbox.insert("end", f"{fname}:{lineno}  {text}")

        def open_selected(event=None):
            sel = listbox.curselection()
            if not sel:
                return
            fname = results[sel[0]][0]
            path = RECORDS_DIR / fname
            try:
                if sys.platform.startswith("win"):
                    os.startfile(path)  # type: ignore[attr-defined]
                elif sys.platform == "darwin":
                    subprocess.Popen(["open", str(path)])
                else:
                    subprocess.Popen(["xdg-open", str(path)])
            except Exception as e:
                messagebox.showerror("Couldn't open file", str(e))

        listbox.bind("<Double-Button-1>", open_selected)

    def _on_close(self):
        if self.recording:
            self._stop_recording()
        self.destroy()


if __name__ == "__main__":
    app = SpeechToTextApp()
    app.mainloop()
