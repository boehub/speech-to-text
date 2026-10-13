# Speech to Text

A fast, accurate, fully local speech-to-text recorder, built on OpenAI's
Whisper via PyTorch. No cloud API, no internet connection needed once the
model is downloaded — your voice never leaves your machine.

> The `records/` folder (your transcripts) and `settings.json` (your microphone and quality choices) are created automatically the first time you run the app. They stay on your own machine and are not part of this repository.

## What it does

- **Open Records** — opens the `records/` folder in File Explorer.
- **Run** — starts listening. It transcribes as you talk (chunking on
  natural pauses in your speech, roughly every sentence) so you see text
  appear live instead of waiting until you stop. Click **Stop** to end the
  session.
- Every day's speech is saved to `records/YYYY-MM-DD.txt`. If you record
  more than once in a day, each new run appends a new timestamped session
  to that same file instead of overwriting it.

### Extra features included

| Feature | What it's for |
|---|---|
| Mic picker | Choose which input device to record from |
| Quality preset (Fast / Balanced / Accurate / Max) | Trade off speed vs. accuracy depending on your hardware |
| Language (English / Auto-detect) | English-only models are faster & more accurate for English speech |
| Live level meter | Confirms your mic is actually picking up sound |
| Live waveform | Scrolling visual of your recent audio, right under the level meter |
| Background model preloading | Whisper loads the moment the app opens (and again in the background whenever you change Quality/Language), so pressing Run doesn't make you wait on a cold model load |
| Session timer & live word count | |
| Search past transcripts | Search box at the bottom searches every file in `records/`; double-click a result to open that day's file |
| Remembers your settings | Mic/quality/language choices persist between runs (`settings.json`) |
| Auto GPU detection | Uses your NVIDIA GPU automatically if one is available, otherwise falls back to CPU |
| `Ctrl+R` hotkey | Start/stop without touching the mouse |

## One-time setup

1. Install Python 3.9+ if you don't already have it.
2. Open a terminal in this folder and install dependencies:

   ```
   pip install -r requirements.txt
   ```

   This pulls in PyTorch, which is a sizeable download (several hundred
   MB) — that's normal, let it finish.

   - On Windows, `sounddevice` should install cleanly with no extra steps
     (it ships its own PortAudio binaries).
   - If you have an **NVIDIA GPU**, transcription will automatically run on
     it and be noticeably faster — you don't need to install anything extra
     for the "Accurate"/"Max" presets to still feel responsive. The plain
     `pip install` above already works fine on CPU-only machines too, it's
     just slower on the bigger presets.
   - This app deliberately uses the `openai-whisper` (PyTorch) backend
     rather than `faster-whisper`/`ctranslate2`. The latter is faster in
     principle, but its Windows installs are prone to a stubborn
     `ctranslate2.dll (or one of its dependencies)` load error that can
     require deep Windows Installer/registry repair to fix. PyTorch's
     Windows wheels are far more self-contained and don't have this
     problem — reliability won out over raw speed here.

3. Run it:

   ```
   python speech_to_text_app.py
   ```

The first time you pick a given quality preset, that Whisper model gets
downloaded automatically (a few hundred MB to ~1.5GB depending on preset)
and cached for next time — that's the only step that needs internet.

## Choosing a quality preset

- **Fast** (`tiny`) — near-instant, good for quick notes, least accurate.
- **Balanced** (`base`) — good everyday default. Recommended starting point
  on a laptop CPU.
- **Accurate** (`small`) — noticeably better accuracy, still fine on a
  decent CPU or any GPU.
- **Max** (`medium`) — best accuracy, but slow on CPU-only machines. Best
  reserved for a machine with a GPU.

You can switch presets any time from the dropdown before pressing Run.

## Notes

- Transcription happens in short chunks split on natural pauses in your
  speech (not fixed time windows), so words don't get cut off mid-sentence.
  Pauses of about half a second trigger a chunk, and any single stretch of
  continuous speech gets force-flushed every 8 seconds so long monologues
  still show up incrementally instead of all landing at once.
- If nothing is happening when you talk, check the level meter — if it
  stays flat, pick a different microphone from the dropdown.
- All files are plain `.txt` — no formatting, easy to search, copy, or feed
  into anything else.

## Performance tuning

A few constants near the top of `speech_to_text_app.py` control how snappy
vs. how forgiving the live transcription feels — tweak them if it's
cutting you off too eagerly or feels laggy:

| Constant | Default | Effect |
|---|---|---|
| `SILENCE_TIMEOUT_S` | `0.55` | How long a pause has to be before that phrase gets sent off for transcription. Lower = text appears faster but can split sentences you paused mid-thought in; higher = fewer splits but more lag. |
| `MAX_CHUNK_S` | `8` | Longest a single unbroken stretch of speech can run before it's force-transcribed anyway. Lower = more frequent updates during long talking; higher = fewer but larger chunks. |
| `TRAILING_PAD_S` | `0.2` | How much trailing silence is kept (for natural cadence) before the rest gets trimmed off each chunk before sending it to Whisper — trimming makes each chunk faster to transcribe. |
| `BLOCK_MS` | `50` | Audio callback granularity — also controls waveform smoothness. Lower = more responsive pause detection and a smoother waveform, at a small CPU cost. |
