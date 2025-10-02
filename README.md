# KlipMind — Multimodal Video Analysis (API & Local)

> Extract **transcriptions**, **visual descriptions**, and **smart summaries** from videos. Run **100% locally** (Whisper + BLIP + Ollama) or via **APIs** (Groq + Gemini). Designed for **long clips**, **block-by-block summaries**, and a customizable **final overview**.

## 🔥**Join the waitlist to get early access!**
[![Join the Waitlist](https://img.shields.io/badge/Join%20the%20Waitlist-Click%20Here-blue?style=for-the-badge)](https://iaap4qo6zs2.typeform.com/to/J43jclr2)



## ✨ Features

- 🎙️ **Audio transcription** in blocks (FFmpeg + local Whisper or Groq Whisper).
- 🖼️ **Visual description** of representative frames (local BLIP or Gemini Vision).
- 🧠 **Multimodal summarization** (combines speech + visuals) with configurable **size**, **language**, and **persona**.
- 🧩 **Two execution modes**:
  - **Local**: no API keys required (faster-whisper + BLIP + Ollama).
  - **API**: Groq (STT + LLM) + Google Gemini (image description).
- 🧱 **Block processing** (`BLOCK_DURATION`) with an aggregated **final summary**.
- 🌐 Accepts **local file** or **URL** (via `utils/download_url.py`).

---

## 📁 Structure

```
video-analysis/
├─ api-models/
│  └─ main.py                 # Pipeline using Groq + Gemini
├─ local-models/
│  └─ main.py                 # Pipeline using Whisper/BLIP + Ollama
├─ utils/
│  ├─ __init__.py
│  └─ download_url.py         # Download from URLs (yt-dlp)
├─ downloads/                 # Downloaded videos / temp artifacts
└─ .gitignore
```

> Note: folder/file names may vary. Keep them as above to follow this guide verbatim.

---

## 🔧 Requirements

### Common
- **Python 3.10+**
- **FFmpeg** (required to extract audio)
- **OpenCV**, **Pillow**

#### FFmpeg installation
- **Windows (winget)**: `winget install Gyan.FFmpeg`
- **Windows (choco)**: `choco install ffmpeg`
- **macOS (brew)**: `brew install ffmpeg`
- **Ubuntu/Debian**: `sudo apt update && sudo apt install -y ffmpeg`

> Verify: `ffmpeg -version`

---

## 🚀 Getting Started (How to run)

### 1) Clone & create a virtual environment
```bash
git clone https://github.com/Ga0512/video-analysis.git
cd video-analysis
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
python -m pip install -U pip
```

### 2) Install common dependencies
```bash
pip install opencv-python pillow python-dotenv yt-dlp
```

### 3) Run in **API Mode** (Groq + Gemini)

**Install API-specific deps:**
```bash
pip install groq google-genai
```

**Create `.env` at the project root:**
```env
GROQ_API_KEY=put_your_groq_key_here
GOOGLE_API_KEY=put_your_gemini_key_here
```

**Set the video:**
- Edit `VIDEO_PATH` in `api-models/main.py` to a local file **or** a URL (YouTube/Instagram/etc.).  
  If it’s a URL, the script downloads it automatically via `utils/download_url.py`.

**(Optional) Tune parameters:**
At the end of `api-models/main.py`:
```python
BLOCK_DURATION = 30         # seconds per block
LANGUAGE = "english"        # "auto-detect" | "portuguese" | ...
SIZE = "large"              # "short" | "medium" | "large"
PERSONA = "Expert"
EXTRA_PROMPTS = "Write the summary as key bullet points."
```

**Run (from the repo root):**
```bash
python api-models/main.py
```

> **Important:** run from the **repo root** so `from utils.download_url import download` resolves correctly.

---

### 4) Run in **Local Mode** (Whisper + BLIP + Ollama)

**Install local-specific deps:**
```bash
pip install faster-whisper transformers
# PyTorch - pick your variant (CPU/CUDA) for your machine
# Example (CPU):
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

**Prepare Ollama & an LLM:**
```bash
# Install Ollama on your system, then:
ollama pull llama3.2:3b
```

**Set the video:**
- Edit `VIDEO_PATH` in `local-models/main.py` to a local file **or** a URL.  
  If it’s a URL, the script downloads it automatically via `utils/download_url.py`.

**(Optional) Tune parameters:**
At the end of `local-models/main.py`:
```python
BLOCK_DURATION = 30
LANGUAGE = "english"
SIZE = "large"
PERSONA = "Funny"
EXTRA_PROMPTS = "Write the summary as key bullet points."
```

**(Optional) Enable GPU for Whisper:**
In `initialize_models()`:
```python
WhisperModel("medium", device="cuda", compute_type="float16")
```

**Run (from the repo root):**
```bash
python local-models/main.py
```

---

## 🔒 Environment variables (.env)

```env
# API mode
GROQ_API_KEY=...
GOOGLE_API_KEY=...

# Optional
# HTTP(S)_PROXY=http://...
# CUDA_VISIBLE_DEVICES=0
```

> **Never** commit your `.env` to Git.

---

## 🧰 Utility (`utils/download_url.py`)

- Downloads videos from URLs (YouTube, Instagram, etc.) using **yt-dlp**.
- Saves to `downloads/` and returns the local path to feed the pipeline.
- If you need guaranteed MP4 with AAC audio, adjust yt-dlp/ffmpeg options there.

---

## 🗂️ Outputs

For each **block**:
- `start_time`, `end_time`
- `transcription` (speech for the segment)
- `frame_description` (visual description of the frame)
- `audio_summary` (multimodal summary for the block)

Final:
- **Final video summary** (aggregates all blocks).

> Currently printed to the terminal. It’s straightforward to extend to **JSON**, **SRT**, or **Markdown** exports.

---

## ⚠️ Important notes (common pitfalls)

- **Function signatures & param order**: ensure calls to `final_video_summary(...)` match the function signature (API and Local).
- **Image MIME for Gemini**: if you saved PNG, pass `mime_type='image/png'`.
- **Audio in Opus (Windows)**: if needed, re-encode to AAC with FFmpeg:
  ```bash
  ffmpeg -i input.ext -c:v libx264 -c:a aac -movflags +faststart output.mp4
  ```
- **`ModuleNotFoundError: No module named 'utils'`**: run scripts from the **repo root** and ensure `utils/__init__.py` exists.

---

## ⚙️ Performance tips

- **GPU** recommended: `WhisperModel(..., device="cuda", compute_type="float16")`.
- Adjust `BLOCK_DURATION` (shorter = finer captions; longer = faster processing).
- Tune `SIZE_TO_TOKENS` according to your LLM.
- For longer videos, cache per-block results to safely resume.

---

## 🧭 Roadmap (suggestions)

- [ ] Export **JSON/SRT/Markdown** (per block and final).
- [ ] CLI: `klipmind --video <path|url> --mode api|local --lang en --size large ...`
- [ ] Web UI (FastAPI/Streamlit) with upload/URL and progress bar.
- [ ] Multi-frame sampling per block.
- [ ] Model selection (Whisper tiny/base/…; BLIP variants; different LLMs).
- [ ] Unit tests for `utils/download_url.py` and parsers.

---

## 🤝 Contributing

Contributions are welcome!  
Open an issue with suggestions/bugs or submit a PR explaining the change.

---

## 📜 License

MIT

---

## 🙌 Credits

- **Whisper** (faster-whisper), **BLIP** (Salesforce), **Ollama** (local models)
- **Groq** (STT + Chat Completions-compatible LLM)
- **Gemini 2.0 Flash-Lite** for vision (frame description)
- **FFmpeg**, **OpenCV**, **Pillow**
