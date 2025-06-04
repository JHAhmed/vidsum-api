# VidSum API – FastAPI Backend for VidSum

This is the backend service for [VidSum](https://github.com/JHAhmed/vidsum), a YouTube video summarizer. It’s built with [FastAPI](https://fastapi.tiangolo.com/) and leverages `yt-dlp` (with `ffmpeg`) to download and process videos.

> [!NOTE]
> This API is intended to be used alongside the [VidSum frontend](https://github.com/JHAhmed/vidsum). It will not function meaningfully on its own.


---

## Features

* **YouTube Audio Extraction:** Downloads and processes video/audio using `yt-dlp`.
* **Transcript Processing:** Converts audio to text using external AI APIs (e.g. Gemini, OpenAI).
* **Summarization API:** Sends transcripts to LLMs and returns structured summaries.

---

## Tech Stack

* Python 3.10+
* FastAPI
* yt-dlp (uses `ffmpeg` and `ffprobe` binaries under `bin/`)

---

## Getting Started

1. **Clone the repo**

   ```bash
   git clone https://github.com/JHAhmed/vidsum-api.git
   cd vidsum-api
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Ensure ffmpeg is available**

   * `bin/ffmpeg.exe` and `bin/ffprobe.exe` are bundled.
   * NOTE: These binaries only work on Windows. If you're on Linux or macOS, you need to install `ffmpeg` and `ffprobe` system-wide.
   * These will be used directly by `yt-dlp`. No need for system-wide installation.

4. **Configure Environment**

   Copy `.env.example` to `.env` and add your API keys:

   ```bash
   cp .env.example .env
   ```

   Fill in the required environment variables (e.g. OpenAI or Gemini credentials).

5. **Run the server**

   ```bash
   uvicorn api.index:app --reload --port 8000
   ```

   Server runs on `http://localhost:8000` by default.

---

## API Endpoints

The core endpoints are consumed by the VidSum frontend. Most users won’t need to interact with them directly.

| Method | Endpoint       | Description                          |
| ------ | -------------- | ------------------------------------ |
| POST   | `/info`       | Accepts YouTube URL, returns info   |
| POST   | `/captions`   | Accepts YouTube URL, returns captions |
| POST   | `/transcribe`  | Accepts YouTube URL, transcribes audio, returns transcript |
| POST   | `/frames`  | Accepts YouTube URL, extracts video frames at given intervals, returns frame data as base64 |

---

## Notes

* This repo is backend-only. For the full experience, use it with [vidsum](https://github.com/JHAhmed/vidsum).
* Avoid modifying unless you're extending or debugging.

> Built with ❤️ using FastAPI.
