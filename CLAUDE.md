# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Setup

```bash
# Install system dependencies (Linux)
sudo apt update && sudo apt install ffmpeg portaudio19-dev

# Install Ollama and pull the LLM
curl -fsSL https://ollama.com/install.sh | sh
ollama pull gemma3:270m

# Install Python dependencies
pip install -r requirements.txt
# Also required (not in requirements.txt):
pip install pyaudio numpy openwakeword
```

## Running

```bash
# Main voice assistant
python app.py

# Isolated component tests
python openwakewortest.py   # Wake word detection only (with cooldown)
python whsipert.py          # Whisper transcription of audio.mp3
python reque.py             # Homey API connectivity and device listing
```

## Architecture

The app runs a single blocking loop — there is no async, no threading. Each iteration:

1. **Audio capture** — PyAudio reads 1280-sample chunks (80ms at 16kHz mono) from the default microphone.
2. **Wake word detection** — OpenWakeWord evaluates each chunk against `Hey_Potato_20260228_130124.onnx`. A confidence score > 0.5 triggers the pipeline.
3. **Speech transcription** — Whisper (`tiny.en`) transcribes the captured audio to text.
4. **LLM command parsing** — The transcript plus the live device list is sent to Ollama (`gemma3:270m`). The LLM is prompted to embed `onoffcontrol(device_name, True/False)` calls directly in its plain-text response.
5. **Device control** — Regex extracts all `onoffcontrol(...)` calls from the LLM response and executes them against the Homey REST API.

### Homey API

Base URL: `http://192.168.3.30` (local network only). All requests use a Bearer token hardcoded at the top of `app.py`. Endpoints used:

- `GET /api/manager/devices/device` — full device list
- `GET /api/manager/zones/zone` — zone list for human-readable room names
- `PUT /api/manager/devices/device/{id}/capability/onoff` — toggle on/off with `{"value": true/false}`

Devices and zones are fetched once at startup. `devices` list is cleared and rebuilt each loop iteration before building the LLM prompt.

### Key model files

- `Hey_Potato_20260228_130124.onnx` — custom wake word model; both `MODEL_PATH` and `WAKEWORD_KEY` must match its filename stem.

### Known issue

Line 96 of `app.py` passes the PyAudio object (`audio`) to `model.transcribe()` instead of the recorded audio data (`audio_chunk`). Transcription currently does not use the live microphone audio.
