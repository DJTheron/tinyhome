from ollama import chat
import pyaudio
import numpy as np
from openwakeword.model import Model
import requests
import faster_whisper
import re
import os
from transformers import AutoModel, AutoTokenizer
from dotenv import load_dotenv

load_dotenv()

model_id = "LiquidAI/LFM2.5-Encoder-350M-Prompt-Router"
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModel.from_pretrained(model_id, trust_remote_code=True).eval()
#model = model.to("cuda")

startuproutes = ["Bedside Light On", "Bedside Light Off", "Desk Light Off", "Desk Light On", "Bedroom Light On", "Bedroom Light Off"]
startprompt = "warmup"

data = model.route(startprompt, startuproutes, tokenizer=tokenizer)

top = max(data, key=lambda x: x["score"])

#print(top["route"], top["score"])


# warm up whisper/ load the model for first run!

""" Fixes
 1. VAD (voice activity/inactivity detection):
    detects when voice stops and then stops recording so program can process stt
    - webrtcvad
    - Silero VAD
 2. Optimised whisper runtime
    - whisper.cpp
    - faster-whisper (i chose this one for better performance and simplicity and runs best on nvidia gpu which i have): https://pypi.org/project/faster-whisper/
 3. Using LFM2.5 router model
"""
# Homey Stuff

bearer = os.environ["BEARER"]

getdevices = requests.get("http://192.168.3.30/api/manager/devices/device", headers={"Authorization": f"Bearer {bearer}"})

devices = {}

for device in getdevices.json().values():
    if device['zone'] == '4cdb0219-bc77-41e8-8fbd-79acd670f01f':
        devices[device['name']] = device['id']
        
        
print(devices)

def onoffcontrol(devicetocontrol, statetoset):
    device_id = devices[devicetocontrol]
    
    if not device_id:
        return False
    
    response = requests.put(f"http://192.168.3.30/api/manager/devices/device/{device_id}/capability/onoff", headers={"Authorization": f"Bearer {bearer}"}, json={"value": statetoset})
    response.raise_for_status()
    return True



# PyAudio Stuff
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 1280
audio = pyaudio.PyAudio()
mic_stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)




#OpenWakeWord Stuff
MODEL_PATH = "Hey_Potato_20260228_130124.onnx"
WAKEWORD_KEY = "Hey_Potato_20260228_130124"

openwakemodel = Model(inference_framework='onnx', wakeword_models=[MODEL_PATH])


# Whisper Stuff
model = faster_whisper.WhisperModel("tiny.en", device="cpu", compute_type="int8")



# Main loop
while True:
    # Get audio
    audio_chunk = np.frombuffer(mic_stream.read(CHUNK), dtype=np.int16)

    # Feed to openWakeWord model
    prediction = openwakemodel.predict(audio_chunk)

    print(prediction)

    if prediction > 0.5: #type: ignore
        print("Wakeword detected!")
        
        command_cup = np.frombuffer(mic_stream.read(RATE * 3), dtype=np.int16).astype(np.float32) / 32768.0
        
        
        segments, _ = model.transcribe("audio.mp3")
        segments = list(segments)
        print(segments)


