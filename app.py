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

# LFM Routing model setup + warmup + functions
model_id = "LiquidAI/LFM2.5-Encoder-350M-Prompt-Router"
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
routingmodel = AutoModel.from_pretrained(model_id, trust_remote_code=True).eval()
routingmodel = routingmodel.to("mps")


def warmup():
    startuproutes = ["Bedside Light On", "Bedside Light Off", "Desk Light Off", "Desk Light On", "Bedroom Light On", "Bedroom Light Off"]
    startprompt = "warmup"
    data = routingmodel.route(startprompt, startuproutes, tokenizer=tokenizer)
    top = max(data, key=lambda x: x["score"])

warmup()

def best_device(userinput, device_routes):
    data = routingmodel.route(userinput, device_routes, tokenizer=tokenizer)
    top = max(data, key=lambda x: x["score"])
    return top["route"], top["score"]

def on_off(userinput):
    routes = ["Turn device on", "Turn device off"]
    data = routingmodel.route(userinput, routes, tokenizer=tokenizer)        
    top = max(data, key=lambda x: x["score"])
    if top["route"] == routes[0]:
        return True
    else:
        return False


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
def get_room_devices(zoneid, bearer):
    getdevices = requests.get("http://192.168.3.30/api/manager/devices/device", headers={"Authorization": f"Bearer {bearer}"})

    devices = {}

    for device in getdevices.json().values():
        if device['zone'] == zoneid:
            devices[device['name']] = device['id']
            
    return devices
            

def onoffcontrol(devicetocontrol, statetoset, deviceslist):
    device_id = deviceslist[devicetocontrol]
    
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

def main():
    global bearer
    bearer = os.environ["BEARER"]
    zoneid = "4cdb0219-bc77-41e8-8fbd-79acd670f01f"

    devices = get_room_devices(zoneid, bearer)
    
    
    
    
    

    # Main loop
    while True:
        # Get audio
        audio_chunk = np.frombuffer(mic_stream.read(CHUNK), dtype=np.int16)

        # Feed to openWakeWord model
        prediction = openwakemodel.predict(audio_chunk)

        print(prediction)

        if prediction[WAKEWORD_KEY] > 0.5: #type: ignore
            print("Wakeword detected!")
            
            command_cup = np.frombuffer(mic_stream.read(RATE * 3), dtype=np.int16).astype(np.float32) / 32768.0
            
            
            segments, _ = model.transcribe(command_cup)
            user = " ".join(s.text for s in segments).strip()
            
            print(user)
            
            onoffcontrol(best_device(user, devices)[0], on_off(user), list(devices))
            
            



main()