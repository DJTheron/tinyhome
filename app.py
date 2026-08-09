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
 3. other lookup method not LLM for faster requests but with LLM as fallback
"""


# Homey Stuff

bearer = os.environ["BEARER"]

getdevices = requests.get("http://192.168.3.30/api/manager/devices/device", headers={"Authorization": f"Bearer {bearer}"})
getzones = requests.get("http://192.168.3.30/api/manager/zones/zone", headers={"Authorization": f"Bearer {bearer}"})
    
zonemap = {}
devices = []

if "error" in getzones.json():
    print("Zones API error:", getzones.json()["error"], "-", getzones.json().get("error_description", ""))
    exit(1)
if "error" in getdevices.json():
    print("Devices API error:", getdevices.json()["error"], "-", getdevices.json().get("error_description", ""))
    exit(1)

for zone in getzones.json().values():
    zonemap.update({zone['id']: zone['name']})

def mappedzone(zone_id):
    if zone_id in zonemap:
        return zonemap[zone_id]

def list_devices():
    for device in getdevices.json().values():   
        devices.append((device['name'].lower(), mappedzone(device['zone'])))
    return devices

def devicemap(prettyname):
    for device_id, device in getdevices.json().items():
        if device['name'].lower() == prettyname.lower():
            return device_id

def onoffcontrol(devicetocontrol, statetoset):
    device_id = devicemap(devicetocontrol)
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

    if prediction > 0.5:
        print("Wakeword detected!")
        
        command_cup = np.frombuffer(mic_stream.read(RATE * 3), dtype=np.int16).astype(np.float32) / 32768.0
        
        
        segments, _ = model.transcribe("audio.mp3")
        segments = list(segments)
        print(segments)

        devices.clear()
        userprompt = "These are the list of available devices in the house: " + str(list_devices()) + "You must control them in a certain way, the current functionality is just turning on and off by including the command onoffcontrol(thedevicesname, thestateyouwanttosetittoeithertrueorfalse) in plain text, not as code." + "This is the users request to you from Text To Speach: " + str(result)

        response = chat(
            model='gemma3:270m',
            messages=[{'role': 'user', 'content': userprompt}],
        )

        airesponse = str(response.message.content)
        print(airesponse)

        matches = re.findall(r'onoffcontrol\(["\']?([^"\']+?)["\']?\s*,\s*(True|False)\)', airesponse, re.IGNORECASE)
        for match in matches:
            devicetocontrol = match[0].strip()
            state = match[1].lower() == "true"
            onoffcontrol(devicetocontrol, state)

