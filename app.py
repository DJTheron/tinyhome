from ollama import chat
import pyaudio
import numpy as np
from openwakeword.model import Model
import requests
import whisper
import re


# Homey Stuff
bearer = "0c2479d6-5c65-4788-8df2-7d7120c57507:6a53f79c-6654-41b8-a449-063e6f4fa18f:2aa82066efc1f34d0cef6ae8f51e7945396b512a"

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

owwModel = Model(inference_framework='onnx', wakeword_models=[MODEL_PATH])




# Whisper Stuff
model = whisper.load_model("tiny.en")




# Main loop
while True:
    # Get audio
    audio_chunk = np.frombuffer(mic_stream.read(CHUNK), dtype=np.int16)

    # Feed to openWakeWord model
    prediction = owwModel.predict(audio_chunk)[WAKEWORD_KEY]  # type: ignore[reportCallIssue]

    print(prediction)

    if prediction > 0.5:
        print("Wakeword detected!")
        
        command_cup = np.frombuffer(mic_stream.read(RATE * 3), dtype=np.int16).astype(np.float32) / 32768.0
        
        
        result = model.transcribe(command_cup, fp16=False)["text"]
        print(result)

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

