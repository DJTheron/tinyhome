
import requests
import os
from dotenv import load_dotenv

load_dotenv()

bearer = os.environ["BEARER"]

getdevices = requests.get("http://192.168.3.30/api/manager/devices/device", headers={"Authorization": f"Bearer {bearer}"})

devices = []

for device in getdevices.json().values():
    if device['zone'] == '4cdb0219-bc77-41e8-8fbd-79acd670f01f':
        devices.append((device['name']))
        
        
print(devices)