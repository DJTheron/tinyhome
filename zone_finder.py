import requests
import os
from dotenv import load_dotenv

load_dotenv()

bearer = os.environ["BEARER"]

getzones = requests.get("http://192.168.3.30/api/manager/zones/zone", headers={"Authorization": f"Bearer {bearer}"})



while True:
    user_zone = str(input("What is the zone/room you wish to find? (type the name exactly as it is on the app, copy and paste is best)\n>>>"))
    for zone in getzones.json().values():
        if zone['name'].lower() == user_zone.lower():
            print(f"Zone {user_zone} UUID is: {zone['id']}\n")
