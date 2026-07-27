import requests

bearer = "0c2479d6-5c65-4788-8df2-7d7120c57507:d86456a1-5d5d-4f86-bb66-1cea4b6fe6db:bafa9ee4b0916c496f9533db16631eb8078b4223"

result = requests.get("http://192.168.1.30/api/manager/devices/device", headers={"Authorization": f"Bearer {bearer}"})
zones = requests.get("http://192.168.1.30/api/manager/zones/zone", headers={"Authorization": f"Bearer {bearer}"})
    
zonemap = {}

for zone in zones.json().values():
    zonemap.update({zone['id']: zone['name']})

def mappedzone(zone_id):
    if zone_id in zonemap:
        return zonemap[zone_id]

for device in result.json().values():   
    print(device['name'], mappedzone(device['zone']))
    

