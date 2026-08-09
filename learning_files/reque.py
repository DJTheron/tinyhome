import requests

bearer = "" # expired token deleted

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
    

