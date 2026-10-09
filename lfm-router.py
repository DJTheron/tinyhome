from transformers import AutoModel, AutoTokenizer
import time

model_id = "LiquidAI/LFM2.5-Encoder-350M-Prompt-Router"
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModel.from_pretrained(model_id, trust_remote_code=True).eval()
model = model.to("mps")

def best_device(userinput, device_routes):
    data = model.route(userinput, device_routes, tokenizer=tokenizer)
    top = max(data, key=lambda x: x["score"])
    return top["route"], top["score"]

def on_off(userinput):
    routes = ["Turn device on", "Turn device off"]
    data = model.route(userinput, routes, tokenizer=tokenizer)        
    top = max(data, key=lambda x: x["score"])
    if top["route"] == routes[0]:
        return True
    else:
        return False

def warmup():
    warmupp = "Hello"
    on_off(warmupp)
    
warmup()


user = "bedside deactivate"
devices = ["Bedside light", "Desk Light", "bedroom light/ceiling light"]

start = time.perf_counter()
print(best_device(user, devices), on_off(user))
user = "bathroom light off"
print(best_device(user, devices), on_off(user))
print("Done in:", time.perf_counter() - start, "seconds")
