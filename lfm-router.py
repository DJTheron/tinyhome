from transformers import AutoModel, AutoTokenizer

model_id = "LiquidAI/LFM2.5-Encoder-350M-Prompt-Router"
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModel.from_pretrained(model_id, trust_remote_code=True).eval()
#model = model.to("cuda")

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

user = "bedside deactivate"
devices = ["Bedside light", "Desk Light", "room light/ceiling light"]
print(best_device(user, devices), on_off(user))
