from transformers import AutoModel, AutoTokenizer

model_id = "LiquidAI/LFM2.5-Encoder-350M-Prompt-Router"
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModel.from_pretrained(model_id, trust_remote_code=True).eval()
#model = model.to("cuda")

routes = ["Bedside Light On", "Bedside Light Off", "Desk Light Off", "Desk Light On", "Bedroom Light On", "Bedroom Light Off"]
prompt = "Turn my bedroom light on"

data = model.route(prompt, routes, tokenizer=tokenizer)

top = max(data, key=lambda x: x["score"])

print(top["route"], top["score"])
