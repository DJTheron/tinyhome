import whisper

model = whisper.load_model("tiny.en")
result = model.transcribe("tinyhome/audio.mp3")
print(result["text"])