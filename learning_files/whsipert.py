import whisper

model = whisper.load_model("tiny.en")
result = model.transcribe("learning_files/audio.mp3")
print(result["text"])