import whisper
import faster_whisper
import time


fastmodel = faster_whisper.WhisperModel("tiny.en", device="cpu", compute_type="int8")
model = whisper.load_model("tiny.en")


start = time.time()
segments, _ = fastmodel.transcribe("learning_files/audio.mp3")
fasterresult = list(segments)
print(fasterresult)
print(time.time() - start)

start = time.time()
result = model.transcribe("learning_files/audio.mp3")
print(result)
print(time.time() - start)


print("".join(segment.text for segment in fasterresult))
print(result["text"])