# Copyright 2022 David Scripka. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Imports
import pyaudio
import numpy as np
from openwakeword.model import Model
import time

cooldown = 1.5
last_detection_time = 0.0


# Get microphone stream
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 1280
audio = pyaudio.PyAudio()
mic_stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)


MODEL_PATH = "Hey_Potato_20260228_130124.onnx"
WAKEWORD_KEY = "Hey_Potato_20260228_130124"

owwModel = Model(inference_framework='onnx', wakeword_models=[MODEL_PATH])

n_models = len(owwModel.models.keys())

# Run capture loop continuously, checking for wakewords

while True:
    # Get audio
    audio_chunk = np.frombuffer(mic_stream.read(CHUNK), dtype=np.int16)

    # Feed to openWakeWord model
    prediction = owwModel.predict(audio_chunk)[WAKEWORD_KEY]  # type: ignore[reportCallIssue]

    current_time = time.time()

    if prediction > 0.5:
        if current_time - last_detection_time >= cooldown:
            print("Wakeword detected!")

            last_detection_time = current_time