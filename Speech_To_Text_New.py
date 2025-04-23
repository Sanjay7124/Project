import sounddevice as sd
import numpy as np
from vosk import Model, KaldiRecognizer
import queue
import json
import os

# ====== LMS Algorithm Class ======
class LMSFilter:
    def __init__(self, filter_order=32, mu=0.01):
        self.filter_order = filter_order
        self.mu = mu
        self.weights = np.zeros(filter_order)
        self.x = np.zeros(filter_order)

    def filter(self, input_signal):
        output = []
        for n in range(len(input_signal)):
            self.x[1:] = self.x[:-1]
            self.x[0] = input_signal[n]
            y = np.dot(self.weights, self.x)
            e = input_signal[n] - y
            self.weights += 2 * self.mu * e * self.x
            output.append(e)
        return np.array(output)

# ====== Audio Input Setup =====
q = queue.Queue()

def audio_callback(indata, frames, time, status):
    if status:
        print(status)
    q.put(indata.copy())

# ====== Initialize Vosk Model ======
if not os.path.exists("vosk-model-small-en-us-0.15"):
    print("Vosk model not found. Please download and unzip it in this directory.")
    exit()

model = Model("vosk-model-small-en-us-0.15")
recognizer = KaldiRecognizer(model, 16000)

# ====== Main Function ======
def main():
    print("Starting Speech-to-Text with LMS Noise Cancellation...")
    lms = LMSFilter(filter_order=32, mu=0.005)  # You can tune mu and order for performance

    with sd.InputStream(channels=1, samplerate=16000, callback=audio_callback):
        try:
            while True:
                audio_data = q.get()
                signal = audio_data[:, 0]

                # Apply LMS Noise Cancellation
                filtered_signal = lms.filter(signal)

                # Convert to 16-bit PCM format for recognizer
                pcm_data = (filtered_signal * 32767).astype(np.int16).tobytes()

                if recognizer.AcceptWaveform(pcm_data):
                    result = json.loads(recognizer.Result())
                    if result.get("text"):
                        print("You said:", result["text"])
                else:
                    partial = json.loads(recognizer.PartialResult())
                    print("Listening:", partial.get("partial", ""), end="\r")

        except KeyboardInterrupt:
            print("\nStopped by user.")

# ====== Run Program ======
if __name__ == "__main__":
    main()
