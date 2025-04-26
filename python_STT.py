import pyaudio
import wave
import numpy as np
import queue
import json
import os
import noisereduce as nr
from vosk import Model, KaldiRecognizer

# Audio settings
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 1024
RECORD_SECONDS = 5
RAW_FILENAME = "output.wav"
CLEAN_FILENAME = "clean_output.wav"

# LMS Noise Cancellation Class
class LMSFilter:
    def __init__(self, filter_order=32, mu=0.005):
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

# Vosk ASR Setup
if not os.path.exists("vosk-model-small-en-us-0.15"):
    print("Vosk model not found. Please download and unzip it in this directory.")
    exit()

model = Model("vosk-model-small-en-us-0.15")
recognizer = KaldiRecognizer(model, RATE)

# Audio Recording Function
def record_audio():
    print("🎙️ Recording... Speak now.")
    audio = pyaudio.PyAudio()
    stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

    frames = []
    for _ in range(int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)

    stream.stop_stream()
    stream.close()
    audio.terminate()

    with wave.open(RAW_FILENAME, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(audio.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))
    print("✅ Recording saved.")

# LMS Denoising Function
def lms_denoise(noisy_signal, noise_ref, mu=0.005, filter_len=32):
    N = len(noisy_signal)
    w = np.zeros(filter_len)
    x = np.zeros(filter_len)
    e = np.zeros(N)

    for n in range(filter_len, N):
        x = noise_ref[n-filter_len:n][::-1]
        y = np.dot(w, x)
        e[n] = noisy_signal[n] - y
        w += 2 * mu * e[n] * x

    return e

# Noise Cancellation and Denoising Function
def combined_denoise(input_file, output_file):
    print("🔇 Running LMS Noise Cancellation...")
    
    # Read the input audio file
    with wave.open(input_file, 'rb') as wf:
        rate = wf.getframerate()
        data = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)

    if len(data.shape) > 1:
        data = data[:, 0]
    data = data.astype(np.float32)

    # Generate noise (replace with real noise reference if available)
    noise = np.random.normal(0, 0.5, len(data))
    lms_output = lms_denoise(data, noise)

    # Reduce noise with Noisereduce
    reduced_noise = nr.reduce_noise(y=lms_output, sr=rate)

    # Save cleaned audio
    cleaned = np.clip(reduced_noise, -32768, 32767).astype(np.int16)
    with wave.open(output_file, 'wb') as wf:
        wf.setnchannels(1)  # Mono channel
        wf.setsampwidth(2)  # 16-bit samples (2 bytes per sample)
        wf.setframerate(rate)
        wf.writeframes(cleaned.tobytes())
    
    print("✅ Denoised file saved:", output_file)

# Speech-to-Text with Vosk
def transcribe_vosk(audio_file):
    print("🧠 Transcribing using Vosk...")
    with wave.open(audio_file, "rb") as wf:
        audio_data = wf.readframes(wf.getnframes())

    if recognizer.AcceptWaveform(audio_data):
        result = json.loads(recognizer.Result())
        if result.get("text"):
            print("🗣️ You said:", result["text"])
        else:
            print("No speech recognized.")
    else:
        print("Partial recognition:", recognizer.PartialResult())

# Main Execution
if __name__ == "__main__":
    record_audio()  # Step 1: Record Audio
    combined_denoise(RAW_FILENAME, CLEAN_FILENAME)  # Step 2: Apply Noise Cancellation
    transcribe_vosk(CLEAN_FILENAME)  # Step 3: Transcribe using Vosk
