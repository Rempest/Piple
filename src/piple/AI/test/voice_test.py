import sounddevice as sd
from scipy.io.wavfile import write
from faster_whisper import WhisperModel

MICROPHONE_DEVICE = 0

DURATION = 5
SAMPLE_RATE = 44100
OUTPUT_FILE = "voice_input.wav"

MODEL_SIZE = "base"

print("Loading Speech-to-Text model...")

model = WhisperModel(
    MODEL_SIZE,
    device="cpu",
    compute_type="int8"
)

print("Speech-to-Text model loaded.")

print("\nGet ready...")
sd.sleep(2000)

print("🎤 Speak now!")

audio = sd.rec(
    int(DURATION * SAMPLE_RATE),
    samplerate=SAMPLE_RATE,
    channels=1,
    device=MICROPHONE_DEVICE
)

sd.wait()

print("Recording finished.")

write(
    OUTPUT_FILE,
    SAMPLE_RATE,
    audio
)

print("Transcribing...")

segments, info = model.transcribe(
    OUTPUT_FILE,
    beam_size=5
)

print(f"Detected language: {info.language}")
print(f"Language probability: {info.language_probability:.2f}")

print("\nPiple heard:")

for segment in segments:
    print(segment.text)