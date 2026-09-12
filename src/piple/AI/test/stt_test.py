from faster_whisper import WhisperModel

MODEL_SIZE = "base"

print("Loading Speech-to-Text model...")

model = WhisperModel(
    MODEL_SIZE,
    device="cpu",
    compute_type="int8"
)

print("Speech-to-Text model loaded.")

segments, info = model.transcribe(
    "test_recording.wav",
    beam_size=5
)

print(f"Detected language: {info.language}")
print(f"Language probability: {info.language_probability:.2f}")

print("\nTranscription:")

for segment in segments:
    print(segment.text)