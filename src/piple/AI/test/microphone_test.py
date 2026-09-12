import sounddevice as sd
import soundfile as sf

MICROPHONE_DEVICE = 0

DURATION = 5
SAMPLE_RATE = 44100
OUTPUT_FILE = "test_recording.wav"

print("Microphone test will start in 2 seconds...")
sd.sleep(2000)

print("Recording... Speak now!")

audio = sd.rec(
    int(DURATION * SAMPLE_RATE),
    samplerate=SAMPLE_RATE,
    channels=1,
    device=MICROPHONE_DEVICE
)

sd.wait()

print("Recording finished.")

sf.write(
    OUTPUT_FILE,
    audio,
    SAMPLE_RATE
)

print(f"Audio saved as: {OUTPUT_FILE}")