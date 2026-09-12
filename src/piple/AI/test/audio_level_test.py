import sounddevice as sd
import numpy as np

MICROPHONE_DEVICE = 0

SAMPLE_RATE = 44100
CHUNK_SIZE = 1408

TEST_DURATION = 10

print("Get ready...")
sd.sleep(2000)

print("Listening...")

stream = sd.InputStream(
    samplerate=SAMPLE_RATE,
    channels=1,
    dtype="float32",
    device=MICROPHONE_DEVICE,
    blocksize=CHUNK_SIZE
)

stream.start()

total_chunks = int(
    TEST_DURATION * SAMPLE_RATE / CHUNK_SIZE
)

for _ in range(total_chunks):

    audio, overflowed = stream.read(CHUNK_SIZE)

    audio = np.squeeze(audio)

    rms = np.sqrt(np.mean(audio ** 2))
    peak = np.max(np.abs(audio))

    print(
        f"RMS: {rms:.4f} | Peak: {peak:.4f}"
    )

stream.stop()
stream.close()

print("\nAudio level test finished.")
