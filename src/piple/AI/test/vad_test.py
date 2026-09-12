import sounddevice as sd
import numpy as np
import torch
from scipy.signal import resample_poly
from silero_vad import load_silero_vad


MICROPHONE_DEVICE = 0

MIC_SAMPLE_RATE = 44100
VAD_SAMPLE_RATE = 16000

MIC_CHUNK_SIZE = 1408
VAD_CHUNK_SIZE = 512

TEST_DURATION = 10

SPEECH_THRESHOLD = 0.5


print("Loading Voice Activity Detection model...")

model = load_silero_vad()
model.reset_states()

print("VAD model loaded.")
print("\nGet ready...")

sd.sleep(2000)

print("Listening...")

stream = sd.InputStream(
    samplerate=MIC_SAMPLE_RATE,
    channels=1,
    dtype="float32",
    device=MICROPHONE_DEVICE,
    blocksize=MIC_CHUNK_SIZE
)

stream.start()

audio_buffer = np.array([], dtype=np.float32)

speech_active = False

total_chunks = int(
    TEST_DURATION * MIC_SAMPLE_RATE / MIC_CHUNK_SIZE
)

for _ in range(total_chunks):

    audio, overflowed = stream.read(MIC_CHUNK_SIZE)

    audio = np.squeeze(audio)

    audio_16k = resample_poly(
        audio,
        VAD_SAMPLE_RATE,
        MIC_SAMPLE_RATE
    ).astype(np.float32)

    audio_buffer = np.concatenate(
        (audio_buffer, audio_16k)
    )

    while len(audio_buffer) >= VAD_CHUNK_SIZE:

        chunk = audio_buffer[:VAD_CHUNK_SIZE]

        audio_buffer = audio_buffer[VAD_CHUNK_SIZE:]

        audio_tensor = torch.from_numpy(chunk)

        speech_probability = model(
            audio_tensor,
            VAD_SAMPLE_RATE
        ).item()

        print(
            f"Speech probability: "
            f"{speech_probability:.3f}"
        )

        if speech_probability >= SPEECH_THRESHOLD:

            if not speech_active:

                speech_active = True

                print(">>> SPEECH STARTED")

        else:

            if speech_active:

                speech_active = False

                print(">>> SPEECH ENDED")


stream.stop()
stream.close()

print("\nVAD test finished.")