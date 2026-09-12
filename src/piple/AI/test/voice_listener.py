import sounddevice as sd
import numpy as np
import torch
from scipy.signal import resample_poly
from silero_vad import load_silero_vad
from scipy.io.wavfile import write


# ==============================
# Audio configuration
# ==============================

MICROPHONE_DEVICE = 0

MIC_SAMPLE_RATE = 44100
VAD_SAMPLE_RATE = 16000

CHANNELS = 1

CHUNK_DURATION = 0.032

MIC_CHUNK_SIZE = int(
    MIC_SAMPLE_RATE * CHUNK_DURATION
)

VAD_CHUNK_SIZE = 512


# ==============================
# Voice detection configuration
# ==============================

SPEECH_THRESHOLD = 0.5

SILENCE_DURATION = 1.0

MAX_RECORDING_DURATION = 15


# ==============================
# Load VAD model
# ==============================

print("Loading Voice Activity Detection model...")

model = load_silero_vad()

print("VAD model loaded.")
print("\nPiple Voice Listener is ready.")
print("Start speaking...")


# ==============================
# Audio buffers
# ==============================

recording = []

is_speaking = False
silence_time = 0.0


# ==============================
# Open microphone
# ==============================

stream = sd.InputStream(
    samplerate=MIC_SAMPLE_RATE,
    channels=CHANNELS,
    dtype="float32",
    device=MICROPHONE_DEVICE,
    blocksize=MIC_CHUNK_SIZE
)

stream.start()


try:

    while True:

        # Read microphone audio
        audio, overflowed = stream.read(
            MIC_CHUNK_SIZE
        )

        audio = np.squeeze(audio)

        # Convert 44100 Hz -> 16000 Hz
        audio_16k = resample_poly(
            audio,
            VAD_SAMPLE_RATE,
            MIC_SAMPLE_RATE
        )

        # Make sure VAD receives exactly 512 samples
        if len(audio_16k) < VAD_CHUNK_SIZE:
            continue

        audio_16k = audio_16k[:VAD_CHUNK_SIZE]

        audio_tensor = torch.from_numpy(
            audio_16k.astype(np.float32)
        )

        # Get speech probability
        speech_probability = model(
            audio_tensor,
            VAD_SAMPLE_RATE
        ).item()


        # ==============================
        # Speech started
        # ==============================

        if speech_probability > SPEECH_THRESHOLD:

            if not is_speaking:

                print("\n>>> SPEECH STARTED")

                is_speaking = True
                silence_time = 0.0

            recording.append(audio)

            silence_time = 0.0


        # ==============================
        # Silence
        # ==============================

        else:

            if is_speaking:

                recording.append(audio)

                silence_time += CHUNK_DURATION

                # User finished speaking
                if silence_time >= SILENCE_DURATION:

                    print(">>> SPEECH FINISHED")

                    break


except KeyboardInterrupt:

    print("\nStopping...")


# ==============================
# Close microphone
# ==============================

stream.stop()
stream.close()


# ==============================
# Save recording
# ==============================

if len(recording) > 0:

    audio_data = np.concatenate(recording)

    audio_int16 = (
        audio_data * 32767
    ).astype(np.int16)

    write(
        "piple_voice.wav",
        MIC_SAMPLE_RATE,
        audio_int16
    )

    print("\nAudio saved:")
    print("piple_voice.wav")

else:

    print("\nNo speech detected.")