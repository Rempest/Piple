
import io
import wave
import os
 
import sounddevice as sd
import numpy as np
import torch
import whisper
import ollama
from scipy.signal import resample_poly
from scipy.io.wavfile import write
from silero_vad import load_silero_vad
from piper import PiperVoice, SynthesisConfig
 
 
# ==========================================
# PIPLE VOICE CONFIGURATION
# ==========================================
 
MICROPHONE_DEVICE = 3
MIC_SAMPLE_RATE = 44100
VAD_SAMPLE_RATE = 16000
CHANNELS = 1
CHUNK_DURATION = 0.032
MIC_CHUNK_SIZE = int(
    MIC_SAMPLE_RATE * CHUNK_DURATION
)
 
VAD_CHUNK_SIZE = 512
 
 
# ==========================================
# VOICE ACTIVITY DETECTION
# ==========================================
 
SPEECH_THRESHOLD = 0.6
SILENCE_DURATION = 1.0
 
# Recordings shorter than this are almost always
# noise/clicks caught by the VAD, not real speech.
MIN_SPEECH_DURATION = 0.5  # seconds
 
 
# ==========================================
# WHISPER CONFIGURATION
# ==========================================
 
WHISPER_PROMPT = (
    "The user is talking to a friendly robot named Piple. "
    "Piple is the name of the robot."
)
 
 
# ==========================================
# PIPLE PERSONALITY
# ==========================================
 
SYSTEM_PROMPT = (
    "You are Piple, a friendly robot assistant. "
    "Respond in the same language as the user. "
    "Be helpful, natural, and easy to understand. "
    "Keep your answers concise unless the user asks for details."
)
 
 
# ==========================================
# TEXT-TO-SPEECH (Piper) CONFIGURATION
# ==========================================
 
# Folder where you downloaded the .onnx voice models, e.g.:
#   python -m piper.download_voices --data-dir ./voices en_US-ryan-high
#   python -m piper.download_voices --data-dir ./voices ru_RU-irina-medium
VOICES_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "voices"
)
 
# Whisper language code -> Piper voice model filename (no .onnx).
# Pick soft/friendly-sounding voices here. Add more languages as needed.
VOICE_MODELS = {
    "en": "en_US-ryan-high",
    "ru": "ru_RU-irina-medium",
}
 
# Voice used when the detected language has no entry in VOICE_MODELS.
DEFAULT_VOICE_LANG = "en"
 
# Tweak these to shape the personality of Piple's voice.
SYNTHESIS_CONFIG = SynthesisConfig(
    volume=1.0,
    length_scale=0.95,   # slightly faster than default = more energetic
    noise_scale=0.667,   # natural variation in the voice
    noise_w_scale=0.9,   # more lively/expressive speaking style
    normalize_audio=True,
)
 
_voice_cache = {}
 
 
def get_voice(language):
    """
    Load (and cache) the Piper voice model for a given language code.
    Falls back to DEFAULT_VOICE_LANG if the requested language isn't
    configured or its model file is missing.
    """
    model_name = VOICE_MODELS.get(language)
 
    if model_name is None:
        model_name = VOICE_MODELS[DEFAULT_VOICE_LANG]
 
    if model_name in _voice_cache:
        return _voice_cache[model_name]
 
    model_path = os.path.join(VOICES_DIR, f"{model_name}.onnx")
    config_path = model_path + ".json"
 
    if not (os.path.exists(model_path) and os.path.exists(config_path)):
        default_model_name = VOICE_MODELS[DEFAULT_VOICE_LANG]
 
        # Only attempt a fallback if it's actually a different voice
        # than the one that just failed, to avoid a pointless loop.
        if model_name != default_model_name:
            print(
                f"⚠️ Voice model '{model_name}' not found in {VOICES_DIR}. "
                f"Falling back to '{default_model_name}'."
            )
            return get_voice(DEFAULT_VOICE_LANG)
 
        raise FileNotFoundError(
            f"Voice model '{model_name}' not found in {VOICES_DIR}.\n"
            f"Download it with:\n"
            f"  python -m piper.download_voices --data-dir {VOICES_DIR} {model_name}"
        )
 
    voice = PiperVoice.load(model_path)
    _voice_cache[model_name] = voice
    return voice
 
 
# ==========================================
# LOAD MODELS
# ==========================================
 
print("=================================")
print("🤖 Starting Piple Voice System")
print("=================================\n")
print("Loading Voice Activity Detection...")
 
vad_model = load_silero_vad()
 
print("VAD loaded.")
print("\nLoading Whisper...")
 
whisper_model = whisper.load_model("base")
 
print("Whisper loaded.")
print("\nLoading Text-to-Speech voice...")
 
get_voice(DEFAULT_VOICE_LANG)
 
print("Text-to-Speech ready.")
 
 
# ==========================================
# PIPLE CONVERSATION MEMORY
# ==========================================
 
messages = [
    {
        "role": "system",
        "content": SYSTEM_PROMPT
    }
]
 
 
# ==========================================
# LISTEN FUNCTION
# ==========================================
 
def listen():
    print("\n🎤 Listening...")
    recording = []
    is_speaking = False
    silence_time = 0.0
 
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
 
            audio, overflowed = stream.read(
                MIC_CHUNK_SIZE
            )
 
            audio = np.squeeze(audio)
 
 
            # ----------------------------------
            # Convert 44100 Hz → 16000 Hz
            # ----------------------------------
 
            audio_16k = resample_poly(
                audio,
                VAD_SAMPLE_RATE,
                MIC_SAMPLE_RATE
            )
 
 
            # ----------------------------------
            # Make sure VAD gets 512 samples
            # ----------------------------------
 
            if len(audio_16k) < VAD_CHUNK_SIZE:
                continue
            audio_16k = audio_16k[:VAD_CHUNK_SIZE]
            audio_tensor = torch.from_numpy(
                audio_16k.astype(np.float32)
            )
 
 
            # ----------------------------------
            # Voice probability
            # ----------------------------------
 
            speech_probability = vad_model(
                audio_tensor,
                VAD_SAMPLE_RATE
            ).item()
 
 
            # ==================================
            # SPEECH STARTED
            # ==================================
 
            if speech_probability > SPEECH_THRESHOLD:
                if not is_speaking:
                    print("🗣️ User is speaking...")
                    is_speaking = True
                    silence_time = 0.0
                recording.append(audio)
                silence_time = 0.0
 
 
            # ==================================
            # SILENCE
            # ==================================
 
            else:
                if is_speaking:
                    recording.append(audio)
                    silence_time += CHUNK_DURATION
                    if silence_time >= SILENCE_DURATION:
                        print("✓ Speech finished.")
                        break
 
 
    finally:
 
        stream.stop()
        stream.close()
 
 
    # ==========================================
    # VALIDATE RECORDING LENGTH
    # ==========================================
 
    if len(recording) == 0:
        return None
 
    audio_data = np.concatenate(recording)
 
    duration = len(audio_data) / MIC_SAMPLE_RATE
 
    if duration < MIN_SPEECH_DURATION:
        print(
            f"⚠️ Recording too short ({duration:.2f}s), "
            "likely noise. Ignoring."
        )
        return None
 
 
    # ==========================================
    # SAVE AUDIO
    # ==========================================
 
    audio_int16 = (
        audio_data * 32767
    ).astype(np.int16)
 
 
    filename = "piple_voice.wav"
 
 
    write(
        filename,
        MIC_SAMPLE_RATE,
        audio_int16
    )
 
 
    return filename
 
 
# ==========================================
# WHISPER FUNCTION
# ==========================================
 
def understand(filename):
    print("\n🧠 Piple is listening to the audio...")
 
    result = whisper_model.transcribe(
        filename,
        fp16=False,
        initial_prompt=WHISPER_PROMPT,
        condition_on_previous_text=False,
        no_speech_threshold=0.6,
        logprob_threshold=-1.0,
    )
 
    text = result["text"].strip()
    language = result.get("language", DEFAULT_VOICE_LANG)
 
    # ----------------------------------
    # Filter out "no speech" segments
    # ----------------------------------
    # Whisper marks segments it doesn't think are real
    # speech with a high no_speech_prob. If every segment
    # is flagged, treat the whole thing as silence.
    segments = result.get("segments", [])
    if segments and all(
        seg.get("no_speech_prob", 0) > 0.6 for seg in segments
    ):
        return "", language
 
    # ----------------------------------
    # Filter out prompt echo
    # ----------------------------------
    # On weak/empty audio Whisper sometimes just repeats
    # back the initial_prompt instead of transcribing.
    normalized = text.lower().strip(" .!?")
    prompt_normalized = WHISPER_PROMPT.lower().strip(" .!?")
    if normalized in prompt_normalized or prompt_normalized in normalized:
        return "", language
 
    return text, language
 
 
# ==========================================
# TEXT-TO-SPEECH FUNCTION
# ==========================================
 
def speak(text, language=DEFAULT_VOICE_LANG):
    if not text:
        return
 
    voice = get_voice(language)
 
    # Synthesize into an in-memory WAV buffer (no temp files needed).
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        voice.synthesize_wav(text, wav_file, syn_config=SYNTHESIS_CONFIG)
 
    buffer.seek(0)
    with wave.open(buffer, "rb") as wav_file:
        sample_rate = wav_file.getframerate()
        raw_audio = wav_file.readframes(wav_file.getnframes())
 
    audio = np.frombuffer(raw_audio, dtype=np.int16).astype(np.float32) / 32768.0
 
    sd.play(audio, sample_rate)
    sd.wait()
 
 
# ==========================================
# PIPLE BRAIN
# ==========================================
 
def think(user_text):
    messages.append(
        {
            "role": "user",
            "content": user_text
        }
    )
 
 
    print("\n🤖 Piple is thinking...")
    response = ollama.chat(
        model="qwen3-4b-local",
        messages=messages
    )
 
 
    answer = response["message"]["content"]
    messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )
 
 
    return answer
 
 
# ==========================================
# MAIN LOOP
# ==========================================
 
print("\n=================================")
print("🤖 Piple is ready!")
print("Speak to Piple.")
print("Press Ctrl+C to stop.")
print("=================================")
 
 
try:
 
    while True:
        # 1. Listen
        audio_file = listen()
        if audio_file is None:
            continue
 
 
        # 2. Speech → Text
        user_text, detected_language = understand(
            audio_file
        )
 
 
        if len(user_text) == 0:
            print("\n(no clear speech detected, listening again)")
            continue
 
 
        print("\n👤 You:")
        print(user_text)
 
 
        # 3. Text → AI
        piple_answer = think(
            user_text
        )
 
 
        print("\n🤖 Piple:")
        print(piple_answer)
 
 
        # 4. AI → Speech
        speak(
            piple_answer,
            language=detected_language
        )
 
 
except KeyboardInterrupt:
    print("\n\n👋 Piple Voice System stopped.")
