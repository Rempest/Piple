import whisper

print("Loading Whisper model...")

model = whisper.load_model("base")

print("Whisper model loaded!")
print("Transcribing piple_voice.wav...\n")

result = model.transcribe(
    "piple_voice.wav",
    fp16=False,

    language="en",

    initial_prompt=(
        "The user is talking to a friendly robot named Piple. "
        "Piple is the name of the robot. "
        "Common phrases include: Hello Piple, Hey Piple, "
        "Piple, who are you?"
    )
)

print("Piple heard:")
print(result["text"])