from speech_toolkit import SpeechProcessor


processor = SpeechProcessor()
result = processor.process_audio_file(
    "sample.wav",
    target_lang="es",
    export_file="output.txt",
    visualize=True,
    visualize_bass=True,
)

print(result)
