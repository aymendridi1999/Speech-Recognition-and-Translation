from speech_toolkit import SpeechProcessor


processor = SpeechProcessor()
result = processor.process_speech(
    target_lang="fr",
    export_file="output.txt",
    translation_audio_file="output.mp3",
)

if result and result["text"]:
    print("Done.")
