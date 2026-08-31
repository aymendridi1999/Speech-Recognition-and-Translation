from speech_toolkit import SpeechProcessor


processor = SpeechProcessor()
processor.process_speech(
    real_time=True,
    transcript_file="transcript.txt",
)
