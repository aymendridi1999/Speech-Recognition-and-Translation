# Speech Recognition & Translation Toolkit

A compact Python project that combines speech-to-text, translation, text-to-speech, real-time transcription, and basic audio-frequency analysis in a single reusable toolkit.

## Features

- Microphone speech-to-text using Google Speech Recognition
- Ambient-noise calibration before microphone capture
- Continuous real-time transcription with timestamps
- Transcript export to UTF-8 text files
- Audio-file transcription through `SpeechRecognition`
- Translation to French, Spanish, and German with automatic source-language detection
- MP3 text-to-speech generation with gTTS
- WAV waveform visualization
- Bass-frequency analysis in a configurable range (20-250 Hz by default)
- Bass spectrogram and average bass magnitude calculation
- Lazy microphone initialization so non-microphone features do not require PyAudio at import time

## Tech stack

- Python 3.10+
- SpeechRecognition
- deep-translator
- gTTS
- NumPy
- SciPy
- Matplotlib
- PyAudio for microphone input
- pytest
- GitHub Actions

## Architecture

```text
Microphone / audio file
        |
        v
SpeechRecognition
        |
        +--> transcript / text export
        |
        +--> translation --> optional MP3 output
        |
        +--> WAV analysis --> waveform / bass spectrogram
```

The custom application code lives in `speech_toolkit/`. Third-party speech-recognition code is installed as a dependency and is not vendored in this repository.

## Requirements

- Python 3.10+
- Internet access for Google speech recognition, translation, and gTTS
- PyAudio only for microphone-based features

Audio-file processing, translation, export, and analysis can be imported without initializing a microphone.

## Installation

Create and activate a virtual environment, then install the project:

```bash
python -m venv .venv
pip install -e .
```

For microphone support:

```bash
pip install -e ".[audio]"
```

PyAudio may require the PortAudio development package on Linux.

## Quick start

### One-shot microphone recognition and translation

```python
from speech_toolkit import SpeechProcessor

processor = SpeechProcessor()
result = processor.process_speech(
    target_lang="fr",
    export_file="output.txt",
    translation_audio_file="output.mp3",
)

print(result)
```

### Real-time transcription

```python
from speech_toolkit import SpeechProcessor

processor = SpeechProcessor()
processor.process_speech(
    real_time=True,
    transcript_file="transcript.txt",
)
```

Press `Ctrl+C` to stop. The complete timestamped transcript is written to `transcript.txt`.

### Process an audio file

```python
from speech_toolkit import SpeechProcessor

processor = SpeechProcessor()
result = processor.process_audio_file(
    "sample.wav",
    target_lang="es",
    visualize=True,
    visualize_bass=True,
)

print(result)
```

Waveform and bass visualization are intended for WAV input files.

## Project structure

```text
.
├── speech_toolkit/
│   ├── __init__.py
│   └── processor.py
├── examples/
│   ├── basic.py
│   ├── realtime.py
│   └── audio_file.py
├── tests/
│   └── test_processor.py
├── .github/workflows/ci.yml
├── pyproject.toml
├── README.md
├── LICENSE
└── .gitignore
```

## Design notes

The project intentionally stays small. `SpeechProcessor` provides the application-level behavior while established libraries handle speech recognition, translation, text-to-speech, plotting, and signal processing.

The microphone is initialized lazily, cloud-backed functionality is isolated behind small methods, and external calls are mocked in the unit tests so the core behavior can be tested without requiring live services.

## Security

No personal API keys are stored in the repository. Local `.env` files, common credential files, private keys, and generated transcript/audio outputs are ignored by Git.

The default Google recognizer used by `SpeechRecognition` does not require a personal Google API key to be stored in this project.

## Limitations

- Cloud-backed recognition, translation, and text-to-speech require an internet connection.
- The default recognizer is Google Speech Recognition.
- Generated MP3 files are not automatically opened, keeping behavior consistent across operating systems.
- Frequency visualization is designed for WAV files.

## Testing

Install development dependencies and run:

```bash
pip install -e ".[dev]"
pytest
```

GitHub Actions runs the test suite on Python 3.10, 3.11, 3.12, and 3.13 for pushes and pull requests targeting `main`.

## Acknowledgments

This project uses the open-source [SpeechRecognition](https://github.com/Uberi/speech_recognition) library as its speech-recognition foundation, along with deep-translator, gTTS, NumPy, SciPy, and Matplotlib.

## License

MIT. See `LICENSE`.
