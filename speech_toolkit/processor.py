from __future__ import annotations

import queue
import tempfile
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import speech_recognition as sr


class SpeechProcessor:
    """Combine speech recognition, translation, transcription, and audio analysis."""

    SUPPORTED_LANGUAGES = {
        "fr": "French",
        "es": "Spanish",
        "de": "German",
    }

    def __init__(self, recognizer: sr.Recognizer | None = None) -> None:
        self.recognizer = recognizer or sr.Recognizer()
        self.supported_languages = dict(self.SUPPORTED_LANGUAGES)
        self._microphone: sr.Microphone | None = None
        self._transcript_queue: queue.Queue[str] = queue.Queue()
        self._transcript_path = Path("transcript.txt")
        self._threads: tuple[threading.Thread, threading.Thread] | tuple[()] = ()
        self.is_transcribing = False

    def _get_microphone(self) -> sr.Microphone:
        """Create the microphone only when a microphone feature is requested."""
        if self._microphone is None:
            try:
                self._microphone = sr.Microphone()
            except Exception as exc:
                raise RuntimeError(
                    "Microphone access requires PyAudio and a working input device. "
                    "Install this project with the 'audio' extra."
                ) from exc
        return self._microphone

    def adjust_for_noise(self, duration: float = 1.0) -> None:
        """Calibrate the recognizer against ambient noise."""
        microphone = self._get_microphone()
        print("A moment of silence, please...")
        with microphone as source:
            self.recognizer.adjust_for_ambient_noise(source, duration=duration)
        print(f"Energy threshold: {self.recognizer.energy_threshold}")

    def recognize_audio(self, audio: sr.AudioData, language: str = "en-US") -> str | None:
        """Recognize one SpeechRecognition AudioData object with Google recognition."""
        try:
            return self.recognizer.recognize_google(audio, language=language)
        except sr.UnknownValueError:
            return None
        except sr.RequestError as exc:
            print(f"Speech recognition request failed: {exc}")
            return None

    def listen_and_recognize(
        self,
        language: str = "en-US",
        timeout: float | None = None,
        phrase_time_limit: float | None = None,
    ) -> tuple[str | None, sr.AudioData]:
        """Capture a single phrase from the microphone and transcribe it."""
        microphone = self._get_microphone()
        print("Say something!")
        with microphone as source:
            audio = self.recognizer.listen(
                source,
                timeout=timeout,
                phrase_time_limit=phrase_time_limit,
            )
        return self.recognize_audio(audio, language=language), audio

    def translate_text(
        self,
        text: str,
        target_lang: str,
        source_lang: str = "auto",
    ) -> str:
        """Translate text into one of the supported target languages."""
        if target_lang not in self.supported_languages:
            supported = ", ".join(sorted(self.supported_languages))
            raise ValueError(f"Unsupported target language '{target_lang}'. Use: {supported}")
        if not text.strip():
            return ""

        from deep_translator import GoogleTranslator

        return GoogleTranslator(source=source_lang, target=target_lang).translate(text)

    def text_to_speech(
        self,
        text: str,
        lang: str,
        output_file: str | Path = "output.mp3",
    ) -> str:
        """Generate an MP3 file from translated text without auto-playing it."""
        if lang not in self.supported_languages:
            supported = ", ".join(sorted(self.supported_languages))
            raise ValueError(f"Unsupported speech language '{lang}'. Use: {supported}")
        if not text.strip():
            raise ValueError("Text-to-speech requires non-empty text.")

        from gtts import gTTS

        path = Path(output_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        gTTS(text=text, lang=lang, slow=False).save(str(path))
        return str(path)

    def export_text(self, text: str, filename: str | Path = "output.txt") -> str:
        """Write text to a UTF-8 file and return its path."""
        path = Path(filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return str(path)

    @staticmethod
    def _load_wav(audio_file: str | Path) -> tuple[int, Any]:
        from scipy.io import wavfile

        return wavfile.read(str(audio_file))

    def visualize_audio(self, audio_file: str | Path) -> None:
        """Plot a WAV waveform."""
        import matplotlib.pyplot as plt
        import numpy as np

        sample_rate, data = self._load_wav(audio_file)
        if getattr(data, "ndim", 1) > 1:
            data = np.mean(data, axis=1)

        duration = len(data) / sample_rate
        time_axis = np.linspace(0, duration, num=len(data), endpoint=False)
        plt.figure(figsize=(10, 4))
        plt.plot(time_axis, data)
        plt.title("Audio Waveform")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.tight_layout()
        plt.show()

    def visualize_bass(
        self,
        audio_file: str | Path,
        low_freq: int = 20,
        high_freq: int = 250,
    ) -> float:
        """Plot the bass-frequency spectrogram of a WAV file and return mean dB."""
        if low_freq < 0 or high_freq <= low_freq:
            raise ValueError("Frequency bounds must satisfy 0 <= low_freq < high_freq.")

        import matplotlib.pyplot as plt
        import numpy as np
        from scipy import signal

        sample_rate, data = self._load_wav(audio_file)
        if getattr(data, "ndim", 1) > 1:
            data = np.mean(data, axis=1)

        frequencies, times, spectrum = signal.stft(
            data,
            fs=sample_rate,
            nperseg=1024,
            noverlap=512,
        )
        bass_mask = (frequencies >= low_freq) & (frequencies <= high_freq)
        bass_frequencies = frequencies[bass_mask]
        bass_magnitude = np.abs(spectrum[bass_mask])

        if bass_magnitude.size == 0:
            raise ValueError("The selected bass range contains no frequency bins.")

        safe_magnitude = np.maximum(bass_magnitude, np.finfo(float).tiny)
        bass_db = 20 * np.log10(safe_magnitude)
        average_bass_db = float(np.mean(bass_db))

        duration = len(data) / sample_rate
        waveform_time = np.linspace(0, duration, num=len(data), endpoint=False)
        fig, (waveform_axis, bass_axis) = plt.subplots(2, 1, figsize=(12, 8))

        waveform_axis.plot(waveform_time, data)
        waveform_axis.set_title("Original Waveform")
        waveform_axis.set_xlabel("Time (s)")
        waveform_axis.set_ylabel("Amplitude")

        bass_plot = bass_axis.pcolormesh(
            times,
            bass_frequencies,
            bass_db,
            shading="gouraud",
        )
        bass_axis.set_title(f"Bass Spectrogram ({low_freq}-{high_freq} Hz)")
        bass_axis.set_xlabel("Time (s)")
        bass_axis.set_ylabel("Frequency (Hz)")
        fig.colorbar(bass_plot, ax=bass_axis, label="Magnitude (dB)")
        bass_axis.text(
            0.02,
            0.95,
            f"Average: {average_bass_db:.1f} dB",
            transform=bass_axis.transAxes,
            verticalalignment="top",
        )

        plt.tight_layout()
        plt.show()
        return average_bass_db

    def process_audio_file(
        self,
        audio_file: str | Path,
        *,
        language: str = "en-US",
        target_lang: str | None = None,
        export_file: str | Path | None = None,
        translation_audio_file: str | Path | None = None,
        visualize: bool = False,
        visualize_bass: bool = False,
    ) -> dict[str, Any]:
        """Transcribe an audio file and optionally translate, export, and analyze it."""
        path = Path(audio_file)
        if not path.exists():
            raise FileNotFoundError(path)

        with sr.AudioFile(str(path)) as source:
            audio = self.recognizer.record(source)

        text = self.recognize_audio(audio, language=language)
        result: dict[str, Any] = {"text": text, "translation": None, "bass_intensity": None}
        if text is None:
            return result

        if target_lang:
            translated = self.translate_text(text, target_lang)
            result["translation"] = translated
            if translation_audio_file:
                self.text_to_speech(translated, target_lang, translation_audio_file)

        if export_file:
            self.export_text(text, export_file)

        if visualize:
            self.visualize_audio(path)
        if visualize_bass:
            result["bass_intensity"] = self.visualize_bass(path)

        return result

    def _transcribe_audio(self, audio: sr.AudioData, language: str) -> str | None:
        text = self.recognize_audio(audio, language=language)
        if text is None:
            return None
        timestamp = datetime.now().strftime("%H:%M:%S")
        return f"[{timestamp}] {text}"

    def _background_transcription(self, language: str) -> None:
        microphone = self._get_microphone()
        with microphone as source:
            self.recognizer.adjust_for_ambient_noise(source, duration=1)
            print("Real-time transcription started. Press Ctrl+C to stop.")
            while self.is_transcribing:
                try:
                    audio = self.recognizer.listen(source, timeout=2, phrase_time_limit=10)
                    text = self._transcribe_audio(audio, language)
                    if text:
                        self._transcript_queue.put(text)
                except sr.WaitTimeoutError:
                    continue
                except sr.RequestError as exc:
                    print(f"Speech recognition request failed: {exc}")
                except Exception as exc:
                    print(f"Transcription error: {exc}")

    def _display_transcription(self) -> None:
        self._transcript_path.parent.mkdir(parents=True, exist_ok=True)
        self._transcript_path.write_text("=== Speech Transcript ===\n\n", encoding="utf-8")

        while self.is_transcribing or not self._transcript_queue.empty():
            try:
                text = self._transcript_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            print(text)
            with self._transcript_path.open("a", encoding="utf-8") as transcript:
                transcript.write(text + "\n")

    def start_real_time_transcription(
        self,
        *,
        language: str = "en-US",
        transcript_file: str | Path = "transcript.txt",
    ) -> tuple[threading.Thread, threading.Thread]:
        """Start background microphone transcription and transcript-file output."""
        if self.is_transcribing and len(self._threads) == 2:
            return self._threads

        self._transcript_path = Path(transcript_file)
        self.is_transcribing = True

        transcription_thread = threading.Thread(
            target=self._background_transcription,
            args=(language,),
            daemon=True,
        )
        display_thread = threading.Thread(target=self._display_transcription, daemon=True)
        self._threads = (transcription_thread, display_thread)
        transcription_thread.start()
        display_thread.start()
        return self._threads

    def stop_real_time_transcription(self) -> None:
        """Signal background transcription threads to stop."""
        self.is_transcribing = False

    def process_speech(
        self,
        *,
        language: str = "en-US",
        target_lang: str | None = None,
        visualize: bool = False,
        export_file: str | Path | None = None,
        visualize_bass: bool = False,
        translation_audio_file: str | Path | None = None,
        real_time: bool = False,
        transcript_file: str | Path = "transcript.txt",
    ) -> dict[str, Any] | None:
        """Run either one-shot microphone processing or continuous transcription."""
        if real_time:
            threads = self.start_real_time_transcription(
                language=language,
                transcript_file=transcript_file,
            )
            try:
                while self.is_transcribing:
                    time.sleep(0.1)
            except KeyboardInterrupt:
                self.stop_real_time_transcription()
            finally:
                for thread in threads:
                    thread.join(timeout=2)
            return None

        self.adjust_for_noise()
        text, audio = self.listen_and_recognize(language=language)
        result: dict[str, Any] = {"text": text, "translation": None, "bass_intensity": None}
        if text is None:
            return result

        print(f"Recognized text: {text}")

        if target_lang:
            translated = self.translate_text(text, target_lang)
            result["translation"] = translated
            print(f"Translated text ({self.supported_languages[target_lang]}): {translated}")
            if translation_audio_file:
                self.text_to_speech(translated, target_lang, translation_audio_file)

        if export_file:
            self.export_text(text, export_file)

        if visualize or visualize_bass:
            temporary_path: Path | None = None
            try:
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temporary_file:
                    temporary_file.write(audio.get_wav_data())
                    temporary_path = Path(temporary_file.name)

                if visualize:
                    self.visualize_audio(temporary_path)
                if visualize_bass:
                    result["bass_intensity"] = self.visualize_bass(temporary_path)
            finally:
                if temporary_path and temporary_path.exists():
                    temporary_path.unlink()

        return result
