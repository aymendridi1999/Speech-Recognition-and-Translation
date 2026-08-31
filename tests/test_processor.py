from unittest.mock import MagicMock

import pytest
import speech_recognition as sr

from speech_toolkit import SpeechProcessor


def test_microphone_is_lazy(monkeypatch):
    def fail_if_called():
        raise AssertionError("Microphone should not be created during initialization")

    monkeypatch.setattr(sr, "Microphone", fail_if_called)
    SpeechProcessor()


def test_recognize_audio_returns_text():
    recognizer = MagicMock(spec=sr.Recognizer)
    recognizer.recognize_google.return_value = "hello world"
    processor = SpeechProcessor(recognizer=recognizer)
    audio = MagicMock(spec=sr.AudioData)

    assert processor.recognize_audio(audio, language="en-US") == "hello world"
    recognizer.recognize_google.assert_called_once_with(audio, language="en-US")


def test_recognize_audio_handles_unknown_value():
    recognizer = MagicMock(spec=sr.Recognizer)
    recognizer.recognize_google.side_effect = sr.UnknownValueError()
    processor = SpeechProcessor(recognizer=recognizer)

    assert processor.recognize_audio(MagicMock(spec=sr.AudioData)) is None


def test_translate_text_uses_auto_source(monkeypatch):
    translator = MagicMock()
    translator.translate.return_value = "bonjour"
    translator_class = MagicMock(return_value=translator)

    import deep_translator

    monkeypatch.setattr(deep_translator, "GoogleTranslator", translator_class)
    processor = SpeechProcessor()

    assert processor.translate_text("hello", "fr") == "bonjour"
    translator_class.assert_called_once_with(source="auto", target="fr")


def test_translate_text_rejects_unsupported_language():
    processor = SpeechProcessor()

    with pytest.raises(ValueError):
        processor.translate_text("hello", "it")


def test_export_text(tmp_path):
    processor = SpeechProcessor()
    output = tmp_path / "transcript.txt"

    returned_path = processor.export_text("hello", output)

    assert output.read_text(encoding="utf-8") == "hello"
    assert returned_path == str(output)


def test_text_to_speech_uses_requested_output(monkeypatch, tmp_path):
    tts = MagicMock()
    tts_class = MagicMock(return_value=tts)

    import gtts

    monkeypatch.setattr(gtts, "gTTS", tts_class)
    processor = SpeechProcessor()
    output = tmp_path / "speech.mp3"

    returned_path = processor.text_to_speech("bonjour", "fr", output)

    tts_class.assert_called_once_with(text="bonjour", lang="fr", slow=False)
    tts.save.assert_called_once_with(str(output))
    assert returned_path == str(output)


def test_transcribe_audio_adds_timestamp():
    recognizer = MagicMock(spec=sr.Recognizer)
    recognizer.recognize_google.return_value = "test phrase"
    processor = SpeechProcessor(recognizer=recognizer)

    result = processor._transcribe_audio(MagicMock(spec=sr.AudioData), "en-US")

    assert result is not None
    assert result.endswith(" test phrase")
    assert result.startswith("[")
