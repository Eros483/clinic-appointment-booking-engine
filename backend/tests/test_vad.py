# ----- 12 unit tests for Silero VAD processor @ backend/tests/test_vad.py -----

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

import backend.core.vad as vad_mod
from backend.core.vad import (
    BARGE_IN,
    PRE_ROLL_FRAMES,
    SILENCE_THRESHOLD_FRAMES,
    SPEECH_PROB_THRESHOLD,
    VADProcessor,
    VAD_END,
    VAD_START,
)


def _make_mock_model(speech_prob: float) -> MagicMock:
    out = torch.tensor([[speech_prob]], dtype=torch.float32)
    mock = MagicMock(return_value=out)
    mock.reset_states = MagicMock()
    return mock


def _silence_frame() -> np.ndarray:
    return np.zeros(512, dtype=np.float32)


def _speech_frame() -> np.ndarray:
    return np.ones(512, dtype=np.float32) * 0.1


@pytest.fixture(autouse=True)
def _reset_model():
    vad_mod._model = None
    yield


class TestVADProcessor:
    def test_vad_start_on_speech_after_silence(self):
        proc = VADProcessor()
        mock_model = _make_mock_model(0.0)
        with patch.object(vad_mod, "_model", mock_model):
            assert proc.process(_silence_frame()) is None
            assert proc.state == "idle"

        mock_speech = _make_mock_model(0.9)
        with patch.object(vad_mod, "_model", mock_speech):
            event = proc.process(_speech_frame())
            assert event == VAD_START
            assert proc.state == "speaking"

    def test_vad_end_after_silence_threshold(self):
        proc = VADProcessor()
        mock_speech = _make_mock_model(0.9)
        with patch.object(vad_mod, "_model", mock_speech):
            proc.process(_speech_frame())
            assert proc.state == "speaking"

        mock_silence = _make_mock_model(0.0)
        with patch.object(vad_mod, "_model", mock_silence):
            for _ in range(SILENCE_THRESHOLD_FRAMES - 1):
                event = proc.process(_silence_frame())
                assert event is None, "no VAD_END before threshold"

            event = proc.process(_silence_frame())
            assert event == VAD_END
            assert proc.state == "idle"

    def test_barge_in_during_tts(self):
        proc = VADProcessor()
        mock_silence = _make_mock_model(0.0)
        with patch.object(vad_mod, "_model", mock_silence):
            assert proc.process(_silence_frame()) is None

        mock_speech = _make_mock_model(0.9)
        with patch.object(vad_mod, "_model", mock_speech):
            event = proc.process(_speech_frame(), is_tts_active=True)
            assert event == BARGE_IN
            assert proc.state == "speaking"

    def test_pre_roll_buffer_captured_on_vad_start(self):
        proc = VADProcessor()
        mock_silence = _make_mock_model(0.0)
        with patch.object(vad_mod, "_model", mock_silence):
            for _ in range(PRE_ROLL_FRAMES):
                proc.process(_silence_frame())

        mock_speech = _make_mock_model(0.9)
        with patch.object(vad_mod, "_model", mock_speech):
            event = proc.process(_speech_frame())

        assert event == VAD_START
        expected_len = PRE_ROLL_FRAMES + 1
        assert len(proc.utterance) == expected_len

    def test_utterance_accumulates_during_speech(self):
        proc = VADProcessor()
        mock_silence = _make_mock_model(0.0)
        with patch.object(vad_mod, "_model", mock_silence):
            for _ in range(PRE_ROLL_FRAMES):
                proc.process(_silence_frame())

        mock_speech = _make_mock_model(0.9)
        with patch.object(vad_mod, "_model", mock_speech):
            proc.process(_speech_frame())
            for _ in range(5):
                proc.process(_speech_frame())

        assert len(proc.utterance) == PRE_ROLL_FRAMES + 1 + 5

    def test_get_utterance_audio_returns_concatenated(self):
        proc = VADProcessor()
        mock_silence = _make_mock_model(0.0)
        with patch.object(vad_mod, "_model", mock_silence):
            for _ in range(PRE_ROLL_FRAMES):
                proc.process(_silence_frame())

        mock_speech = _make_mock_model(0.9)
        with patch.object(vad_mod, "_model", mock_speech):
            proc.process(_speech_frame())

        audio = proc.get_utterance_audio()
        assert audio.dtype == np.float32
        assert len(audio) == (PRE_ROLL_FRAMES + 1) * 512

    def test_get_utterance_audio_empty_when_no_speech(self):
        proc = VADProcessor()
        audio = proc.get_utterance_audio()
        assert audio.shape == (0,)

    def test_reset_clears_all_state(self):
        proc = VADProcessor()
        mock_speech = _make_mock_model(0.9)
        with patch.object(vad_mod, "_model", mock_speech):
            proc.process(_speech_frame())

            assert proc.state == "speaking"
            assert len(proc.utterance) > 0

            proc.reset()
            assert proc.state == "idle"
            assert len(proc.utterance) == 0
            assert len(proc.pre_roll) == 0
            mock_speech.reset_states.assert_called_once()

    def test_short_burst_does_not_emit_early_end(self):
        proc = VADProcessor()
        mock_speech = _make_mock_model(0.9)
        with patch.object(vad_mod, "_model", mock_speech):
            proc.process(_speech_frame())

        mock_silence = _make_mock_model(0.0)
        with patch.object(vad_mod, "_model", mock_silence):
            event = proc.process(_silence_frame())
            assert event is None

    def test_full_idle_speaking_idle_cycle(self):
        proc = VADProcessor()
        mock_speech = _make_mock_model(0.9)
        with patch.object(vad_mod, "_model", mock_speech):
            e1 = proc.process(_speech_frame())
            assert e1 == VAD_START

        mock_silence = _make_mock_model(0.0)
        with patch.object(vad_mod, "_model", mock_silence):
            for _ in range(SILENCE_THRESHOLD_FRAMES):
                e2 = proc.process(_silence_frame())
            assert e2 == VAD_END
            assert proc.state == "idle"

            e3 = proc.process(_silence_frame())
            assert e3 is None
            assert proc.state == "idle"

    def test_raises_on_wrong_frame_size(self):
        proc = VADProcessor()
        with pytest.raises(ValueError):
            proc.process(np.zeros(256, dtype=np.float32))

    def test_barge_in_flushes_previous_utterance(self):
        proc = VADProcessor()
        mock_silence = _make_mock_model(0.0)
        with patch.object(vad_mod, "_model", mock_silence):
            for _ in range(PRE_ROLL_FRAMES):
                proc.process(_silence_frame())

        mock_speech = _make_mock_model(0.9)
        with patch.object(vad_mod, "_model", mock_speech):
            proc.process(_speech_frame())

        assert len(proc.utterance) == PRE_ROLL_FRAMES + 1

        barge_model = _make_mock_model(0.9)
        with patch.object(vad_mod, "_model", barge_model):
            event = proc.process(_speech_frame(), is_tts_active=True)
            assert event == BARGE_IN
            assert len(proc.utterance) == PRE_ROLL_FRAMES + 1
            barge_model.reset_states.assert_called_once()
