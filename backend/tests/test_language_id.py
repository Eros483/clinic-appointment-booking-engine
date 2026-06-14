# ----- 8 unit tests for custom lang ID @ backend/tests/test_language_id.py -----

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

import backend.core.language_id as lang_id_mod
from backend.core.language_id import (
    LANG_CODES,
    identify_language,
    update_active_language,
)


@pytest.fixture(autouse=True)
def _reset_session():
    lang_id_mod._session = None
    yield


class TestLangCodes:
    def test_lang_codes_contains_all_five_languages(self):
        assert set(LANG_CODES) == {"hi", "en", "ta", "bn", "mr"}


class TestIdentifyLanguage:
    def _make_mock_session(self, logits: list[float]) -> MagicMock:
        mock_session = MagicMock()
        mock_session.run.return_value = [np.array([logits], dtype=np.float32)]
        return mock_session

    def test_identify_language_fallback_on_no_model(self):
        lang, conf, raw = identify_language(np.zeros(16000, dtype=np.float32))
        assert lang == "hi"
        assert conf == 0.0

    def test_identify_language_returns_correct_label(self):
        logits = [2.0, 0.5, 0.3, 0.2, 0.1]
        mock_session = self._make_mock_session(logits)
        with patch.object(lang_id_mod, "_load_model", return_value=None):
            lang_id_mod._session = mock_session
            lang, conf, raw = identify_language(np.zeros(16000, dtype=np.float32))

        assert lang == "hi"
        assert conf == pytest.approx(0.581, rel=1e-2)

    def test_identify_language_returns_english_when_top(self):
        logits = [0.1, 3.0, 0.2, 0.3, 0.1]
        mock_session = self._make_mock_session(logits)
        with patch.object(lang_id_mod, "_load_model", return_value=None):
            lang_id_mod._session = mock_session
            lang, conf, raw = identify_language(np.zeros(16000, dtype=np.float32))

        assert lang == "en"
        assert conf == pytest.approx(0.808, rel=1e-2)

    def test_identify_language_returns_tamil_when_top(self):
        logits = [0.1, 0.2, 3.0, 0.3, 0.1]
        mock_session = self._make_mock_session(logits)
        with patch.object(lang_id_mod, "_load_model", return_value=None):
            lang_id_mod._session = mock_session
            lang, conf, raw = identify_language(np.zeros(16000, dtype=np.float32))

        assert lang == "ta"
        assert conf == pytest.approx(0.808, rel=1e-2)

    def test_identify_language_handles_low_confidence(self):
        logits = [0.3, 0.25, 0.28, 0.27, 0.26]
        mock_session = self._make_mock_session(logits)
        with patch.object(lang_id_mod, "_load_model", return_value=None):
            lang_id_mod._session = mock_session
            lang, conf, raw = identify_language(np.zeros(16000, dtype=np.float32))

        assert lang == "hi"
        assert conf == pytest.approx(0.206, rel=1e-1)

    def test_identify_language_resamples_audio(self):
        logits = [0.1, 3.0, 0.2, 0.3, 0.1]
        mock_session = self._make_mock_session(logits)
        with patch.object(lang_id_mod, "_load_model", return_value=None):
            lang_id_mod._session = mock_session
            audio_8k = np.zeros(8000, dtype=np.float32)
            lang, conf, raw = identify_language(audio_8k, sr=8000)

        assert lang == "en"


class TestUpdateActiveLanguage:
    def test_update_active_language_switches_on_high_confidence_and_word_count(self):
        result = update_active_language(
            prediction="en",
            confidence=0.85,
            word_count=6,
            current_language="hi",
            is_first_utterance=False,
        )
        assert result == "en"

    def test_update_active_language_stays_on_low_confidence(self):
        result = update_active_language(
            prediction="en",
            confidence=0.60,
            word_count=10,
            current_language="hi",
            is_first_utterance=False,
        )
        assert result == "hi"

    def test_update_active_language_first_utterance_defaults_to_hi(self):
        result = update_active_language(
            prediction="en",
            confidence=0.70,
            word_count=3,
            current_language="en",
            is_first_utterance=True,
        )
        assert result == "hi"

    def test_update_active_language_ignores_switch_on_low_word_count(self):
        result = update_active_language(
            prediction="ta",
            confidence=0.90,
            word_count=3,
            current_language="hi",
            is_first_utterance=False,
        )
        assert result == "hi"

    def test_update_active_language_keeps_current_when_no_condition_met(self):
        result = update_active_language(
            prediction="bn",
            confidence=0.79,
            word_count=4,
            current_language="mr",
            is_first_utterance=False,
        )
        assert result == "mr"
