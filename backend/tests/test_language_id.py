# ----- 8 unit tests for language ID @ backend/tests/test_language_id.py -----

import math
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

import backend.core.language_id as lang_id_mod
from backend.core.language_id import (
    VOXLINGUA_TO_CODE,
    identify_language,
    update_active_language,
)


@pytest.fixture(autouse=True)
def _reset_classifier():
    lang_id_mod._classifier = None
    yield


class TestVoxLinguaToCode:
    def test_voxlingua_to_code_maps_all_five_languages(self):
        assert VOXLINGUA_TO_CODE == {
            "Hindi": "hi",
            "English": "en",
            "Tamil": "te",
            "Bengali": "bn",
            "Marathi": "mr",
        }


class TestIdentifyLanguage:
    def _make_mock_classifier(self, label: str, confidence: float) -> MagicMock:
        """Build a mock classifier whose classify_batch returns the given
        label and (log-prob → exponentiates back to *confidence*)."""
        mock_clf = MagicMock()
        # score[0] must be a real tensor so .exp().item() works naturally.
        score_tensor = torch.tensor([math.log(confidence)])
        mock_clf.classify_batch.return_value = (
            MagicMock(),  # out_prob
            score_tensor,  # score  (log-probability)
            MagicMock(),  # index
            [label],  # text_lab
        )
        return mock_clf

    def test_identify_language_fallback_on_unrecognised_label(self):
        mock_clf = self._make_mock_classifier("xyz UnknownLang", 0.9)
        with patch(
            "backend.core.language_id._load_classifier",
            return_value=mock_clf,
        ):
            lang, conf = identify_language(np.zeros(16000, dtype=np.float32))

        assert lang == "hi"
        assert conf == pytest.approx(0.9, rel=1e-5)

    def test_identify_language_returns_confidence_from_model(self):
        mock_clf = self._make_mock_classifier("Hindi", 0.95)
        with patch(
            "backend.core.language_id._load_classifier",
            return_value=mock_clf,
        ):
            lang, conf = identify_language(np.zeros(16000, dtype=np.float32))

        assert lang == "hi"
        assert conf == pytest.approx(0.95, rel=1e-5)

    def test_identify_language_handles_code_colon_name_format(self):
        """SpeechBrain sometimes returns 'hi: Hindi' instead of plain 'Hindi'."""
        mock_clf = self._make_mock_classifier("hi: Hindi", 0.88)
        with patch(
            "backend.core.language_id._load_classifier",
            return_value=mock_clf,
        ):
            lang, conf = identify_language(np.zeros(16000, dtype=np.float32))

        assert lang == "hi"
        assert conf == pytest.approx(0.88, rel=1e-5)


class TestUpdateActiveLanguage:
    def test_update_active_language_switches_on_high_confidence_and_word_count(
        self,
    ):
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
            prediction="te",
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
