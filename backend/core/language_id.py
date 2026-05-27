# ----- Language ID (VoxLingua107) @ backend/core/language_id.py -----

import numpy as np
import torch

# Maps VoxLingua107 language names/codes → 5 supported ISO codes.
VOXLINGUA_TO_CODE: dict[str, str] = {
    "Hindi": "hi",
    "English": "en",
    "Telugu": "te",
    "Bengali": "bn",
    "Marathi": "mr",
    "hi": "hi",
    "en": "en",
    "te": "te",
    "bn": "bn",
    "mr": "mr",
}

_classifier = None


def _load_classifier():
    """Lazy-load the SpeechBrain VoxLingua107 ECAPA-TDNN classifier."""
    from speechbrain.inference.classifiers import EncoderClassifier

    return EncoderClassifier.from_hparams(
        source="speechbrain/lang-id-voxlingua107-ecapa",
        savedir="tmp/speechbrain_models",
    )


def identify_language(audio: np.ndarray, sr: int = 16000) -> tuple[str, float]:
    """Identify the language of an audio utterance.

    Parameters
    ----------
    audio : np.ndarray
        Mono audio waveform as a 1-D float32 array (values in [-1, 1]).
    sr : int
        Sampling rate of ``audio`` (default 16000).

    Returns
    -------
    lang_code : str
        One of ``"hi"``, ``"en"``, ``"te"``, ``"bn"``, ``"mr"``.
        Falls back to ``"hi"`` when the predicted language is not in the
        supported set.
    confidence : float
        Softmax probability of the top prediction (0–1 range).
    """
    global _classifier

    if _classifier is None:
        _classifier = _load_classifier()

    signal = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)
    out_prob, score, index, text_lab = _classifier.classify_batch(signal)

    # score is a log-probability; exponentiate for a 0–1 confidence value.
    confidence = float(score[0].exp().item())

    # text_lab may be "Hindi" (name) or "bn: Bengali" (code: name)
    # depending on the SpeechBrain version. Handle both.
    raw_label = str(text_lab[0])
    if ": " in raw_label:
        label_code, label_name = raw_label.split(": ", maxsplit=1)
        lang_code = VOXLINGUA_TO_CODE.get(label_code) or VOXLINGUA_TO_CODE.get(
            label_name
        )
    else:
        lang_code = VOXLINGUA_TO_CODE.get(raw_label)

    if lang_code is None and raw_label == "Tamil":
        # Keep a defensive alias for the design-doc/code mismatch around `te`.
        lang_code = "te"

    if lang_code is None:
        lang_code = "hi"
    return lang_code, confidence


def update_active_language(
    prediction: str,
    confidence: float,
    word_count: int,
    current_language: str,
    is_first_utterance: bool,
) -> str:
    """Decide whether to switch the active call language.

    Parameters
    ----------
    prediction : str
        Language code predicted by :func:`identify_language`.
    confidence : float
        Confidence of the prediction (0–1).
    word_count : int
        Number of words in the current utterance.
    current_language : str
        Currently active language for this call.
    is_first_utterance : bool
        Whether this is the first utterance in the call.

    Returns
    -------
    str
        The language code to use for subsequent processing.
    """
    if is_first_utterance and confidence < 0.80:
        return "hi"
    if confidence >= 0.80 and word_count > 5:
        return prediction
    return current_language
