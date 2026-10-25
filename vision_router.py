import logging
import os
import time

import local_vision
import ollama_client

log = logging.getLogger("vision")

MODE = os.environ.get("VISION_MODE", "auto")             # auto | cloud | local
COOLDOWN_S = float(os.environ.get("CLOUD_COOLDOWN_S", "30"))
CLOUD_TIMEOUT = (3, 15)                                  # fail fast so the fallback feels quick

READ_SYSTEM = ("You read text from camera images for a blind person who hears the answer "
               "through a speaker. Output only the visible text in natural reading order, "
               "with no commentary. If there is no readable text, say exactly: "
               "I don't see any text.")

_down_until = 0.0
_last_source = "cloud"


def _run(kind, cloud_fn, local_fn):
    """Returns (text, source). Announces offline/online changes by prefixing the answer."""
    global _down_until, _last_source
    text, source = None, "local"
    if MODE != "local" and time.monotonic() >= _down_until:
        try:
            text, source = cloud_fn(), "cloud"
        except ollama_client.OllamaError as e:
            log.warning("cloud %s failed: %s", kind, e)
            _down_until = time.monotonic() + COOLDOWN_S      # skip the cloud for a while
            if MODE == "cloud":
                return str(e), "error"
    if text is None:
        text = local_fn()
    prefix = ""
    if MODE == "auto" and source != _last_source:
        prefix = "Back online. " if source == "cloud" else "Offline mode. "
    _last_source = source
    return prefix + text, source


def describe(frame):
    return _run("see",
                lambda: ollama_client.describe_frame(frame, timeout=CLOUD_TIMEOUT),
                lambda: local_vision.describe(frame))


def read(frame):
    return _run("read",
                lambda: ollama_client.describe_frame(
                    frame, prompt="Read the text in this image.", system=READ_SYSTEM,
                    max_tokens=300, timeout=CLOUD_TIMEOUT),
                lambda: local_vision.read_text(frame))


def ask(frame, question):
    prompt = f"The person asked: {question}\nAnswer using only what is visible in the image."
    return _run("ask",
                lambda: ollama_client.describe_frame(frame, prompt=prompt, timeout=CLOUD_TIMEOUT),
                lambda: local_vision.answer_question(question, frame))