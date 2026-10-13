import os
import sys

import pytest
import requests
import streamlit as st
from streamlit.testing.v1 import AppTest

from conftest import APP_DIR, PROJECT_DIR, SAMPLES_DIR

DEPLOY = os.path.join(APP_DIR, "deploy.py")


def sample(name):
    with open(os.path.join(SAMPLES_DIR, name), "rb") as f:
        return f.read()


def new_session(script=DEPLOY):
    return AppTest.from_file(script, default_timeout=300).run()


def upload(at, data=None, name="leaf.jpg"):
    at.file_uploader[0].upload(name, data if data is not None else sample("AppleCedarRust1.JPG"), "image/jpeg").run()


def detect(at):
    next(b for b in at.button if b.label == "Detect Disease").click().run()


def detected(at):
    return [m.value for m in at.markdown if m.value.startswith("### Detected Disease")]


def use_gpt(at, key=None):
    at.sidebar.selectbox[0].select("gpt4o_mini").run()
    if key:
        next(t for t in at.sidebar.text_input if "API Key" in t.label).input(key).run()


def assert_no_errors(at):
    assert not at.exception, [e.value for e in at.exception]
    assert not at.error, [e.value for e in at.error]


@pytest.fixture
def openai_calls(monkeypatch):
    """Replace the OpenAI endpoint with a recorder and start from an empty advice cache."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    st.cache_resource.clear()
    calls = []

    class FakeResponse:
        status_code = 200
        text = "ok"

        def json(self):
            return {"choices": [{"message": {"content": "# LLM advice"}}]}

    def fake_post(url, headers=None, json=None, timeout=None):
        calls.append({"auth": headers["Authorization"], "temperature": json["temperature"]})
        return FakeResponse()

    monkeypatch.setattr(requests, "post", fake_post)
    return calls


@pytest.mark.parametrize("page, subheader", [
    ("About", "About This Project"),
    ("How It Works", "How It Works"),
    ("Settings", "Application Settings"),
])
def test_pages_render(page, subheader):
    at = new_session()
    at.sidebar.radio[0].set_value(page).run()
    assert_no_errors(at)
    assert subheader in [s.value for s in at.subheader]


def test_detection_offline(openai_calls):
    at = new_session()
    assert at.sidebar.selectbox[0].value == "offline"
    upload(at)
    detect(at)
    assert_no_errors(at)
    assert detected(at) == ["### Detected Disease: **Apple - Cedar apple rust**"]
    assert openai_calls == []


def test_landing_page_lists_only_supported_plants():
    text = " ".join(m.value for m in new_session().markdown)
    assert "38 classes across 14 plant species" in text
    assert "Cotton" not in text


def test_results_survive_reruns():
    at = new_session()
    upload(at)
    detect(at)
    at.run()
    assert detected(at)


def test_new_upload_clears_previous_result():
    at = new_session()
    upload(at)
    detect(at)
    upload(at, sample("PotatoHealthy1.JPG"), "other.jpg")
    assert detected(at) == []
    detect(at)
    assert detected(at) == ["### Detected Disease: **Potato - Healthy**"]


def test_unreadable_upload_shows_error():
    at = new_session()
    upload(at, b"definitely not a jpeg", "broken.jpg")
    assert not at.exception
    assert [e.value for e in at.error] == ["Could not read this file as an image. Please upload a JPG or PNG photo of a leaf."]


def test_api_key_is_not_shared_between_sessions(openai_calls):
    first = new_session()
    use_gpt(first, "sk-USER-A")
    upload(first)
    detect(first)
    assert [c["auth"] for c in openai_calls] == ["Bearer sk-USER-A"]

    second = new_session()
    use_gpt(second)
    upload(second, sample("PotatoHealthy1.JPG"))
    detect(second)
    assert [c["auth"] for c in openai_calls] == ["Bearer sk-USER-A"]
    assert "OPENAI_API_KEY" not in os.environ
    assert_no_errors(second)


def test_server_key_is_used_by_default(openai_calls, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-SERVER")
    at = new_session()
    assert at.sidebar.selectbox[0].value == "gpt4o_mini"
    upload(at)
    detect(at)
    assert [c["auth"] for c in openai_calls] == ["Bearer sk-SERVER"]


def test_temperature_setting_reaches_the_api(openai_calls):
    at = new_session()
    use_gpt(at, "sk-USER-A")
    at.sidebar.radio[0].set_value("Settings").run()
    next(s for s in at.slider if s.label.startswith("Temperature")).set_value(0.1).run()
    at.sidebar.radio[0].set_value("Disease Detection").run()
    upload(at)
    detect(at)
    assert [c["temperature"] for c in openai_calls] == [0.1]


def test_advice_is_cached(openai_calls):
    at = new_session()
    use_gpt(at, "sk-USER-A")
    upload(at)
    detect(at)
    detect(at)
    at.run()
    assert len(openai_calls) == 1


def test_app_works_without_transformers(monkeypatch):
    monkeypatch.setitem(sys.modules, "transformers", None)
    at = new_session()
    upload(at)
    detect(at)
    assert_no_errors(at)
    assert detected(at)


def test_app_py_entry_point():
    at = new_session(os.path.join(PROJECT_DIR, "app.py"))
    upload(at)
    detect(at)
    assert_no_errors(at)
    assert detected(at) == ["### Detected Disease: **Apple - Cedar apple rust**"]
