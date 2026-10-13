import pytest

import deploy
import plant_model

DEFAULT_ADVICE = deploy.get_fallback_advice("not_a_known_label")


@pytest.mark.parametrize("arch", ["resnet50", "custom_cnn"])
def test_fallback_advice_ignores_label_case(arch):
    advice = {label.lower(): deploy.get_fallback_advice(label) for label in plant_model.load_class_names(arch)}
    for label in ["apple___apple_scab", "grape___black_rot", "tomato___early_blight", "tomato___late_blight"]:
        assert advice[label] != DEFAULT_ADVICE, label


@pytest.mark.parametrize("arch", ["resnet50", "custom_cnn"])
def test_healthy_leaves_get_preventive_advice(arch):
    healthy = [label for label in plant_model.load_class_names(arch) if label.lower().endswith("___healthy")]
    assert healthy
    for label in healthy:
        advice = deploy.get_fallback_advice(label)
        assert "# Healthy Plant" in advice
        assert "appears to be a plant disease" not in advice


def test_gpt_without_key_falls_back_without_calling_the_api(monkeypatch):
    monkeypatch.setattr(deploy.requests, "post", lambda *a, **k: pytest.fail("API called without a key"))
    assert deploy.get_advice("Tomato___Late_blight", "gpt4o_mini", api_key=None) == \
        deploy.get_fallback_advice("Tomato___Late_blight")


@pytest.fixture
def tiny_local_llm(tmp_path, monkeypatch):
    """A tiny randomly initialised GPT-2 saved locally, standing in for Phi-2 (no download)."""
    pytest.importorskip("transformers")
    import torch
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

    words = deploy.create_disease_prompt("Apple - Apple scab").split()
    vocab = {"[UNK]": 0, "<eos>": 1, **{w: i + 2 for i, w in enumerate(dict.fromkeys(words))}}
    tokenizer = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token="[UNK]", eos_token="<eos>",
                            pad_token="<eos>").save_pretrained(tmp_path)
    torch.manual_seed(0)
    GPT2LMHeadModel(GPT2Config(vocab_size=len(vocab), n_embd=16, n_layer=1, n_head=2,
                               bos_token_id=1, eos_token_id=1)).save_pretrained(tmp_path)

    monkeypatch.setitem(deploy.LLM_CONFIG["phi2"], "model_id", str(tmp_path))
    monkeypatch.setitem(deploy.LLM_CONFIG["phi2"], "max_new_tokens", 16)
    deploy.load_llm.clear()
    deploy.get_advice_cache.clear()
    yield
    deploy.load_llm.clear()


def test_local_llm_returns_generated_text(tiny_local_llm, monkeypatch):
    errors = []
    monkeypatch.setattr(deploy.st, "error", lambda msg, *a, **k: errors.append(msg))

    advice = deploy.get_advice("Apple___Apple_scab", "phi2", temperature=0.7)

    assert errors == []
    assert advice != deploy.get_fallback_advice("Apple___Apple_scab")
    assert "agricultural expert" not in advice  # prompt is not echoed back
