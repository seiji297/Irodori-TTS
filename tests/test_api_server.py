"""api_server モジュールのユニットテスト（split_text チャンキングロジック）。"""

import sys
import types

# api_server をインポートする前に重いモジュールをモック化する
# （GPU不要テスト環境でもimportできるようにする）
for mod_name in [
    "torch", "torchaudio", "fastapi", "fastapi.responses", "pydantic",
    "huggingface_hub", "irodori_tts", "irodori_tts.inference_runtime",
]:
    if mod_name not in sys.modules:
        sys.modules[mod_name] = types.ModuleType(mod_name)

# fastapi stub
import fastapi as _fa  # noqa: E402
def _noop_decorator(*a, **kw):
    def decorator(fn):
        return fn
    return decorator
_dummy_app = type("FastAPI", (), {
    "__init__": lambda *a, **kw: None,
    "post": staticmethod(_noop_decorator),
    "get": staticmethod(_noop_decorator),
})()
_fa.FastAPI = lambda **kw: _dummy_app  # type: ignore

# pydantic stub
import pydantic as _pd  # noqa: E402
_pd.BaseModel = object  # type: ignore
_pd.Field = lambda *a, **kw: None  # type: ignore

# irodori_tts.inference_runtime stub
_rt = sys.modules["irodori_tts.inference_runtime"]
_rt.InferenceRuntime = type("InferenceRuntime", (), {})  # type: ignore
_rt.RuntimeKey = type("RuntimeKey", (), {"__init__": lambda *a, **kw: None})  # type: ignore
_rt.SamplingRequest = type("SamplingRequest", (), {"__init__": lambda *a, **kw: None})  # type: ignore
_rt.default_runtime_device = lambda: "cpu"  # type: ignore

# huggingface_hub stub
import huggingface_hub as _hf  # noqa: E402
_hf.hf_hub_download = lambda **kw: "/tmp/model.safetensors"  # type: ignore

# torch stub (need zeros, cat, Tensor)
import torch as _torch  # noqa: E402
_torch.zeros = lambda *a, **kw: []  # type: ignore
_torch.cat = lambda parts, dim=0: []  # type: ignore
_torch.Tensor = list  # type: ignore

# torchaudio stub
import torchaudio as _ta  # noqa: E402
_ta.save = lambda *a, **kw: None  # type: ignore

# fastapi.responses stub
import fastapi.responses as _far  # noqa: E402
_far.Response = type("Response", (), {"__init__": lambda *a, **kw: None})  # type: ignore

from api_server import split_text  # noqa: E402


class TestSplitText:
    """split_text() のスマートチャンキングロジックのテスト。"""

    def test_short_text_no_split(self):
        """80文字以下の短文は分割されない。"""
        text = "こんにちは、世界！"
        result = split_text(text)
        assert result == ["こんにちは、世界！"]

    def test_long_text_split_by_kuten(self):
        """句点で分割可能な長文は句点で分割される。"""
        text = "これは最初の文です。これは二番目の文です。これは三番目の文です。"
        result = split_text(text)
        assert len(result) == 3
        assert "これは最初の文です。" in result
        assert "これは二番目の文です。" in result
        assert "これは三番目の文です。" in result

    def test_very_long_text_split_by_touten(self):
        """80文字超の文は読点で再分割される。"""
        # 読点で区切られた80文字超の文を作成
        long_sentence = "あ" * 40 + "、" + "い" * 40 + "。"
        result = split_text(long_sentence, max_chars=80)
        # 80文字超なので読点で分割されるはず
        assert len(result) >= 2

    def test_empty_string_returns_empty_list(self):
        """空文字列は空リストを返す。"""
        assert split_text("") == []

    def test_whitespace_only_returns_empty_list(self):
        """空白のみの文字列は空リストを返す。"""
        assert split_text("   ") == []
        assert split_text("\n\t") == []

    def test_no_punctuation_long_text_returned_as_is(self):
        """句点も読点もない超長文はそのまま返す（分割不能）。"""
        text = "あ" * 100  # 100文字、区切り文字なし
        result = split_text(text, max_chars=80)
        # 分割不能なので1要素のリストになる
        assert result == [text]

    def test_exclamation_splits(self):
        """感嘆符（！）で分割されること。"""
        text = "すごい！やったね！"
        result = split_text(text)
        assert len(result) == 2

    def test_question_mark_splits(self):
        """疑問符（？）で分割されること。"""
        text = "どうですか？いいですね？"
        result = split_text(text)
        assert len(result) == 2

    def test_newline_splits(self):
        """改行（\\n）で分割されること。"""
        text = "一行目\n二行目\n三行目"
        result = split_text(text)
        assert len(result) == 3

    def test_strip_removes_extra_whitespace(self):
        """各チャンクの前後の空白が除去されること。"""
        text = "  テスト文。  別の文。  "
        result = split_text(text)
        for chunk in result:
            assert chunk == chunk.strip()

    def test_comma_splits_long_sentence(self):
        """カンマ（,）でも再分割されること。"""
        long_sentence = "a" * 41 + "," + "b" * 41 + "。"
        result = split_text(long_sentence, max_chars=80)
        assert len(result) >= 2

    def test_max_chars_respected(self):
        """分割後の各チャンクがmax_chars以内に収まること（区切り文字がある場合）。"""
        # 句点で分割できる長文
        text = "短文です。" * 5
        result = split_text(text, max_chars=80)
        # 各チャンクは5文字 = "短文です。" なのでmax_chars内
        for chunk in result:
            assert len(chunk) <= 80
