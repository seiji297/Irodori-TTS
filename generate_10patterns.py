#!/usr/bin/env python3
"""10パターンの声色WAVを生成して outputs/ に保存する。"""
import json
import os
import subprocess
import sys
import tempfile
import urllib.request

API_URL = "http://localhost:50032/generate"
OUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

# 3文の日本語テキスト（十分な長さで10秒以上を確保）
TEXT = (
    "今日はとても良い天気ですね。空が青くて、風も気持ちよく吹いています。"
    "こんな日は外に出かけて、散歩でもしたくなりますね。"
    "みなさんも良い一日をお過ごしください。"
)

# 10パターン：声色・トーン・話し方を大きく変える
PATTERNS = [
    {
        "name": "pattern01_young_bright",
        "caption": "非常に幼い女の子の声で、明るく元気いっぱいに、ハキハキと喋ってください。声のトーンは高めで、可愛らしく、通る声でお願いします。",
    },
    {
        "name": "pattern02_adult_calm",
        "caption": "落ち着いた成人女性の声で、ゆっくりと丁寧に話してください。声のトーンは中程度で、知的で信頼感のある話し方でお願いします。",
    },
    {
        "name": "pattern03_teen_energetic",
        "caption": "元気なティーンエイジャーの女の子の声で、テンポよく明るく話してください。少し高めの声で、活発でフレンドリーな雰囲気でお願いします。",
    },
    {
        "name": "pattern04_mature_gentle",
        "caption": "穏やかで優しい大人の女性の声で、ゆったりとした口調で話してください。温かみがあり、包容力を感じる話し方でお願いします。",
    },
    {
        "name": "pattern05_soft_whisper",
        "caption": "柔らかく囁くような女性の声で、静かに優しく話してください。息が多めで、親密感のあるトーンでお願いします。",
    },
    {
        "name": "pattern06_cheerful_high",
        "caption": "高くてキラキラした声の女の子で、はっきりとした発音で元気よく話してください。笑顔が伝わるような明るいトーンでお願いします。",
    },
    {
        "name": "pattern07_professional",
        "caption": "プロフェッショナルな女性アナウンサーのような声で、明瞭に正確に話してください。落ち着いた中音域で、聞き取りやすい話し方でお願いします。",
    },
    {
        "name": "pattern08_sweet_cute",
        "caption": "甘くて可愛い声の少女で、ゆっくりとした話し方でお願いします。少しはにかんだような、恥ずかしそうな雰囲気で話してください。",
    },
    {
        "name": "pattern09_lively_fast",
        "caption": "活発で声が大きめの女の子で、早口で生き生きと話してください。興奮気味で楽しそうな雰囲気のトーンでお願いします。",
    },
    {
        "name": "pattern10_deep_calm",
        "caption": "少し低めの落ち着いた女性の声で、ゆっくりと堂々と話してください。深みのある声で、威厳と優しさを兼ね備えた話し方でお願いします。",
    },
]


def generate(caption: str, out_wav: str) -> int:
    payload = json.dumps({"text": TEXT, "caption": caption}).encode()
    req = urllib.request.Request(
        API_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = resp.read()

    # API出力（24kHz）→ 48kHz モノ に変換
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(data)
        tmp_path = tmp.name

    result = subprocess.run(
        [
            "ffmpeg", "-y", "-i", tmp_path,
            "-ar", "48000", "-ac", "1",
            out_wav,
        ],
        capture_output=True,
    )
    os.unlink(tmp_path)

    if result.returncode != 0:
        print(f"  [ERROR] ffmpeg: {result.stderr[-200:]}", flush=True)
        return 0

    size = os.path.getsize(out_wav)
    return size


def main():
    print(f"出力先: {OUT_DIR}", flush=True)
    print(f"テキスト: {TEXT[:40]}...", flush=True)
    print("=" * 60, flush=True)

    for i, pat in enumerate(PATTERNS, 1):
        out_path = os.path.join(OUT_DIR, f"{pat['name']}.wav")
        print(f"[{i:02d}/10] {pat['name']} ...", end=" ", flush=True)
        try:
            size = generate(pat["caption"], out_path)
            # duration estimate: WAV = header(44) + PCM(size-44), 48kHz 16bit mono
            dur = (size - 44) / (48000 * 2) if size > 44 else 0
            print(f"OK  {size:,} bytes  ~{dur:.1f}秒", flush=True)
        except Exception as e:
            print(f"FAIL: {e}", flush=True)

    print("=" * 60, flush=True)
    print("完了。outputs/ を確認してください。", flush=True)


if __name__ == "__main__":
    main()
