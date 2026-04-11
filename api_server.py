#!/usr/bin/env python3
"""Irodori-TTS FastAPI server for AI agent TTS."""
from __future__ import annotations

import asyncio
import io
import re
import secrets
import subprocess
import tempfile
from contextlib import asynccontextmanager

import numpy as np
import soundfile as sf
import torch
import torchaudio
from fastapi import FastAPI
from fastapi.responses import Response
from huggingface_hub import hf_hub_download
from pydantic import BaseModel, Field

from irodori_tts.inference_runtime import (
    InferenceRuntime,
    RuntimeKey,
    SamplingRequest,
    default_runtime_device,
)

HF_REPO = "Aratako/Irodori-TTS-500M-v2"
REF_WAV = "/home/mnadmin/Irodori-TTS/input/teen_001.wav"
PORT = 50032

_runtime: InferenceRuntime | None = None
_lock: asyncio.Lock | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _runtime, _lock
    _lock = asyncio.Lock()

    print("[api] Downloading checkpoint...", flush=True)
    checkpoint_path = hf_hub_download(repo_id=HF_REPO, filename="model.safetensors")
    print(f"[api] Checkpoint: {checkpoint_path}", flush=True)

    device = default_runtime_device()
    print(f"[api] Loading model on {device}...", flush=True)
    _runtime = InferenceRuntime.from_key(
        RuntimeKey(
            checkpoint=checkpoint_path,
            model_device=device,
            codec_device=device,
            model_precision="bf16" if device == "cuda" else "fp32",
            codec_precision="bf16" if device == "cuda" else "fp32",
        )
    )
    print("[api] Model loaded. Server ready.", flush=True)
    yield
    _runtime = None
    _lock = None


app = FastAPI(title="Irodori-TTS API", lifespan=lifespan)


def split_text(text: str, max_chars: int = 80) -> list[str]:
    """テキストをスマートチャンキングで分割する。

    1. 句点（。！？\\n）で分割
    2. max_chars超の文は読点（、）やカンマ（,）で再分割
    3. strip()で空白除去、空文字除外
    """
    if not text or not text.strip():
        return []

    # Step 1: 句点・感嘆符・疑問符・改行で分割（区切り文字を保持）
    primary_pattern = re.compile(r"(?<=[。！？\n])")
    sentences = primary_pattern.split(text)

    chunks: list[str] = []
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        if len(sentence) <= max_chars:
            chunks.append(sentence)
        else:
            # Step 2: 読点・カンマで再分割
            secondary_pattern = re.compile(r"(?<=[、,])")
            sub_sentences = secondary_pattern.split(sentence)
            for sub in sub_sentences:
                sub = sub.strip()
                if sub:
                    chunks.append(sub)

    return chunks


class GenerateRequest(BaseModel):
    text: str = Field(..., max_length=2000)
    seed: int | None = Field(None, description="乱数シード。Noneの場合はランダム")


@app.post("/generate")
async def generate(req: GenerateRequest):
    if _runtime is None or _lock is None:
        return Response(content="Model not loaded", status_code=503)

    chunks = split_text(req.text)
    if not chunks:
        return Response(content="Empty text", status_code=400)

    # リクエスト単位でseedを1つ決定（全チャンク共通）
    used_seed = req.seed if req.seed is not None else int(secrets.randbits(63))
    print(f"[api] request seed: {used_seed}", flush=True)

    async with _lock:
        audio_segments: list[torch.Tensor] = []
        sample_rate: int = 24000  # fallback

        for chunk in chunks:
            try:
                result = _runtime.synthesize(
                    SamplingRequest(
                        text=chunk,
                        ref_wav=REF_WAV,
                        num_steps=20,
                        seconds=10.0,
                        seed=used_seed,
                    ),
                    log_fn=lambda msg: print(msg, flush=True),
                )
            except Exception as exc:
                print(f"[api] synthesize failed: {exc}", flush=True)
                return Response(content="Internal inference error", status_code=500)
            audio_segments.append(result.audio)
            sample_rate = result.sample_rate

        # チャンク間に0.3秒無音を挿入して結合
        silence_frames = int(sample_rate * 0.3)
        # result.audio shape: (channels, samples) or (samples,)
        if audio_segments[0].dim() == 1:
            silence = torch.zeros(silence_frames)
        else:
            channels = audio_segments[0].shape[0]
            silence = torch.zeros(channels, silence_frames)

        merged_parts: list[torch.Tensor] = []
        for i, seg in enumerate(audio_segments):
            merged_parts.append(seg)
            if i < len(audio_segments) - 1:
                merged_parts.append(silence)

        dim = 0 if audio_segments[0].dim() == 1 else 1
        merged = torch.cat(merged_parts, dim=dim)

        # 1次元の場合はtorchaudioのため2次元に変換
        if merged.dim() == 1:
            merged = merged.unsqueeze(0)

        buf = io.BytesIO()
        audio_np = merged.cpu().numpy().T if merged.dim() > 1 else merged.cpu().numpy()
        sf.write(buf, audio_np, sample_rate, format='WAV')
        buf.seek(0)

    # 1.2倍速変換（atempoフィルタ、pitch維持）
    wav_bytes = _apply_atempo(buf.read(), speed=1.2)
    return Response(content=wav_bytes, media_type="audio/wav")


def _apply_atempo(wav_bytes: bytes, speed: float = 1.2) -> bytes:
    """ffmpeg atempoフィルタで速度変換。失敗時は元データを返す。"""
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as fin:
            fin.write(wav_bytes)
            fin_path = fin.name
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as fout:
            fout_path = fout.name
        result = subprocess.run(
            [
                "ffmpeg", "-y", "-i", fin_path,
                "-filter:a", f"atempo={speed}",
                "-ar", "24000",
                fout_path,
            ],
            capture_output=True,
            timeout=30,
        )
        if result.returncode == 0:
            with open(fout_path, "rb") as f:
                return f.read()
        print(f"[api] ffmpeg atempo failed: {result.stderr[-200:]}", flush=True)
    except Exception as exc:
        print(f"[api] _apply_atempo error: {exc}", flush=True)
    finally:
        import os
        for p in (fin_path, fout_path):
            try:
                os.unlink(p)
            except Exception:
                pass
    return wav_bytes


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": _runtime is not None}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=PORT)
