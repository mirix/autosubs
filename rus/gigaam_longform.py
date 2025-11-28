#!/usr/bin/env python3
"""
GigaAM Long-Form Transcription + Word-Level Alignment
Automatic Short/Long Pipeline Switching
**Now using sound-activity based chunking instead of Silero VAD.**

Audio I/O via librosa + soundfile. No torchaudio / torchcodec.
"""

import os
from typing import List, Dict, Tuple

import numpy as np
import torch
import soundfile as sf
import librosa

import gigaam
from gigaam.preprocess import SAMPLE_RATE


###############################################################
#                CONSTANTS / THRESHOLDS
###############################################################

SHORT_THRESHOLD_SEC = 25.0     # short-mode direct alignment


###############################################################
#          ORIGINAL GIGAAM ALIGNMENT IMPLEMENTATION
###############################################################

def _decode_with_alignment_rnnt(head, encoded_seq, seq_len, blank_id, max_symbols):
    hyp = []
    token_frames = []
    dec_state = None
    last_label = None

    for t in range(seq_len):
        encoder_step = encoded_seq[t, :, :].unsqueeze(1)
        not_blank = True
        emitted = 0

        while not_blank and emitted < max_symbols:
            decoder_step, hidden = head.decoder.predict(last_label, dec_state)
            joint = head.joint.joint(encoder_step, decoder_step)[0, 0, 0, :]
            k = int(torch.argmax(joint).item())

            if k == blank_id:
                not_blank = False
                continue

            hyp.append(k)
            token_frames.append(t)

            last_label = torch.tensor([[k]], dtype=torch.long, device=encoded_seq.device)
            dec_state = hidden
            emitted += 1

    return hyp, token_frames


def _decode_with_alignment_ctc(head, encoded_seq, seq_len, blank_id):
    log_probs = head(encoder_output=encoded_seq)
    labels = log_probs.argmax(dim=-1)
    frame_labels = labels[0, :seq_len].cpu().tolist()

    hyp, token_frames = [], []
    prev = blank_id

    for t, lbl in enumerate(frame_labels):
        if lbl != blank_id and (lbl != prev or prev == blank_id):
            hyp.append(lbl)
            token_frames.append(t)
        prev = lbl

    return hyp, token_frames


def _get_token_str(tok, token_id: int) -> str:
    return tok.vocab[token_id] if tok.charwise else tok.model.IdToPiece(token_id)


def _chars_to_words(chars, frames, frame_shift):
    words = []
    curr_chars, curr_frames = [], []

    def push():
        if not curr_chars:
            return
        text = "".join(curr_chars).strip()
        if text:
            words.append({
                "word": text,
                "start": curr_frames[0] * frame_shift,
                "end":   (curr_frames[-1] + 1) * frame_shift,
            })
        curr_chars.clear()
        curr_frames.clear()

    for c, f in zip(chars, frames):
        if c == " " or c.startswith("▁"):
            push()
            c = c.lstrip("▁")
            if c:
                curr_chars.append(c)
                curr_frames.append(f)
            continue
        curr_chars.append(c)
        curr_frames.append(f)

    push()
    return words


@torch.inference_mode()
def extract_word_timestamps(model, audio_path: str, max_symbols_per_step=3):
    """
    Uses the existing GigaAM model API: model.prepare_wav(audio_path) etc.
    """
    wav, length = model.prepare_wav(audio_path)
    encoded, encoded_len = model.forward(wav, length)

    seq_len = int(encoded_len[0].item())
    frame_shift = float(length[0].item()) / SAMPLE_RATE / seq_len

    blank_id = model.decoding.blank_id
    tokenizer = model.decoding.tokenizer
    head = model.head

    is_rnnt = hasattr(head, "decoder") and hasattr(head, "joint")

    if is_rnnt:
        encoded_seq = encoded.transpose(1, 2)[0, :, :].unsqueeze(1)
        token_ids, token_frames = _decode_with_alignment_rnnt(
            head, encoded_seq, seq_len, blank_id, max_symbols_per_step
        )
    else:
        token_ids, token_frames = _decode_with_alignment_ctc(
            head, encoded, seq_len, blank_id
        )

    chars = [_get_token_str(tokenizer, i) for i in token_ids]
    word_segments = _chars_to_words(chars, token_frames, frame_shift)

    return {
        "transcript": tokenizer.decode(token_ids).strip(),
        "frame_shift": frame_shift,
        "words": word_segments,
    }


###############################################################
#                   SEGMENT MERGING UTILITIES
###############################################################

def merge_word_segments(words, tol=0.15):
    if not words:
        return []

    words.sort(key=lambda w: w["start"])
    merged = [words[0]]

    for w in words[1:]:
        prev = merged[-1]
        if w["word"] == prev["word"] and w["start"] <= prev["end"] + tol:
            continue
        merged.append(w)

    return merged

###############################################################
#       NEW: SOUND-ACTIVITY BASED LONG-FORM CHUNKING
###############################################################

def extract_longform_soundactivity(model, audio_path):
    """
    Sound-activity long-form slicing.
    Fixed to avoid:
        - infinite loops
        - overlapping-chunk duplication
    Uses gap-based splitting instead of overlapping.
    """

    MAX_CHUNK_SEC = 24.5
    MIN_CHUNK_SEC = 3.0
    FRAME = 2048
    HOP = 512

    # slightly higher threshold to suppress micro-blips
    ENERGY_THRESHOLD = 0.025

    # boundary expansion to avoid chopping words
    EXPAND_SEC = 0.35

    # load audio
    y, sr = librosa.load(audio_path, sr=16000, mono=True)
    total_sec = len(y) / sr

    # RMS energy
    rms = librosa.feature.rms(y=y, frame_length=FRAME, hop_length=HOP)[0]
    times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=HOP)
    active = rms > ENERGY_THRESHOLD

    # gather raw active regions
    segments = []
    start = None
    for i, a in enumerate(active):
        if a and start is None:
            start = times[i]
        elif not a and start is not None:
            end = times[i]
            if end - start >= 0.12:
                segments.append({"start": start, "end": end})
            start = None
    if start is not None:
        end = total_sec
        if end - start >= 0.12:
            segments.append({"start": start, "end": end})

    # expand segments to avoid chopping
    for seg in segments:
        seg["start"] = max(0, seg["start"] - EXPAND_SEC)
        seg["end"]   = min(total_sec, seg["end"] + EXPAND_SEC)

    # merge touching or nearly-touching segments
    merged = []
    for seg in segments:
        if not merged:
            merged.append(seg)
            continue
        last = merged[-1]
        if seg["start"] <= last["end"] + 0.1:
            last["end"] = max(last["end"], seg["end"])
        else:
            merged.append(seg)

    # split long merged segments into independent, non-overlapping blocks
    final_segments = []
    for seg in merged:
        seg_start = seg["start"]
        seg_end = seg["end"]

        length = seg_end - seg_start

        # tiny → keep as-is
        if length <= MIN_CHUNK_SEC:
            final_segments.append(seg)
            continue

        # If long, chunk without overlap
        s = seg_start
        while s < seg_end:
            e = min(s + MAX_CHUNK_SEC, seg_end)

            # progress guarantee
            if e <= s + 0.05:
                break

            final_segments.append({"start": s, "end": e})

            s = e   # NO OVERLAP, FIXES DUPLICATION

    # === PROCESS EACH NON-OVERLAPPING CHUNK ===
    all_words = []

    for idx, seg in enumerate(final_segments):
        print(f"[Segment {idx+1}/{len(final_segments)}] {seg['start']:.2f}–{seg['end']:.2f}s")

        s_i = int(seg["start"] * sr)
        e_i = int(seg["end"] * sr)

        chunk = y[s_i:e_i]
        chunk_path = "_tmp_chunk.wav"
        sf.write(chunk_path, chunk, sr)

        out = extract_word_timestamps(model, chunk_path)

        for w in out["words"]:
            all_words.append({
                "word":  w["word"],
                "start": w["start"] + seg["start"],
                "end":   w["end"]   + seg["start"],
            })

    # merge short duplicates (same word, tiny gap)
    final_words = merge_word_segments(all_words)

    return {
        "words": final_words,
        "transcript": " ".join(w["word"] for w in final_words),
        "segments": final_segments
    }

###############################################################
#                 AUTOMATIC SHORT / LONG SWITCH
###############################################################

def transcribe_auto(model, audio_path):
    """
    Auto mode that selects short or long-form pipeline based on duration.
    """
    duration_sec = float(librosa.get_duration(path=audio_path))

    if duration_sec <= SHORT_THRESHOLD_SEC:
        print(f"Short audio detected ({duration_sec:.2f}s) → direct alignment.")
        return extract_word_timestamps(model, audio_path)

    print(f"Long audio detected ({duration_sec:.2f}s) → sound-activity long-form pipeline.")
    return extract_longform_soundactivity(model, audio_path)


###############################################################
#                           MAIN
###############################################################

if __name__ == "__main__":
    model = gigaam.load_model("v3_e2e_rnnt", device="cuda")

    result = transcribe_auto(model, "long_example.wav")

    print("\n=== FINAL WORD TIMESTAMPS ===\n")
    for w in result["words"]:
        print(f"{w['start']:.2f} – {w['end']:.2f}\t{w['word']}")
