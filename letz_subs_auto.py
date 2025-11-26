#!/usr/bin/env python3
"""
letz_subs_auto.py (refactored)
- sanitizes word-level timestamps before chunking
- prints original sentence chunks and translated lines for debugging
- loads translation model in 8-bit (bitsandbytes) for faster inference on laptops
- uses fuzzy matching to collapse hallucinated word repetitions
- **NEW**: uses audio energy (silence detection) to filter hallucinations
"""
import os
import sys
import gc
import argparse
import shutil
import zipfile
import requests
import subprocess
import logging
import difflib
import glob
import re
import torch
import pandas as pd
import ffmpeg
import yt_dlp
import stable_whisper
from pathlib import Path
from audio_separator.separator import Separator
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

# Try to import WhisperHF wrapper
try:
    from stable_whisper import WhisperHF
except ImportError:
    from stable_whisper.whisper_word_level.hf_whisper import WhisperHF

# Translation model imports
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

###############################################################
###                    HELPER FUNCTIONS                     ###
###############################################################

def install_fonts():
    font_name = "PT Sans"
    try:
        result = subprocess.run(['fc-list', ':family'], capture_output=True, text=True)
        if font_name in result.stdout:
            return
    except Exception:
        pass
    font_dir = Path.home() / ".fonts"
    font_dir.mkdir(parents=True, exist_ok=True)
    if list(font_dir.glob("*PTSans-Bold.ttf")):
        return
    print(f"\n=== Font '{font_name}' not found. Downloading... ===")
    url = "https://font.download/dl/font/pt-sans-2.zip"
    try:
        r = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
        zip_path = "pt_sans.zip"
        with open(zip_path, 'wb') as f:
            f.write(r.content)
        with zipfile.ZipFile(zip_path, 'r') as z:
            for file in z.namelist():
                if file.lower().endswith(".ttf"):
                    with open(font_dir / os.path.basename(file), "wb") as t:
                        shutil.copyfileobj(z.open(file), t)
        os.remove(zip_path)
        os.system("fc-cache -f -v > /dev/null 2>&1")
    except Exception as e:
        print(f"⚠ Font warning: {e}")

def download_video(url, base_name):
    print(f"\n=== Downloading Video: {base_name} ===")
    final_filename = f"{base_name}.mp4"
    ydl_opts = {
        'format': 'bestvideo+bestaudio/best',
        'outtmpl': f"{base_name}.%(ext)s",
        'merge_output_format': 'mp4',
        'quiet': True,
        'no_warnings': True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    if os.path.exists(final_filename):
        return final_filename

    for ext in ['.mp4', '.mkv', '.webm', '.avi', '.mov']:
        candidate = f"{base_name}{ext}"
        if os.path.exists(candidate):
            print(f"   -> Found video file: {candidate}")
            return candidate

    candidates = glob.glob(f"{base_name}.*")
    if candidates:
        print(f"   -> Found video file: {candidates[0]}")
        return candidates[0]

    print(f"ERROR: Could not find downloaded video file for {base_name}")
    sys.exit(1)

def extract_audio(video_path, audio_output_path):
    print(f"\n=== Extracting Audio to {audio_output_path} ===")

    if not os.path.exists(video_path):
        print(f"ERROR: Video file not found: {video_path}")
        print(f"   Current directory: {os.getcwd()}")
        print(f"   Files in directory: {os.listdir('.')}")
        sys.exit(1)

    print(f"   -> Input video: {video_path} ({os.path.getsize(video_path)} bytes)")

    try:
        process = (
            ffmpeg
            .input(video_path)
            .output(audio_output_path, acodec='pcm_s16le', ar='16000', ac=1)
            .overwrite_output()
            .run(capture_stdout=True, capture_stderr=True)
        )
        print(f"   -> Audio extracted successfully: {audio_output_path}")
    except ffmpeg.Error as e:
        stdout_msg = e.stdout.decode('utf8') if e.stdout else "No stdout"
        stderr_msg = e.stderr.decode('utf8') if e.stderr else "No stderr"
        print(f"FFmpeg Error:")
        print(f"   stdout: {stdout_msg}")
        print(f"   stderr: {stderr_msg}")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error during audio extraction: {type(e).__name__}: {e}")
        sys.exit(1)

def burn_subtitles(video_path, top_srt, bottom_srt, output_path):
    print(f"\n=== Burning Subtitles to {output_path} ===")

    for fpath, desc in [(video_path, "Video"), (top_srt, "Top SRT"), (bottom_srt, "Bottom SRT")]:
        if not os.path.exists(fpath):
            print(f"ERROR: {desc} file not found: {fpath}")
            sys.exit(1)

    top_abs = os.path.abspath(top_srt)
    bot_abs = os.path.abspath(bottom_srt)
    style_top = "Fontname=PT Sans,Fontsize=24,Bold=1,PrimaryColour=&H00FFFFFF,Alignment=2,MarginV=70"
    style_bot = "Fontname=PT Sans,Fontsize=18,Bold=1,PrimaryColour=&H00C0C0C0,Alignment=2,MarginV=20"

    try:
        v = ffmpeg.input(video_path)
        a = v.audio
        v = ffmpeg.filter(v.video, 'subtitles', top_abs, force_style=style_top)
        v = ffmpeg.filter(v, 'subtitles', bot_abs, force_style=style_bot)
        (
            ffmpeg
            .output(v, a, output_path, vcodec='libx264', acodec='aac', audio_bitrate='192k')
            .overwrite_output()
            .run(capture_stdout=True, capture_stderr=True)
        )
        print(f"✓ Final video: {output_path}")
    except ffmpeg.Error as e:
        stdout_msg = e.stdout.decode('utf8') if e.stdout else "No stdout"
        stderr_msg = e.stderr.decode('utf8') if e.stderr else "No stderr"
        print(f"Burning Error:")
        print(f"   stdout: {stdout_msg}")
        print(f"   stderr: {stderr_msg}")

def clean_memory():
    gc.collect()
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass

def get_max_volume(audio_path, start, end):
    """
    Uses ffmpeg volumedetect to find max volume in dB for a segment.
    Returns float dB (e.g. -14.5).
    Returns 0.0 on error (fail-safe to assume loud).
    """
    try:
        if end - start < 0.1:
            # Too short to reliably measure, assume signal to be safe
            return 0.0

        cmd = [
            'ffmpeg', '-nostdin',
            '-ss', f"{start:.3f}",
            '-to', f"{end:.3f}",
            '-i', audio_path,
            '-af', 'volumedetect',
            '-f', 'null',
            '/dev/null'
        ]
        # Capture stderr where volumedetect prints info
        res = subprocess.run(cmd, capture_output=True, text=True)
        # Parse: [Parsed_volumedetect_0 @ ...] max_volume: -45.2 dB
        m = re.search(r"max_volume:\s+([-\d\.]+)\s+dB", res.stderr)
        if m:
            return float(m.group(1))
    except Exception:
        pass
    return 0.0

###############################################################
###           SANITIZE & CHUNKING HELPERS                    ###
###############################################################

def sanitize_and_flatten_words(segments, audio_path=None, min_gap=0.02, max_word_duration=5.0, max_consecutive_repeats=2, similarity_threshold=0.75):
    """
    Return a cleaned word_list from Whisper segments.
    Integrates silence detection if audio_path is provided.
    """

    # --- 1. Flatten & Basic Clean ---
    word_list = []
    for seg in segments:
        seg_words = seg.get("words") or []
        if seg_words:
            for w in seg_words:
                word = (w.get("word") or "").strip()
                if not word: continue
                start = float(w.get("start", seg.get("start", 0.0)))
                end = float(w.get("end", seg.get("end", start)))
                if end - start > max_word_duration:
                    end = start + max_word_duration
                word_list.append({"word": word, "start": start, "end": end})
        else:
            txt = (seg.get("text") or "").strip()
            if txt:
                start = float(seg.get("start", 0.0))
                end = float(seg.get("end", start))
                if end - start > max_word_duration:
                    end = start + max_word_duration
                word_list.append({"word": txt, "start": start, "end": end})

    word_list.sort(key=lambda x: (x["start"], x["end"]))

    cleaned = []
    for w in word_list:
        if not cleaned:
            cleaned.append(w)
            continue
        prev = cleaned[-1]
        # Overlap/Duplicate handling
        if w["word"] == prev["word"] and abs(w["start"] - prev["start"]) < 0.05:
            if w["end"] > prev["end"]: prev["end"] = w["end"]
            continue
        if w["end"] <= w["start"] or w["end"] <= prev["start"]:
            prev["end"] = max(prev["end"], w["end"], prev["start"] + 0.01)
            continue
        # Merge tiny gaps
        gap = w["start"] - prev["end"]
        if gap < min_gap and gap >= 0:
            w["start"] = prev["end"]
        cleaned.append(w)

    # --- 2. Collapse Repeats (Before Filtering) ---
    def words_are_similar(w1, w2, threshold=similarity_threshold):
        if w1 == w2: return True
        w1_clean = w1.lower().strip('.,!?;:"\'"')
        w2_clean = w2.lower().strip('.,!?;:"\'"')
        if w1_clean == w2_clean: return True
        return difflib.SequenceMatcher(None, w1_clean, w2_clean).ratio() >= threshold

    collapsed = []
    i = 0
    while i < len(cleaned):
        current = cleaned[i]
        repeat_count = 1
        last_end = current["end"]
        j = i + 1
        while j < len(cleaned) and words_are_similar(current["word"], cleaned[j]["word"]):
            repeat_count += 1
            last_end = max(last_end, cleaned[j]["end"])
            j += 1

        if repeat_count > max_consecutive_repeats:
            kept_word = {
                "word": current["word"],
                "start": current["start"],
                "end": last_end,
                "_collapsed_repeats": repeat_count
            }
            collapsed.append(kept_word)
            print(f"   [COLLAPSE] '{current['word']}' repeated {repeat_count}x @ {current['start']:.2f}s -> collapsed")
        else:
            for k in range(i, j):
                collapsed.append(cleaned[k])
        i = j

    cleaned = collapsed

    # --- 3. Filter Start/End Hallucinations (Lookahead + Silence Detection) ---

    BAD_REPEAT_THRESHOLD = 3
    BAD_DURATION_THRESHOLD = 6.0
    STRICT_START_GAP = 0.5
    SILENCE_GAP_THRESHOLD = 1.2

    # Silence threshold in dB. Isolated vocals should be very clean, so -45dB is a safe noise floor.
    SILENCE_DB_THRESHOLD = -45.0

    # --- Filter Start (Lookahead) ---
    while len(cleaned) > 0:
        words_removed = False
        scan_limit = min(len(cleaned), 5)
        cut_index = -1

        # 1. Repeats/Duration Check
        for k in range(scan_limit):
            w = cleaned[k]
            repeats = w.get("_collapsed_repeats", 1)
            duration = w["end"] - w["start"]

            if repeats > BAD_REPEAT_THRESHOLD:
                print(f"   [FILTER START] Found high-repeat word '{w['word']}' at pos {k}. Removing preceding block.")
                cut_index = k
                break
            if duration > BAD_DURATION_THRESHOLD:
                print(f"   [FILTER START] Found long-duration word '{w['word']}' at pos {k}. Removing preceding block.")
                cut_index = k
                break

        if cut_index != -1:
            cleaned = cleaned[cut_index+1:]
            words_removed = True
            continue

        # 2. Start Gap Check
        if len(cleaned) > 1:
            gap = cleaned[1]["start"] - cleaned[0]["end"]
            if gap > STRICT_START_GAP:
                 print(f"   [FILTER START] Removing '{cleaned[0]['word']}' due to gap {gap:.2f}s > {STRICT_START_GAP}s.")
                 cleaned.pop(0)
                 words_removed = True
                 continue

        # 3. Silence Check (The Silver Bullet)
        if audio_path:
            w = cleaned[0]
            max_vol = get_max_volume(audio_path, w['start'], w['end'])
            if max_vol < SILENCE_DB_THRESHOLD:
                print(f"   [FILTER START] Removing '{w['word']}' @ {w['start']:.2f}s. Max Vol: {max_vol}dB (Silent).")
                cleaned.pop(0)
                words_removed = True
                continue

        if not words_removed:
            break

    # --- Filter End (Lookback) ---
    while len(cleaned) > 0:
        w = cleaned[-1]
        duration = w["end"] - w["start"]
        repeats = w.get("_collapsed_repeats", 1)

        gap_from_prev = 0.0
        if len(cleaned) > 1:
            gap_from_prev = w["start"] - cleaned[-2]["end"]

        should_remove = False
        reason = ""

        # Standard heuristics
        if repeats > BAD_REPEAT_THRESHOLD:
            should_remove = True
            reason = f"high repeats ({repeats})"
        elif duration > BAD_DURATION_THRESHOLD:
            should_remove = True
            reason = f"long duration ({duration:.1f}s)"
        elif gap_from_prev > SILENCE_GAP_THRESHOLD:
            should_remove = True
            reason = f"preceded by silence ({gap_from_prev:.1f}s)"

        # Silence Check
        if not should_remove and audio_path:
            max_vol = get_max_volume(audio_path, w['start'], w['end'])
            if max_vol < SILENCE_DB_THRESHOLD:
                should_remove = True
                reason = f"Max Vol: {max_vol}dB (Silent)"

        if should_remove:
            print(f"   [FILTER END] Removing '{w['word']}' @ {w['start']:.2f}s -> {reason}")
            cleaned.pop()
        else:
            break

    return cleaned

def simple_sentence_chunking_lb_from_wordlist(word_list, min_words_for_comma=5):
    """
    Build sentence-like chunks from a sanitized word_list.
    """
    CAP_SILENCE_THRESHOLD = 0.5
    HARD_STOPS = ('.', '!', '?', ';', '…')
    COMMA = (',',)

    sentences = []
    current_words = []
    current_start = None
    last_end = None

    for i, w in enumerate(word_list):
        word = w["word"]

        should_split_on_silence = False
        if current_words and i > 0 and word and word[0].isupper():
            prev_end = word_list[i-1]["end"]
            gap = w["start"] - prev_end
            if gap > CAP_SILENCE_THRESHOLD:
                should_split_on_silence = True
                print(f"   [SPLIT SILENCE] Breaking before '{word}' at {w['start']:.2f}s due to long gap ({gap:.2f}s) and capitalization.")

        if should_split_on_silence:
            if current_words:
                text = " ".join([cw["word"] for cw in current_words]).strip()
                sentences.append({
                    "Text": text,
                    "Start": current_start,
                    "End": last_end,
                    "Words": [w.copy() for w in current_words],
                    "WordCount": len(current_words)
                })
                current_words = []
                current_start = None

        if current_start is None:
            current_start = w["start"]
        current_words.append(w)
        last_end = w["end"]

        is_last = (i == len(word_list) - 1)
        has_hard_stop = any(word.strip(' ').endswith(hs) for hs in HARD_STOPS)
        has_comma = any(word.strip(' ').endswith(c) for c in COMMA)

        should_break = False
        if is_last:
            should_break = True
        elif has_hard_stop:
            should_break = True
        elif has_comma and len(current_words) >= min_words_for_comma:
            should_break = True

        if should_break:
            text = " ".join([cw["word"] for cw in current_words]).strip()
            sentences.append({
                "Text": text,
                "Start": current_start,
                "End": last_end,
                "Words": [w.copy() for w in current_words],
                "WordCount": len(current_words)
            })
            current_words = []
            current_start = None

    if current_words:
        text = " ".join([cw["word"] for cw in current_words]).strip()
        sentences.append({
            "Text": text,
            "Start": current_start,
            "End": last_end,
            "Words": [w.copy() for w in current_words],
            "WordCount": len(current_words)
        })

    return sentences

def collapse_identical_chunks(sentences, max_repeats=2):
    out = []
    for s in sentences:
        if out and s['Text'] == out[-1]['Text'] and abs(s['Start'] - out[-1]['Start']) < 1e-6:
            if 'repeats' not in out[-1]:
                out[-1]['repeats'] = 1
            out[-1]['repeats'] += 1
            out[-1]['End'] = max(out[-1]['End'], s['End'])
            if out[-1]['repeats'] > max_repeats:
                continue
        else:
            out.append(s)
    return out

def adjust_translation_to_match_lines(translation_lines, target_word_counts):
    target_count = len(target_word_counts)
    lines = translation_lines.copy()

    def get_word_counts(lines):
        return [len(line.split()) for line in lines]

    def calculate_word_count_distance(trans_counts, rus_counts):
        if len(trans_counts) != len(rus_counts):
            return float('inf')
        return sum(abs(t - r) for t, r in zip(trans_counts, rus_counts))

    while len(lines) > target_count:
        best_merge_idx = None
        best_distance = float('inf')
        for i in range(len(lines) - 1):
            test_lines = lines.copy()
            test_lines[i] = test_lines[i] + " " + test_lines[i + 1]
            test_lines.pop(i + 1)
            if len(test_lines) == target_count:
                distance = calculate_word_count_distance(get_word_counts(test_lines), target_word_counts)
                if distance < best_distance:
                    best_distance = distance
                    best_merge_idx = i
        if best_merge_idx is None:
            lines[0] = lines[0] + " " + lines[1]
            lines.pop(1)
        else:
            lines[best_merge_idx] = lines[best_merge_idx] + " " + lines[best_merge_idx + 1]
            lines.pop(best_merge_idx + 1)

    while len(lines) < target_count:
        best_split_idx = None
        best_split_point = None
        best_distance = float('inf')
        found_valid_split = False

        for i in range(len(lines)):
            words = lines[i].split()
            if len(words) < 3:
                continue
            split_candidates = []
            for j, wd in enumerate(words):
                if j > 0 and j < len(words) and wd.endswith(','):
                    split_candidates.append(j + 1)
            if not split_candidates:
                for j, wd in enumerate(words):
                    if j > 1 and j < len(words) - 1 and wd.lower() in ['and', 'but', 'or', 'while', 'when', 'as', 'though', 'with', 'where', 'who', 'which', 'that']:
                        split_candidates.append(j)
            if not split_candidates:
                if len(words) >= 6:
                    split_candidates.extend([len(words) // 3, len(words) // 2, (2 * len(words)) // 3])
                elif len(words) >= 3:
                    split_candidates.append(len(words) // 2)

            for sp in split_candidates:
                if sp <= 0 or sp >= len(words):
                    continue
                test_lines = lines.copy()
                first_half = ' '.join(words[:sp])
                second_half = ' '.join(words[sp:])
                test_lines[i] = first_half
                test_lines.insert(i + 1, second_half)
                if len(test_lines) == target_count:
                    distance = calculate_word_count_distance([len(x.split()) for x in test_lines], target_word_counts)
                    if distance < best_distance:
                        best_distance = distance
                        best_split_idx = i
                        best_split_point = sp
                        found_valid_split = True
        if found_valid_split and best_split_point:
            words = lines[best_split_idx].split()
            first_half = ' '.join(words[:best_split_point])
            second_half = ' '.join(words[best_split_point:])
            lines[best_split_idx] = first_half
            lines.insert(best_split_idx + 1, second_half)
        else:
            break

    return lines

###############################################################
###                       MAIN LOGIC                        ###
###############################################################

def main():
    parser = argparse.ArgumentParser(description="Luxembourgish Auto-Karaoke")
    parser.add_argument("project_name", help="Base name for files")
    parser.add_argument("url", help="YouTube URL or local path to video")
    args = parser.parse_args()

    project = args.project_name
    files = {
        'vid': f"{project}.mp4",
        'audio': f"{project}.wav",
        'srt_lb': f"{project}_lb.srt",
        'srt_en': f"{project}_en.srt",
        'final': f"{project}_subbed.mp4"
    }

    install_fonts()

    if os.path.exists(args.url):
        video_file = args.url
        print(f"\n=== Using local video file: {video_file} ===")
    else:
        video_file = download_video(args.url, project)

    if not os.path.exists(video_file):
        print(f"ERROR: Video file does not exist: {video_file}")
        sys.exit(1)

    print(f"   -> Video file confirmed: {video_file} ({os.path.getsize(video_file)} bytes)")

    extract_audio(video_file, files['audio'])

    print("\n=== Step 1: Vocal Isolation ===")
    separator = Separator(output_single_stem='vocals')
    separator.load_model(model_filename='model_bs_roformer_ep_317_sdr_12.9755.ckpt')
    isolated_path = separator.separate(files['audio'], {'Vocals': 'isolated_vocals'})[0]
    print(f"✓ Vocals isolated: {isolated_path}")
    del separator
    clean_memory()

    print("\n=== Step 2: Transcription (Unilux Whisper) ===")
    model_id = "unilux/whisper-medium-v1-luxembourgish"
    print("   -> Loading model manually to bypass library version conflict...")

    hf_model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id,
        device_map="cuda" if torch.cuda.is_available() else None,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        low_cpu_mem_usage=True,
        use_safetensors=True
    )
    processor = AutoProcessor.from_pretrained(model_id)
    pipe = pipeline(
        "automatic-speech-recognition",
        model=hf_model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="cuda" if torch.cuda.is_available() else None,
    )
    model = WhisperHF(model_id, pipeline=pipe)
    print("   -> Transcribing...")
    result = model.transcribe(isolated_path, language='lb', regroup=True, suppress_silence=True, vad=False)
    segments = result.to_dict().get('segments', [])
    del model, pipe, hf_model, processor
    clean_memory()
    print(f"   -> Generated {len(segments)} whisper segments.")

    print("\n=== Step 3: Sanitize word-level timestamps & chunk into sentences ===")
    # Pass the isolated audio path to the sanitizer for silence detection
    word_list = sanitize_and_flatten_words(segments, audio_path=isolated_path)
    print(f"   -> Word tokens after sanitization: {len(word_list)}")

    sentences = simple_sentence_chunking_lb_from_wordlist(word_list)
    sentences = collapse_identical_chunks(sentences, max_repeats=2)

    if not sentences:
        sentences = [{
            'Text': s.get('text', '').strip(),
            'Start': s.get('start', 0.0),
            'End': s.get('end', 0.0),
            'Words': s.get('words', []),
            'WordCount': len(s.get('text', '').split())
        } for s in segments]

    print("\n--- DEBUG: Original sentence chunks ---")
    for idx, s in enumerate(sentences):
        print(f"[{idx+1}] {s['Start']:.2f}-{s['End']:.2f} | {s['Text']}")
    print("--- END DEBUG ---\n")

    source_lines = [s['Text'] for s in sentences]
    source_text_block = "\n".join(source_lines)
    num_lines = len(source_lines)
    print(f"   -> Created {num_lines} sentence chunks for translation.")

    print("\n=== Step 4: Translation (8-bit quant) ===")
    tr_model_id = "etamin/Letz-MT-gemma2-2b-lb-en"
    tokenizer = AutoTokenizer.from_pretrained(tr_model_id)

    bnb_config = BitsAndBytesConfig(load_in_8bit=True, llm_int8_threshold=6.0)
    llm = AutoModelForCausalLM.from_pretrained(
        tr_model_id,
        device_map="auto",
        quantization_config=bnb_config,
        trust_remote_code=False
    )

    target_word_counts = [s['WordCount'] for s in sentences]
    translated_lines = []
    attempt = 0
    max_retries = 12

    def parse_translated_output(text):
        candidates = [line.strip() for line in text.split('\n') if line.strip()]
        cleaned = []
        for line in candidates:
            if len(line) > 2 and line[0].isdigit():
                for sep in ['. ', ')', ']']:
                    if sep in line[:5]:
                        parts = line.split(sep, 1)
                        if len(parts) > 1:
                            line = parts[1].strip()
                            break
            cleaned.append(line)
        return cleaned

    while attempt < max_retries:
        attempt += 1
        temp = 0.20 + attempt * 0.05
        print(f"   -> Translation attempt {attempt} (temp={temp:.2f})...")
        prompt_content = (
            f"You are an award-winning bilingual Luxembourgish-English literary translator specializing in songs and poetry. "
            f"Translate this Luxembourgish song to English while preserving meter, syllable counts, rhyme scheme, and poetic qualities.\n"
            f"Maintain a one-to-one verse correspondence: the output must have EXACTLY {num_lines} lines.\n"
            f"Do not add any intro or outro text. Just the translation.\n\n"
            f"{source_text_block}"
        )
        messages = [{"role": "user", "content": prompt_content}]
        input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True).to(llm.device)
        with torch.inference_mode():
            outputs = llm.generate(input_ids, max_new_tokens=1024, do_sample=True, temperature=temp, top_p=0.9)
        response = outputs[0][input_ids.shape[-1]:]
        decoded_text = tokenizer.decode(response, skip_special_tokens=True).strip()
        candidates = parse_translated_output(decoded_text)
        if len(candidates) == num_lines:
            translated_lines = candidates
            print(f"   -> ✓ Success: Got exactly {num_lines} lines on attempt {attempt}.")
            break
        else:
            print(f"   -> Mismatch: Got {len(candidates)} lines (need {num_lines}).")

    if not translated_lines:
        print("   -> Batched attempts failed. Trying per-line deterministic translation...")
        candidates = []
        for sline in source_lines:
            prompt = f"Translate this Luxembourgish line to English (single line, no numbering):\n\n{sline}"
            messages = [{"role": "user", "content": prompt}]
            input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True).to(llm.device)
            with torch.inference_mode():
                outs = llm.generate(input_ids, max_new_tokens=128, do_sample=False)
            out = tokenizer.decode(outs[0][input_ids.shape[1]:], skip_special_tokens=True).strip()
            candidates.append(out)
        if len(candidates) == num_lines:
            translated_lines = candidates
            print("   -> ✓ Per-line translation succeeded.")
        else:
            print("   -> Per-line fallback did not perfectly match; attempting heuristic adjustment...")
            adjusted = adjust_translation_to_match_lines(candidates, target_word_counts)
            if len(adjusted) == num_lines:
                translated_lines = adjusted
                print("   -> ✓ Heuristic adjustment successful.")
            else:
                print("   -> Falling back to best-effort mapping (pad/trim).")
                translated_lines = candidates[:num_lines]
                while len(translated_lines) < num_lines:
                    translated_lines.append("(Translation Error)")

    print("\n--- DEBUG: Translated lines ---")
    for idx, tl in enumerate(translated_lines):
        print(f"[{idx+1}] {tl}")
    print("--- END TRANSLATION DEBUG ---\n")

    del llm, tokenizer
    clean_memory()

    print("\n=== Step 5: generating SRTs ===")
    def format_ts(seconds):
        ms = int((seconds - int(seconds)) * 1000)
        m, s = divmod(int(seconds), 60)
        h, m = divmod(m, 60)
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

    HIGHLIGHT = "#DAA425"
    if os.path.exists(files['srt_lb']):
        os.remove(files['srt_lb'])
    if os.path.exists(files['srt_en']):
        os.remove(files['srt_en'])

    with open(files['srt_en'], 'w', encoding='utf-8') as f_en:
        for i, s in enumerate(sentences):
            f_en.write(f"{i+1}\n")
            f_en.write(f"{format_ts(s['Start'])} --> {format_ts(s['End'])}\n")
            f_en.write(f"{translated_lines[i]}\n\n")

    karaoke_entries = []
    for sent in sentences:
        words = sent.get('Words', [])
        if not words:
            karaoke_entries.append((sent['Start'], sent['End'], sent['Text']))
            continue
        for j in range(len(words)):
            w_start = words[j]['start']
            if j < len(words) - 1:
                w_end = words[j + 1]['start']
            else:
                w_end = sent['End']
            line_parts = []
            for k, w in enumerate(words):
                txt = w['word'].strip()
                if k == j:
                    line_parts.append(f'<font color="{HIGHLIGHT}">{txt}</font>')
                else:
                    line_parts.append(txt)
            text_payload = " ".join(line_parts)
            karaoke_entries.append((w_start, w_end, text_payload))

    with open(files['srt_lb'], 'w', encoding='utf-8') as f:
        for idx, (s, e, txt) in enumerate(karaoke_entries):
            f.write(f"{idx+1}\n")
            f.write(f"{format_ts(s)} --> {format_ts(e)}\n")
            f.write(f"{txt}\n\n")

    print(f"✓ SRTs generated: {files['srt_lb']} & {files['srt_en']}")

    burn_subtitles(video_file, files['srt_lb'], files['srt_en'], files['final'])

if __name__ == "__main__":
    main()
