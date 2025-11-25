import os
import sys
import gc
import argparse
import shutil
import zipfile
import requests
import subprocess
import logging
import torch
import librosa
import pandas as pd
import ffmpeg  # pip install ffmpeg-python
import yt_dlp
from pathlib import Path

# --- SILENCE NEMO WARNINGS ---
from nemo.utils import logging as nemo_logging
nemo_logging.setLevel(logging.ERROR)
logging.getLogger("nemo_logger").setLevel(logging.ERROR)
# -----------------------------

import nemo.collections.asr as nemo_asr
from sentence_transformers import SentenceTransformer, util
from audio_separator.separator import Separator
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

###############################################################
###                    HELPER FUNCTIONS                     ###
###############################################################

def install_fonts():
    """
    Checks for PT Sans font using system tools (fc-list).
    If missing, downloads from the user-provided source using a browser user-agent.
    """
    font_name = "PT Sans"

    # 1. Robust System Check
    try:
        result = subprocess.run(['fc-list', ':family'], capture_output=True, text=True)
        if font_name in result.stdout:
            print(f"✓ Font '{font_name}' detected in system configuration.")
            return
    except Exception:
        pass

    # 2. Fallback File Check
    font_dir = Path.home() / ".fonts"
    font_dir.mkdir(parents=True, exist_ok=True)

    if list(font_dir.glob("*PTSans-Bold.ttf")) or list(font_dir.glob("*PT_Sans-Web-Bold.ttf")):
         print(f"✓ Font '{font_name}' found in {font_dir}.")
         return

    # 3. Download
    print(f"\n=== Font '{font_name}' not found. Downloading... ===")
    url = "https://font.download/dl/font/pt-sans-2.zip"

    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        r = requests.get(url, headers=headers)
        r.raise_for_status()

        zip_path = "pt_sans.zip"
        with open(zip_path, 'wb') as f:
            f.write(r.content)

        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            for file in zip_ref.namelist():
                if file.lower().endswith(".ttf"):
                    source = zip_ref.open(file)
                    filename = os.path.basename(file)
                    target_path = font_dir / filename
                    with open(target_path, "wb") as target:
                        shutil.copyfileobj(source, target)

        os.remove(zip_path)
        print(f"✓ Fonts downloaded and installed to {font_dir}")
        os.system("fc-cache -f -v > /dev/null 2>&1")

    except Exception as e:
        print(f"⚠ Failed to download fonts: {e}")
        print("Proceeding, but ffmpeg might warn about missing fonts.")

def download_video(url, base_name):
    print(f"\n=== Downloading Video: {base_name} ===")
    output_template = f"{base_name}.%(ext)s"
    final_filename = f"{base_name}.mp4"

    ydl_opts = {
        'format': 'bestvideo+bestaudio/best',
        'outtmpl': output_template,
        'merge_output_format': 'mp4',
        'quiet': True,
        'no_warnings': True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    if not os.path.exists(final_filename):
        raise FileNotFoundError("Video download failed.")
    print(f"✓ Video saved to {final_filename}")
    return final_filename

def extract_audio(video_path, audio_output_path):
    print(f"\n=== Extracting Audio to {audio_output_path} ===")
    try:
        (
            ffmpeg
            .input(video_path)
            .output(audio_output_path, acodec='libopus', loglevel="quiet")
            .overwrite_output()
            .run()
        )
        print(f"✓ Audio extracted.")
    except ffmpeg.Error as e:
        print("FFmpeg Error:", e.stderr.decode('utf8'))
        sys.exit(1)

def burn_subtitles(video_path, rus_srt, eng_srt, output_path):
    print(f"\n=== Burning Subtitles to {output_path} ===")
    rus_srt_abs = os.path.abspath(rus_srt)
    eng_srt_abs = os.path.abspath(eng_srt)

    style_rus = "Fontname=PT Sans,Fontsize=24,Bold=1,PrimaryColour=&H00FFFFFF,Alignment=2,MarginV=70"
    style_eng = "Fontname=PT Sans,Fontsize=18,Bold=1,PrimaryColour=&H00C0C0C0,Alignment=2,MarginV=20"

    try:
        input_stream = ffmpeg.input(video_path)
        audio_stream = input_stream.audio
        video_stream = input_stream.video

        video_stream = ffmpeg.filter(video_stream, 'subtitles', rus_srt_abs, force_style=style_rus)
        video_stream = ffmpeg.filter(video_stream, 'subtitles', eng_srt_abs, force_style=style_eng)

        output = ffmpeg.output(
            video_stream, audio_stream, output_path,
            vcodec='libx264', acodec='aac', audio_bitrate='192k', loglevel="warning"
        )
        output.overwrite_output().run()
        print(f"✓ Final video created: {output_path}")
    except ffmpeg.Error as e:
        print("FFmpeg Error during burning:", e.stderr.decode('utf8'))

###############################################################
###                       MAIN LOGIC                        ###
###############################################################

def main():
    parser = argparse.ArgumentParser(description="Automated Karaoke Subtitle Generator")
    parser.add_argument("project_name", help="Base name for generated files")
    parser.add_argument("url", help="YouTube URL or Video ID")
    args = parser.parse_args()

    project_name = args.project_name
    url = args.url

    # Variables
    video_file = f"{project_name}.mp4"
    input_audio = f"{project_name}.opus"
    output_csv_path = f"{project_name}.csv"
    output_srt_rus = f"{project_name}_rus.srt"
    output_srt_eng = f"{project_name}_eng.srt"
    final_output_video = f"{project_name}_subbed.mp4"

    VOCAL_MODEL_FILENAME = 'model_bs_roformer_ep_317_sdr_12.9755.ckpt'
    bnb_config = BitsAndBytesConfig(load_in_8bit=True, llm_int8_threshold=6.0)

    # --- SETUP ---
    install_fonts()
    download_video(url, project_name)
    extract_audio(video_file, input_audio)

    ###############################################################
    ###                     VOCAL ISOLATION                     ###
    ###############################################################
    print("\n=== Step 1: Vocal Isolation (ViperX) ===")

    separator = Separator(output_single_stem='vocals')
    separator.load_model(model_filename=VOCAL_MODEL_FILENAME)
    output_names = {'Vocals': 'isolated_vocals'}
    vocal_audio_files = separator.separate(input_audio, output_names)

    if isinstance(vocal_audio_files, list):
        vocal_path = vocal_audio_files[0]
    else:
        vocal_path = vocal_audio_files

    sample, sr = librosa.load(vocal_path, sr=16000)
    print(f"✓ Loaded vocal track: {vocal_path}")

    del separator
    torch.cuda.empty_cache()
    gc.collect()

    ###############################################################
    ###                     TRANSCRIPTION                       ###
    ###############################################################
    print("\n=== Step 2: Transcription (Parakeet) ===")

    asr_model = nemo_asr.models.ASRModel.from_pretrained(model_name='nvidia/parakeet-tdt-0.6b-v3')
    transcript = asr_model.transcribe(sample, timestamps=True)
    word_timestamps = transcript[0].timestamp['word']

    del asr_model
    torch.cuda.empty_cache()
    gc.collect()

    ###############################################################
    ###                  SIMPLE SENTENCE CHUNKING               ###
    ###############################################################
    print("\n=== Step 3: Simple Sentence Chunking ===")

    def simple_sentence_chunking(word_timestamps, min_words_for_comma=5):
        sentences = []
        current_sentence_words = []
        current_start = None
        last_end = None
        HARD_STOPS = ('.', '!', '?', ';', '。', '．', '！', '？', '；')
        COMMA = (',', '，')

        for i, stamp in enumerate(word_timestamps):
            word = stamp['word']
            if current_start is None: current_start = stamp['start']
            current_sentence_words.append(stamp)
            last_end = stamp['end']
            is_last_word = (i == len(word_timestamps) - 1)
            has_hard_stop = any(word.endswith(stop) for stop in HARD_STOPS)
            has_comma = any(word.endswith(comma) for comma in COMMA)
            should_break = False

            if is_last_word: should_break = True
            elif has_hard_stop: should_break = True
            elif has_comma and len(current_sentence_words) >= min_words_for_comma: should_break = True

            if should_break:
                sentences.append({
                    'Text': ' '.join([w['word'] for w in current_sentence_words]),
                    'Start': current_start,
                    'End': last_end,
                    'Words': current_sentence_words,
                    'WordCount': len(current_sentence_words)
                })
                current_sentence_words = []
                current_start = None
        return sentences

    sentences_data = simple_sentence_chunking(word_timestamps, min_words_for_comma=5)
    df = pd.DataFrame(sentences_data)
    print(f"   -> Created {len(df)} sentence segments")
    russian_word_counts = df['WordCount'].tolist()

    ###############################################################
    ###          TRANSLATION ADJUSTMENT FUNCTIONS               ###
    ###############################################################

    def adjust_translation_to_match_lines(translation_lines, russian_word_counts):
        target_count = len(russian_word_counts)
        current_count = len(translation_lines)
        if current_count == target_count: return translation_lines
        lines = translation_lines.copy()

        def get_word_counts(lines): return [len(line.split()) for line in lines]
        def calculate_word_count_distance(trans_counts, rus_counts):
            if len(trans_counts) != len(rus_counts): return float('inf')
            return sum(abs(t - r) for t, r in zip(trans_counts, rus_counts))

        while len(lines) > target_count:
            best_merge_idx = 0
            best_distance = float('inf')
            for i in range(len(lines) - 1):
                test_lines = lines.copy()
                test_lines[i] = test_lines[i] + " " + test_lines[i + 1]
                test_lines.pop(i + 1)
                if len(test_lines) == target_count:
                    distance = calculate_word_count_distance(get_word_counts(test_lines), russian_word_counts)
                    if distance < best_distance:
                        best_distance = distance
                        best_merge_idx = i
            lines[best_merge_idx] = lines[best_merge_idx] + " " + lines[best_merge_idx + 1]
            lines.pop(best_merge_idx + 1)
            print(f"   -> Merged lines {best_merge_idx+1} and {best_merge_idx+2}")

        while len(lines) < target_count:
            best_split_idx = 0
            best_split_point = 0
            best_distance = float('inf')
            found_valid_split = False
            for i in range(len(lines)):
                line_to_split = lines[i]
                words = line_to_split.split()
                if len(words) < 3: continue
                split_candidates = []
                for j, word in enumerate(words):
                    if j > 0 and j < len(words) and word.endswith(','): split_candidates.append(j + 1)
                if not split_candidates:
                    for j, word in enumerate(words):
                        if j > 1 and j < len(words) - 1:
                            if word.lower() in ['and', 'but', 'or', 'while', 'when', 'as', 'though', 'with', 'where', 'who', 'which', 'that']:
                                split_candidates.append(j)
                if not split_candidates:
                    if len(words) >= 6: split_candidates.extend([len(words) // 3, len(words) // 2, (2 * len(words)) // 3])
                    elif len(words) >= 3: split_candidates.append(len(words) // 2)

                for split_point in split_candidates:
                    if split_point <= 0 or split_point >= len(words): continue
                    test_lines = lines.copy()
                    first_half = ' '.join(words[:split_point])
                    second_half = ' '.join(words[split_point:])
                    test_lines[i] = first_half
                    test_lines.insert(i + 1, second_half)
                    distance = calculate_word_count_distance(get_word_counts(test_lines), russian_word_counts)
                    if distance < best_distance:
                        best_distance = distance
                        best_split_idx = i
                        best_split_point = split_point
                        found_valid_split = True

            if found_valid_split and best_split_point > 0:
                words = lines[best_split_idx].split()
                first_half = ' '.join(words[:best_split_point])
                second_half = ' '.join(words[best_split_point:])
                lines[best_split_idx] = first_half
                lines.insert(best_split_idx + 1, second_half)
                print(f"   -> Split line {best_split_idx+1} at word {best_split_point}")
            else:
                print(f"   -> WARNING: Cannot find valid split point")
                break
        return lines

    ###############################################################
    ###     TRANSLATION WITH CHIMERA REFINEMENT                 ###
    ###############################################################

    def translate_with_chimera_numbered(df_rus, russian_word_counts):
        num_lines = len(df_rus)
        russian_lines = df_rus['Text'].tolist()
        russian_text_numbered = "\n".join([f"{i+1}. {line}" for i, line in enumerate(russian_lines)])

        # --- PHASE 1: Generate Candidates with Base Model ---
        print(f"\n=== Step 4a: Generating Candidates (Hunyuan-MT-7B) - {num_lines} lines ===")

        tokenizer = AutoTokenizer.from_pretrained("tencent/Hunyuan-MT-7B", trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained("tencent/Hunyuan-MT-7B", device_map="auto", quantization_config=bnb_config, trust_remote_code=True)

        all_translations = []
        perfect_count = 0
        attempt = 0

        while attempt < 21:
            attempt += 1
            prompt = (
                f"You are an award-winning bilingual Russian-English literary translator specializing in songs and poetry. "
                f"Translate this Russian song to English while preserving meter, syllable counts, rhyme scheme, and poetic qualities.\n\n"
                f"CRITICAL: The output must have EXACTLY {num_lines} numbered lines (1., 2., 3., etc.)\n"
                f"Each line should translate the corresponding Russian line.\n\n"
                f"Russian text ({num_lines} lines):\n{russian_text_numbered}\n\n"
                f"English translation (exactly {num_lines} numbered lines):"
            )
            inputs = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True, return_tensors="pt").to(model.device)

            with torch.inference_mode():
                outputs = model.generate(inputs, max_new_tokens=1024, do_sample=True, top_k=30, top_p=0.85, repetition_penalty=1.12, temperature=0.3 + (attempt * 0.08))

            translation = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True).strip()
            lines = []
            for line in translation.split('\n'):
                line = line.strip()
                if line and len(line) > 2:
                    if line[0].isdigit():
                        for sep in ['.', ')', ']']:
                            if sep in line[:5]:
                                parts = line.split(sep, 1)
                                if len(parts) > 1:
                                    line = parts[1].strip()
                                    break
                    if line: lines.append(line)

            line_count = len(lines)
            all_translations.append((lines, line_count))

            if line_count == num_lines:
                perfect_count += 1
                print(f"   -> Candidate {attempt}: SUCCESS ({line_count} lines) [Found {perfect_count}/6]")
            else:
                print(f"   -> Candidate {attempt}: Got {line_count} lines (need {num_lines})")

            # STOPPING CONDITION: Stop if we have 6 matching translations
            if perfect_count >= 6:
                print("   -> Reached 6 perfect candidates. Stopping generation.")
                break

        print("   -> Unloading base model...")
        del model, tokenizer
        torch.cuda.empty_cache()
        gc.collect()

        # Sort and Select Candidates
        all_translations.sort(key=lambda x: abs(x[1] - num_lines))
        candidates_for_refinement = []
        print(f"   -> Processing top candidates for adjustment...")

        for lines, count in all_translations[:10]:
            if count == num_lines:
                candidates_for_refinement.append(lines)
            elif abs(count - num_lines) <= 5:
                try:
                    adjusted = adjust_translation_to_match_lines(lines, russian_word_counts)
                    if len(adjusted) == num_lines:
                        candidates_for_refinement.append(adjusted)
                        print(f"   -> Adjusted {count} → {num_lines} lines successfully")
                except Exception as e:
                    print(f"   -> Adjustment error: {e}")

            if len(candidates_for_refinement) >= 6:
                break

        if not candidates_for_refinement:
            print("   -> WARNING: No viable candidates. Using emergency fallback.")
            return [f"Translation line {i+1}" for i in range(num_lines)]

        # --- PHASE 2: Refine with Chimera ---
        print(f"\n=== Step 4b: Refinement (Chimera-7B) - {num_lines} lines ===")

        chimera_tokenizer = AutoTokenizer.from_pretrained("tencent/Hunyuan-MT-Chimera-7B", trust_remote_code=True)
        chimera_model = AutoModelForCausalLM.from_pretrained("tencent/Hunyuan-MT-Chimera-7B", device_map="auto", quantization_config=bnb_config, trust_remote_code=True)

        candidates_block = "\n\n".join([f"Candidate {i+1}:\n" + "\n".join([f"{j+1}. {line}" for j, line in enumerate(cand)]) for i, cand in enumerate(candidates_for_refinement)])
        refined_translation = None
        chimera_attempts = 0

        while refined_translation is None and chimera_attempts < 10:
            chimera_attempts += 1
            print(f"   -> Refinement attempt {chimera_attempts}...")

            chimera_prompt = (
                f"You are an award-winning bilingual Russian-English literary translator specializing in songs and poetry. "
                f"Review these {len(candidates_for_refinement)} translation candidates and produce the best refined version.\n\n"
                f"CRITICAL REQUIREMENTS:\n"
                f"- Output EXACTLY {num_lines} numbered lines (1., 2., 3., etc.)\n"
                f"- Preserve meter, syllable counts, and rhyme scheme from Russian\n"
                f"- Combine the best aspects of all candidates\n"
                f"- Each line must match the corresponding Russian line\n\n"
                f"Original Russian ({num_lines} lines):\n{russian_text_numbered}\n\n"
                f"Translation Candidates:\n{candidates_block}\n\n"
                f"Refined English translation (exactly {num_lines} numbered lines):"
            )

            c_inputs = chimera_tokenizer.apply_chat_template([{"role": "user", "content": chimera_prompt}], add_generation_prompt=True, return_tensors="pt").to(chimera_model.device)

            with torch.inference_mode():
                c_outputs = chimera_model.generate(c_inputs, max_new_tokens=1024, do_sample=True, top_k=25, top_p=0.9, repetition_penalty=1.1, temperature=0.3 + (chimera_attempts * 0.05))

            candidate = chimera_tokenizer.decode(c_outputs[0][c_inputs.shape[1]:], skip_special_tokens=True).strip()
            lines = []
            for line in candidate.split('\n'):
                line = line.strip()
                if line and len(line) > 2:
                    if line[0].isdigit():
                        for sep in ['.', ')', ']']:
                            if sep in line[:5]:
                                parts = line.split(sep, 1)
                                if len(parts) > 1:
                                    line = parts[1].strip()
                                    break
                    if line: lines.append(line)

            if len(lines) == num_lines:
                refined_translation = lines
                print(f"   -> ✓ Refinement successful ({num_lines} lines)")
                # STOPPING CONDITION: Stop as soon as one matching translation is produced
                break
            else:
                print(f"   -> Refinement got {len(lines)} lines (need {num_lines})")

        del chimera_model, chimera_tokenizer
        torch.cuda.empty_cache()
        gc.collect()

        final_lines = refined_translation if refined_translation else candidates_for_refinement[0]

        # --- PHASE 3: Quality Check ---
        print("\n=== Step 4c: Translation Quality Check ===")
        embedder = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
        ru_embeddings = embedder.encode(russian_lines, convert_to_tensor=True)
        en_embeddings = embedder.encode(final_lines, convert_to_tensor=True)
        similarities = []
        for i in range(num_lines):
            sim = util.pytorch_cos_sim(ru_embeddings[i], en_embeddings[i]).item()
            similarities.append(sim)

        avg_sim = sum(similarities) / num_lines
        print(f"   -> Average semantic match: {avg_sim:.3f}")

        misalignment_detected = False
        for i in range(num_lines):
            correct_sim = similarities[i]
            for j in range(num_lines):
                if i != j:
                    cross_sim = util.pytorch_cos_sim(ru_embeddings[i], en_embeddings[j]).item()
                    if cross_sim > correct_sim + 0.15:
                        print(f"   -> POSSIBLE MISALIGNMENT: Russian line {i+1} may better match English line {j+1}")
                        misalignment_detected = True

        if not misalignment_detected: print("   -> ✓ No structural misalignments detected")
        del embedder
        torch.cuda.empty_cache()

        return final_lines

    ###############################################################
    ###                   RUN TRANSLATION                       ###
    ###############################################################

    # Run translation with Chimera refinement
    english_lines = translate_with_chimera_numbered(df, russian_word_counts)
    df['Translation'] = english_lines
    df.to_csv(output_csv_path, encoding='utf-8', index=False)
    print(f"\n✓ CSV saved to {output_csv_path}")

    ###############################################################
    ###              DUAL SRT GENERATION                        ###
    ###############################################################
    print("\n=== Step 5: Generating Dual SRT Files ===")

    def format_time(seconds):
        milliseconds = int((seconds - int(seconds)) * 1000)
        seconds = int(seconds)
        minutes, seconds = divmod(seconds, 60)
        hours, minutes = divmod(minutes, 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"

    HIGHLIGHT_COLOR = "#DAA425"  # Gold

    if os.path.exists(output_srt_rus): os.remove(output_srt_rus)
    if os.path.exists(output_srt_eng): os.remove(output_srt_eng)

    with open(output_srt_eng, 'a', encoding='utf-8') as f_eng:
        for i, row in df.iterrows():
            f_eng.write(f"{i+1}\n")
            f_eng.write(f"{format_time(row['Start'])} --> {format_time(row['End'])}\n")
            f_eng.write(f"{row['Translation']}\n\n")

    sub_counter = 1
    with open(output_srt_rus, 'a', encoding='utf-8') as f_rus:
        for index, row in df.iterrows():
            words_data = row['Words']
            for i in range(len(words_data)):
                active_word_data = words_data[i]
                current_start = active_word_data['start']
                if i < len(words_data) - 1:
                    current_end = words_data[i+1]['start']
                else:
                    current_end = active_word_data['end']

                formatted_russian = []
                for j, word_obj in enumerate(words_data):
                    word_text = word_obj['word']
                    if i == j:
                        formatted_russian.append(f'<font color="{HIGHLIGHT_COLOR}">{word_text}</font>')
                    else:
                        formatted_russian.append(word_text)

                f_rus.write(f"{sub_counter}\n")
                f_rus.write(f"{format_time(current_start)} --> {format_time(current_end)}\n")
                f_rus.write(f"{' '.join(formatted_russian)}\n\n")
                sub_counter += 1

    print(f"✓ Russian karaoke subtitles saved to {output_srt_rus}")

    # --- FINAL STEP: BURN SUBTITLES ---
    burn_subtitles(video_file, output_srt_rus, output_srt_eng, final_output_video)
    print("\n=== Translation Complete ===")

if __name__ == "__main__":
    main()
