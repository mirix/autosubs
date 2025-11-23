import os
import gc
import torch
import librosa
import pandas as pd
import nemo.collections.asr as nemo_asr
from sentence_transformers import SentenceTransformer, util
from audio_separator.separator import Separator
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

###############################################################
###                        VARIABLES                        ###
###############################################################

input_audio = 'Ekaterina.opus'
output_csv_path = 'Ekaterina.csv'
output_srt_rus = 'Ekaterina_rus.srt'
output_srt_eng = 'Ekaterina_eng.srt'

VOCAL_MODEL_FILENAME = 'model_bs_roformer_ep_317_sdr_12.9755.ckpt'

bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=6.0
)

###############################################################
###                     VOCAL ISOLATION                     ###
###############################################################
print("\n=== Step 1: Vocal Isolation (ViperX) ===")

separator = Separator(output_single_stem='vocals')
separator.load_model(model_filename=VOCAL_MODEL_FILENAME)
output_names = {'Vocals': 'isolated_vocals'}
vocal_audio_files = separator.separate(input_audio, output_names)
vocal_path = vocal_audio_files[0]

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
raw_text = transcript[0].timestamp['segment'][0]['segment']

del asr_model
torch.cuda.empty_cache()
gc.collect()

###############################################################
###                  SIMPLE SENTENCE CHUNKING               ###
###############################################################
print("\n=== Step 3: Simple Sentence Chunking ===")

def simple_sentence_chunking(word_timestamps, min_words_for_comma=5):
    """
    Simple, predictable chunking:
    1. Always split on . ? ! ;
    2. Only split on comma if line has at least min_words_for_comma words
    """
    sentences = []
    current_sentence_words = []
    current_start = None
    last_end = None

    HARD_STOPS = ('.', '!', '?', ';', '。', '．', '！', '？', '；')
    COMMA = (',', '，')

    for i, stamp in enumerate(word_timestamps):
        word = stamp['word']

        # Initialize start time
        if current_start is None:
            current_start = stamp['start']

        current_sentence_words.append(stamp)
        last_end = stamp['end']

        is_last_word = (i == len(word_timestamps) - 1)
        has_hard_stop = any(word.endswith(stop) for stop in HARD_STOPS)
        has_comma = any(word.endswith(comma) for comma in COMMA)

        # Simple decision logic
        should_break = False

        if is_last_word:
            should_break = True
        elif has_hard_stop:
            should_break = True
        elif has_comma and len(current_sentence_words) >= min_words_for_comma:
            should_break = True

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

# Store word counts for later use in translation adjustment
russian_word_counts = df['WordCount'].tolist()

###############################################################
###          TRANSLATION ADJUSTMENT FUNCTIONS               ###
###############################################################

def adjust_translation_to_match_lines(translation_lines, russian_word_counts):
    """
    Adjust translation line count to match Russian lines.
    Uses word count similarity as criterion for split/merge decisions.
    """
    target_count = len(russian_word_counts)
    current_count = len(translation_lines)

    if current_count == target_count:
        return translation_lines

    lines = translation_lines.copy()

    # Calculate word counts for current translation lines
    def get_word_counts(lines):
        return [len(line.split()) for line in lines]

    def calculate_word_count_distance(trans_counts, rus_counts):
        """Calculate sum of absolute differences in word counts"""
        if len(trans_counts) != len(rus_counts):
            return float('inf')
        return sum(abs(t - r) for t, r in zip(trans_counts, rus_counts))

    # Too many lines - need to merge
    while len(lines) > target_count:
        best_merge_idx = 0
        best_distance = float('inf')

        # Try each possible merge and pick the one with best word count match
        for i in range(len(lines) - 1):
            # Simulate merge
            test_lines = lines.copy()
            test_lines[i] = test_lines[i] + " " + test_lines[i + 1]
            test_lines.pop(i + 1)

            # If we're at target count, evaluate the match
            if len(test_lines) == target_count:
                distance = calculate_word_count_distance(get_word_counts(test_lines), russian_word_counts)
                if distance < best_distance:
                    best_distance = distance
                    best_merge_idx = i

        # Perform the best merge
        lines[best_merge_idx] = lines[best_merge_idx] + " " + lines[best_merge_idx + 1]
        lines.pop(best_merge_idx + 1)
        print(f"   -> Merged lines {best_merge_idx+1} and {best_merge_idx+2}")

    # Too few lines - need to split
    while len(lines) < target_count:
        best_split_idx = 0
        best_split_point = 0
        best_distance = float('inf')
        found_valid_split = False

        # Try each possible split and pick the one with best word count match
        for i in range(len(lines)):
            line_to_split = lines[i]
            words = line_to_split.split()

            if len(words) < 3:  # Lowered threshold from 4 to 3
                continue  # Can't split this line

            # Try splitting at commas first, then other natural breaks
            split_candidates = []

            # Find commas (most natural split point)
            for j, word in enumerate(words):
                if j > 0 and j < len(words) and word.endswith(','):
                    split_candidates.append(j + 1)

            # Try conjunctions and prepositions
            if not split_candidates:
                for j, word in enumerate(words):
                    if j > 1 and j < len(words) - 1:  # Relaxed boundaries
                        if word.lower() in ['and', 'but', 'or', 'while', 'when', 'as', 'though',
                                           'with', 'where', 'who', 'which', 'that']:
                            split_candidates.append(j)

            # Try splitting after semicolons or em-dashes
            if not split_candidates:
                for j, word in enumerate(words):
                    if j > 1 and j < len(words) and (word.endswith(';') or word.endswith('—') or word.endswith('–')):
                        split_candidates.append(j + 1)

            # If still nothing, try multiple split points (1/3, 1/2, 2/3)
            if not split_candidates:
                if len(words) >= 6:
                    split_candidates.extend([len(words) // 3, len(words) // 2, (2 * len(words)) // 3])
                elif len(words) >= 3:
                    split_candidates.append(len(words) // 2)

            # Test each split candidate
            for split_point in split_candidates:
                if split_point <= 0 or split_point >= len(words):
                    continue

                test_lines = lines.copy()
                first_half = ' '.join(words[:split_point])
                second_half = ' '.join(words[split_point:])
                test_lines[i] = first_half
                test_lines.insert(i + 1, second_half)

                # Evaluate regardless of whether we're at target (greedy approach)
                distance = calculate_word_count_distance(get_word_counts(test_lines), russian_word_counts)
                if distance < best_distance:
                    best_distance = distance
                    best_split_idx = i
                    best_split_point = split_point
                    found_valid_split = True

        # Perform the best split
        if found_valid_split and best_split_point > 0:
            words = lines[best_split_idx].split()
            first_half = ' '.join(words[:best_split_point])
            second_half = ' '.join(words[best_split_point:])
            lines[best_split_idx] = first_half
            lines.insert(best_split_idx + 1, second_half)
            print(f"   -> Split line {best_split_idx+1} at word {best_split_point}")
        else:
            print(f"   -> WARNING: Cannot find valid split point (remaining lines too short)")
            break

    # Report final word count match
    if len(lines) == target_count:
        distance = calculate_word_count_distance(get_word_counts(lines), russian_word_counts)
        print(f"   -> Final word count distance: {distance}")

    return lines

###############################################################
###     TRANSLATION WITH CHIMERA REFINEMENT                 ###
###############################################################

def translate_with_chimera_numbered(df_rus, russian_word_counts):
    """
    Two-phase translation using numbered lines to enforce structure
    """
    num_lines = len(df_rus)
    russian_lines = df_rus['Text'].tolist()

    # Create numbered format for translation
    russian_text_numbered = "\n".join([f"{i+1}. {line}" for i, line in enumerate(russian_lines)])

    # --- PHASE 1: Generate Candidates with Base Model ---
    print(f"\n=== Step 4a: Generating Candidates (Hunyuan-MT-7B) - {num_lines} lines ===")

    tokenizer = AutoTokenizer.from_pretrained("tencent/Hunyuan-MT-7B", trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        "tencent/Hunyuan-MT-7B",
        device_map="auto",
        quantization_config=bnb_config,
        trust_remote_code=True,
    )

    all_translations = []  # Store all attempts, not just perfect ones
    attempt = 0

    while attempt < 21:  # Generate more attempts
        attempt += 1

        prompt = (
            f"You are an award-winning bilingual Russian-English literary translator specializing in songs and poetry. "
            f"Translate this Russian song to English while preserving meter, syllable counts, rhyme scheme, and poetic qualities.\n\n"
            f"CRITICAL: The output must have EXACTLY {num_lines} numbered lines (1., 2., 3., etc.)\n"
            f"Each line should translate the corresponding Russian line.\n\n"
            f"Russian text ({num_lines} lines):\n{russian_text_numbered}\n\n"
            f"English translation (exactly {num_lines} numbered lines):"
        )

        inputs = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(model.device)

        with torch.inference_mode():
            outputs = model.generate(
                inputs,
                max_new_tokens=1024,
                do_sample=True,
                top_k=30,
                top_p=0.85,
                repetition_penalty=1.12,
                temperature=0.3 + (attempt * 0.08)
            )

        translation = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True).strip()

        # Parse numbered lines
        lines = []
        for line in translation.split('\n'):
            line = line.strip()
            # Remove number prefix if present
            if line and len(line) > 2:
                if line[0].isdigit():
                    for sep in ['.', ')', ']']:
                        if sep in line[:5]:
                            parts = line.split(sep, 1)
                            if len(parts) > 1:
                                line = parts[1].strip()
                                break
                if line:
                    lines.append(line)

        line_count = len(lines)
        all_translations.append((lines, line_count))

        if line_count == num_lines:
            print(f"   -> Candidate {attempt}: SUCCESS ({num_lines} lines)")
        else:
            print(f"   -> Candidate {attempt}: Got {line_count} lines (need {num_lines})")

    print("   -> Unloading base model...")
    del model, tokenizer
    torch.cuda.empty_cache()
    gc.collect()

    # Sort by how close to target, take best candidates
    all_translations.sort(key=lambda x: abs(x[1] - num_lines))
    candidates_for_refinement = []

    print(f"   -> Processing top candidates for adjustment...")

    for lines, count in all_translations[:10]:  # Check more candidates
        # Try to adjust if within reasonable range
        if count == num_lines:
            # Perfect match - use as is
            candidates_for_refinement.append(lines)
            print(f"   -> Added perfect candidate ({count} lines)")
        elif abs(count - num_lines) <= 5:  # Increased tolerance from 2 to 5
            # Try to adjust
            try:
                adjusted = adjust_translation_to_match_lines(lines, russian_word_counts)
                if len(adjusted) == num_lines:
                    candidates_for_refinement.append(adjusted)
                    print(f"   -> Adjusted {count} → {num_lines} lines successfully")
                else:
                    print(f"   -> Failed to adjust {count} lines (got {len(adjusted)})")
            except Exception as e:
                print(f"   -> Adjustment error: {e}")

        if len(candidates_for_refinement) >= 6:  # Chimera limit
            break

    if not candidates_for_refinement:
        print("   -> WARNING: No viable candidates. Using emergency fallback.")
        return [f"Translation line {i+1}" for i in range(num_lines)]

    print(f"   -> Prepared {len(candidates_for_refinement)} candidates for refinement")

    # --- PHASE 2: Refine with Chimera ---
    print(f"\n=== Step 4b: Refinement (Chimera-7B) - {num_lines} lines ===")

    chimera_tokenizer = AutoTokenizer.from_pretrained("tencent/Hunyuan-MT-Chimera-7B", trust_remote_code=True)
    chimera_model = AutoModelForCausalLM.from_pretrained(
        "tencent/Hunyuan-MT-Chimera-7B",
        device_map="auto",
        quantization_config=bnb_config,
        trust_remote_code=True,
    )

    # Format candidates with numbers
    candidates_block = "\n\n".join([
        f"Candidate {i+1}:\n" + "\n".join([f"{j+1}. {line}" for j, line in enumerate(cand)])
        for i, cand in enumerate(candidates_for_refinement)
    ])

    refined_translation = None
    chimera_attempts = 0

    while refined_translation is None and chimera_attempts < 21:
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

        c_inputs = chimera_tokenizer.apply_chat_template(
            [{"role": "user", "content": chimera_prompt}],
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(chimera_model.device)

        with torch.inference_mode():
            c_outputs = chimera_model.generate(
                c_inputs,
                max_new_tokens=1024,
                do_sample=True,
                top_k=25,
                top_p=0.9,
                repetition_penalty=1.1,
                temperature=0.3 + (chimera_attempts * 0.05)
            )

        candidate = chimera_tokenizer.decode(c_outputs[0][c_inputs.shape[1]:], skip_special_tokens=True).strip()

        # Parse refined output
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
                if line:
                    lines.append(line)

        if len(lines) == num_lines:
            refined_translation = lines
            print(f"   -> ✓ Refinement successful ({num_lines} lines)")
        else:
            print(f"   -> Refinement got {len(lines)} lines (need {num_lines})")

    del chimera_model, chimera_tokenizer
    torch.cuda.empty_cache()
    gc.collect()

    # Use refined if successful, otherwise best candidate
    final_lines = refined_translation if refined_translation else candidates_for_refinement[0]

    # --- PHASE 3: Enhanced Quality Verification (No Rearrangement) ---
    print("\n=== Step 4c: Translation Quality Check ===")

    embedder = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
    ru_embeddings = embedder.encode(russian_lines, convert_to_tensor=True)
    en_embeddings = embedder.encode(final_lines, convert_to_tensor=True)

    # Calculate per-line similarity (diagonal - correct alignment)
    similarities = []
    for i in range(num_lines):
        sim = util.pytorch_cos_sim(ru_embeddings[i], en_embeddings[i]).item()
        similarities.append(sim)

    avg_sim = sum(similarities) / num_lines
    min_sim = min(similarities)

    print(f"   -> Average semantic match: {avg_sim:.3f}")
    print(f"   -> Weakest line match: {min_sim:.3f}")

    # Detect potential misalignments by checking if other positions score higher
    misalignment_detected = False
    for i in range(num_lines):
        correct_sim = similarities[i]
        # Check if any other English line is a much better match
        for j in range(num_lines):
            if i != j:
                cross_sim = util.pytorch_cos_sim(ru_embeddings[i], en_embeddings[j]).item()
                if cross_sim > correct_sim + 0.15:  # Significantly better match elsewhere
                    print(f"   -> POSSIBLE MISALIGNMENT: Russian line {i+1} may better match English line {j+1}")
                    print(f"      Current: {correct_sim:.3f} vs Alternative: {cross_sim:.3f}")
                    misalignment_detected = True

    if not misalignment_detected:
        print("   -> ✓ No structural misalignments detected")
    else:
        print("   -> ⚠ Consider reviewing the chunking parameters or translation quality")

    # Flag weak translations that might need manual review
    weak_lines = [i+1 for i, sim in enumerate(similarities) if sim < 0.35]
    if weak_lines:
        print(f"   -> Lines needing review: {', '.join(map(str, weak_lines))}")

    del embedder
    torch.cuda.empty_cache()

    return final_lines

###############################################################
###                   RUN TRANSLATION                       ###
###############################################################

# Run translation with Chimera refinement
english_lines = translate_with_chimera_numbered(df, russian_word_counts)
df['Translation'] = english_lines

# Save CSV
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

# Clear old files
if os.path.exists(output_srt_rus): os.remove(output_srt_rus)
if os.path.exists(output_srt_eng): os.remove(output_srt_eng)

# --- GENERATE ENGLISH SRT (Sentence-level, Stable) ---
with open(output_srt_eng, 'a', encoding='utf-8') as f_eng:
    for i, row in df.iterrows():
        start_time = row['Start']
        end_time = row['End']
        text = row['Translation']

        f_eng.write(f"{i+1}\n")
        f_eng.write(f"{format_time(start_time)} --> {format_time(end_time)}\n")
        f_eng.write(f"{text}\n\n")

print(f"✓ English subtitles saved to {output_srt_eng}")

# --- GENERATE RUSSIAN SRT (Word-level, Continuous) ---
sub_counter = 1
with open(output_srt_rus, 'a', encoding='utf-8') as f_rus:
    for index, row in df.iterrows():
        words_data = row['Words']

        for i in range(len(words_data)):
            active_word_data = words_data[i]
            current_start = active_word_data['start']

            # Extend to next word start (flicker prevention)
            if i < len(words_data) - 1:
                current_end = words_data[i+1]['start']
            else:
                current_end = active_word_data['end']

            # Build formatted text
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
print("\n=== Translation Complete ===")
