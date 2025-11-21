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
# CHANGED: Define two separate output paths
output_srt_rus = 'Ekaterina_rus.srt'
output_srt_eng = 'Ekaterina_eng.srt'

# VOCAL MODEL: ViperX 1297 (SOTA)
VOCAL_MODEL_FILENAME = 'model_bs_roformer_ep_317_sdr_12.9755.ckpt'

# MEMORY OPTIMIZATION CONFIG (4-bit)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
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

################################################################
###                  SENTENCE CHUNKING                       ###
################################################################
print("\n=== Step 3: Sentence Chunking ===")

def sentence_chunking(word_timestamps, hard_stops, soft_stops, min_words_soft=5, gap_threshold=2.0):
    sentences = []
    current_sentence_words = []
    current_sentence_text = []
    current_start = None
    current_end = None

    for i, stamp in enumerate(word_timestamps):
        word = stamp['word']

        if (current_end is not None and
            stamp['start'] - current_end > gap_threshold and
            current_sentence_words):

            sentences.append({
                'Text': ' '.join(current_sentence_text),
                'Start': current_start,
                'End': current_end,
                'Words': current_sentence_words
            })
            current_sentence_words = []
            current_sentence_text = []
            current_start = None
            current_end = None

        if current_start is None:
            current_start = stamp['start']

        current_sentence_words.append(stamp)
        current_sentence_text.append(word)
        current_end = stamp['end']

        should_end = False
        is_last_word = (i == len(word_timestamps) - 1)

        if any(word.endswith(stop) for stop in hard_stops):
            should_end = True
        elif not should_end and any(word.endswith(stop) for stop in soft_stops) and len(current_sentence_text) >= min_words_soft:
            should_end = True
        if is_last_word:
            should_end = True

        if should_end:
            sentences.append({
                'Text': ' '.join(current_sentence_text),
                'Start': current_start,
                'End': current_end,
                'Words': current_sentence_words
            })
            current_sentence_words = []
            current_sentence_text = []
            current_start = None
            current_end = None

    return sentences

HARD_STOPS = ('。', '．', '.', '！', '!', '?', '？')
SOFT_STOPS = (',', '，', ':' , ';', '；', '—', '–')

sentences_data = sentence_chunking(word_timestamps, HARD_STOPS, SOFT_STOPS)
df = pd.DataFrame(sentences_data)

### =======================================================================
### 4. TRANSLATION HELPERS
### =======================================================================

def split_translation_text(translation_text, hard_stops=HARD_STOPS, soft_stops=SOFT_STOPS, min_words_soft=5):
    words = translation_text.split()
    sentences = []
    current_sentence = []
    for i, word in enumerate(words):
        current_sentence.append(word)
        should_end = False
        is_last_word = (i == len(words) - 1)
        if any(word.endswith(stop) for stop in hard_stops): should_end = True
        elif not should_end and any(word.endswith(stop) for stop in soft_stops) and len(current_sentence) >= min_words_soft: should_end = True
        if is_last_word: should_end = True
        if should_end:
            sentences.append(' '.join(current_sentence))
            current_sentence = []
    if current_sentence: sentences.append(' '.join(current_sentence))
    return sentences

def apply_translation_splitting(translations, df_rus):
    split_translations = []
    for translation in translations:
        split_sentences = split_translation_text(translation)
        if len(split_sentences) > len(df_rus):
            while len(split_sentences) > len(df_rus):
                merged = split_sentences[-2] + " " + split_sentences[-1]
                split_sentences = split_sentences[:-2] + [merged]
        elif len(split_sentences) < len(df_rus):
            while len(split_sentences) < len(df_rus):
                longest_idx = max(range(len(split_sentences)), key=lambda i: len(split_sentences[i].split()))
                words = split_sentences[longest_idx].split()
                if len(words) > 1:
                    mid = len(words) // 2
                    split_sentences = split_sentences[:longest_idx] + [' '.join(words[:mid]), ' '.join(words[mid:])] + split_sentences[longest_idx+1:]
                else: break
        split_translations.append("\n".join(split_sentences))
    return split_translations

def rematch_multiple_translations(df_rus, translations, threshold=0.1):
    print("   -> Running semantic rematching...")
    embedder = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
    ru_embeddings = embedder.encode(df_rus['Text'].tolist(), convert_to_tensor=True)

    rematched_translations = []
    for translation in translations:
        english_lines = [line.strip() for line in translation.split('\n') if line.strip()]
        if len(english_lines) != len(df_rus):
            rematched_translations.append(translation); continue

        en_embeddings = embedder.encode(english_lines, convert_to_tensor=True)
        current_lines = english_lines.copy()
        current_embeddings = en_embeddings.clone()

        for _ in range(len(df_rus)):
            improved = False
            for j in range(len(df_rus) - 1):
                orig_sim = util.pytorch_cos_sim(ru_embeddings[j], current_embeddings[j]).item() + \
                           util.pytorch_cos_sim(ru_embeddings[j+1], current_embeddings[j+1]).item()
                swap_sim = util.pytorch_cos_sim(ru_embeddings[j], current_embeddings[j+1]).item() + \
                           util.pytorch_cos_sim(ru_embeddings[j+1], current_embeddings[j]).item()

                if swap_sim > orig_sim + threshold:
                    current_lines[j], current_lines[j+1] = current_lines[j+1], current_lines[j]
                    current_embeddings[j], current_embeddings[j+1] = current_embeddings[j+1].clone(), current_embeddings[j].clone()
                    improved = True
            if not improved: break
        rematched_translations.append("\n".join(current_lines))

    del embedder
    torch.cuda.empty_cache()
    return rematched_translations

### =======================================================================
### 5. TRANSLATION + CHIMERA REFINEMENT (4-BIT OPTIMIZED)
### =======================================================================

def translate_with_chimera(df_rus):
    full_russian_text = "\n".join(df_rus['Text'].tolist())
    num_russian_lines = len(df_rus)

    # --- PHASE 1: Generate Candidates with Base Model ---
    print("\n=== Step 4a: Generating Candidates (Hunyuan-MT-7B 4-bit) ===")

    tokenizer = AutoTokenizer.from_pretrained("tencent/Hunyuan-MT-7B", trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        "tencent/Hunyuan-MT-7B",
        device_map="auto",
        quantization_config=bnb_config,
        trust_remote_code=True,
    )

    raw_translations = []
    attempt = 0
    while len(raw_translations) < 4 and attempt < 10:
        attempt += 1
        prompt = (f"Translate the following Russian song lyrics to English. "
                  f"Match the original verse by verse.\n\n{full_russian_text}\n\nEnglish translation:")

        inputs = tokenizer.apply_chat_template([{"role": "user", "content": prompt}],
                                             add_generation_prompt=True, return_tensors="pt").to(model.device)

        with torch.inference_mode():
            outputs = model.generate(inputs, max_new_tokens=1024, do_sample=True, top_k=30, top_p=0.85,
                                   repetition_penalty=1.1, temperature=0.3 + (attempt * 0.1))

        translation = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True).strip()
        raw_translations.append(translation)
        print(f"   -> Candidate {attempt} generated.")

    print("   -> Unloading base model...")
    del model, tokenizer
    torch.cuda.empty_cache()
    gc.collect()

    split_translations = apply_translation_splitting(raw_translations, df_rus)
    matching_translations = [t for t in split_translations if len(t.strip().split('\n')) == num_russian_lines]

    if not matching_translations:
        matching_translations = split_translations[:3]

    rematched_translations = rematch_multiple_translations(df_rus, matching_translations)

    # --- PHASE 2: Refine with Chimera ---
    print("\n=== Step 4b: Refinement (Chimera-7B 4-bit) ===")

    chimera_tokenizer = AutoTokenizer.from_pretrained("tencent/Hunyuan-MT-Chimera-7B", trust_remote_code=True)
    chimera_model = AutoModelForCausalLM.from_pretrained(
        "tencent/Hunyuan-MT-Chimera-7B",
        device_map="auto",
        quantization_config=bnb_config,
        trust_remote_code=True,
    )

    refined_translation = None
    chimera_attempts = 0

    translations_block = "\n\n".join([f"Translation {i+1}:\n{trans}" for i, trans in enumerate(rematched_translations)])

    while refined_translation is None and chimera_attempts < 5:
        chimera_attempts += 1
        print(f"   -> Refinement attempt {chimera_attempts}...")

        chimera_prompt = (
            f"I need you to refine an English translation of a Russian song. "
            f"Combine the best elements of these translations while maintaining the exact same line structure.\n\n"
            f"Original Russian ({num_russian_lines} lines):\n{full_russian_text}\n\n"
            f"Candidates:\n{translations_block}\n\n"
            f"CRITICAL: Result MUST have exactly {num_russian_lines} lines.\n\n"
            f"Refined English translation:"
        )

        c_inputs = chimera_tokenizer.apply_chat_template([{"role": "user", "content": chimera_prompt}],
                                                       add_generation_prompt=True, return_tensors="pt").to(chimera_model.device)

        with torch.inference_mode():
            c_outputs = chimera_model.generate(c_inputs, max_new_tokens=1024, do_sample=True, temperature=0.3)

        candidate = chimera_tokenizer.decode(c_outputs[0][c_inputs.shape[1]:], skip_special_tokens=True).strip()

        split_refined = apply_translation_splitting([candidate], df_rus)
        if len(split_refined[0].strip().split('\n')) == num_russian_lines:
            refined_translation = split_refined[0]
            print("   -> ✓ Refinement successful and matched line count.")

    del chimera_model, chimera_tokenizer
    torch.cuda.empty_cache()
    gc.collect()

    final_text = refined_translation if refined_translation else rematched_translations[0]
    final_lines = [line.strip() for line in final_text.split('\n') if line.strip()]

    final_rematch = rematch_multiple_translations(df_rus, ["\n".join(final_lines)])
    final_lines = [line.strip() for line in final_rematch[0].split('\n') if line.strip()]

    df_rus['Translation'] = final_lines
    return df_rus

df = translate_with_chimera(df)
df.to_csv(output_csv_path, encoding='utf-8', index=False)

### =======================================================================
### 6. DUAL SRT GENERATION (FLICKER-FREE)
### =======================================================================
print("\n=== Step 5: Generating Dual SRT Files ===")

def format_time(seconds):
    milliseconds = int((seconds - int(seconds)) * 1000)
    seconds = int(seconds)
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"

HIGHLIGHT_COLOR = "#00ff00" # Green

# Clear old files
if os.path.exists(output_srt_rus): os.remove(output_srt_rus)
if os.path.exists(output_srt_eng): os.remove(output_srt_eng)

# --- GENERATE ENGLISH SRT (Sentence-level, Stable) ---
with open(output_srt_eng, 'a', encoding='utf-8') as f_eng:
    for i, row in df.iterrows():
        # English simply spans the entire duration of the Russian sentence
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

        # We must look ahead to ensure continuity
        for i in range(len(words_data)):
            active_word_data = words_data[i]

            # START: When this word begins
            current_start = active_word_data['start']

            # END: Logic to prevent flickering
            if i < len(words_data) - 1:
                # If there is a next word, extend this subtitle UNTIL the next word starts.
                # This fills the silence gap so the text doesn't disappear.
                current_end = words_data[i+1]['start']
            else:
                # If it's the last word, use the sentence end (or word end + buffer)
                current_end = active_word_data['end']
                # Optional: Ensure strictly no gap between sentences if they are close?
                # For now, sticking to word-end is safer to allow a 'breath' between lines.

            # Build the text line with formatting
            formatted_russian = []
            for j, word_obj in enumerate(words_data):
                word_text = word_obj['word']
                if i == j:
                    # Active Word
                    formatted_russian.append(f'<font color="{HIGHLIGHT_COLOR}">{word_text}</font>')
                else:
                    # Inactive Words
                    formatted_russian.append(word_text)

            # Write the block
            f_rus.write(f"{sub_counter}\n")
            f_rus.write(f"{format_time(current_start)} --> {format_time(current_end)}\n")
            f_rus.write(f"{' '.join(formatted_russian)}\n\n")

            sub_counter += 1

print(f"✓ Russian karaoke subtitles saved to {output_srt_rus}")
