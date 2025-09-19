### =======================================================================
### 0. SETUP AND IMPORTS
### =======================================================================
import pandas as pd
import torch
import gc
import os
import librosa
import re
import numpy as np
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, AutoModelForCausalLM, AutoTokenizer, pipeline
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# --- Configuration ---
audio_path = "letz.mp3" # Replace with your audio file
output_csv_path = 'letz_contextual.csv'
output_srt_path = 'letz_contextual.srt'


### =======================================================================
### 1. AUTOMATIC SPEECH RECOGNITION (ASR)
### =======================================================================
print("Step 1: Starting speech-to-text transcription...")

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_id_asr = "unilux/whisper-medium-v1-luxembourgish"

model_asr = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id_asr, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
)
model_asr.to(device)

processor_asr = AutoProcessor.from_pretrained(model_id_asr)

pipe_asr = pipeline(
    "automatic-speech-recognition",
    model=model_asr,
    tokenizer=processor_asr.tokenizer,
    feature_extractor=processor_asr.feature_extractor,
    torch_dtype=torch_dtype,
    device=device,
    return_timestamps="word",
)

generate_kwargs = {
    "max_new_tokens": 445,
    "num_beams": 1,
    "condition_on_prev_tokens": False,
    "compression_ratio_threshold": 1.35,
    "temperature": (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
    "logprob_threshold": -0.9,
    "no_speech_threshold": 0.9,
    "language": "luxembourgish",
}

sample, sr = librosa.load(audio_path, sr=16_000, mono=True)
result = pipe_asr(sample, generate_kwargs=generate_kwargs)
chunks = result['chunks']

print("Transcription complete.")

# --- Cleanup ASR model from GPU ---
del model_asr, processor_asr, pipe_asr
torch.cuda.empty_cache()
gc.collect()
print("Cleaned up ASR model from memory.")


### =======================================================================
### 2. SENTENCE SPLITTING
### =======================================================================
def split_sentences_into_chunks(data):
    """
    Splits the ASR word-level chunks into sentence-like segments
    based on punctuation, capitalization, and length heuristics.
    """
    primary_stops = ['.', '!', '?']
    secondary_stops = [':', ';', ',']

    sentences, start_times, end_times = [], [], []
    current_sentence, current_start, current_end = [], None, None

    for i, item in enumerate(data):
        text = item['text']
        start, end = item['timestamp']

        if current_start is None:
            current_start = start

        current_sentence.append(text)
        current_end = end

        ends_with_primary = any(text.endswith(stop) for stop in primary_stops)
        ends_with_secondary = any(text.endswith(stop) for stop in secondary_stops)
        is_last_token = i == len(data) - 1

        next_starts_upper = False
        if i + 1 < len(data):
            next_text = data[i + 1]['text'].strip()
            if next_text and (next_text[0].isupper() or next_text[0].isdigit()):
                next_starts_upper = True

        current_sentence_str = ''.join(current_sentence).strip()

        # Split conditions
        split_now = False
        if is_last_token:
            split_now = True
        elif ends_with_primary and next_starts_upper:
            split_now = True
        elif len(current_sentence_str) > 44 and ends_with_secondary:
            split_now = True

        if split_now and current_sentence:
            sentences.append(current_sentence_str)
            start_times.append(current_start)
            end_times.append(current_end)
            current_sentence, current_start, current_end = [], None, None

    return pd.DataFrame({
        'sentence_lb': sentences,
        'start_time': start_times,
        'end_time': end_times
    })

df_initial = split_sentences_into_chunks(chunks)
print("Step 2: Initial sentence splitting complete.")
print(f"Found {len(df_initial)} initial sentence chunks.")


### =======================================================================
### 3. FULL-TEXT TRANSLATION FOR CONTEXT
### =======================================================================
print("Step 3: Starting full-text translation...")

model_id_trans = "etamin/Letz-MT-gemma2-2b-lb-en"
dtype_trans = torch.bfloat16

tokenizer_trans = AutoTokenizer.from_pretrained(model_id_trans)
model_trans = AutoModelForCausalLM.from_pretrained(
    model_id_trans,
    device_map="cuda",
    torch_dtype=dtype_trans,
)

def translate_full_block(text_block):
    chat = [
        {"role": "user", "content": f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a helpful AI assistant for translation.
<|eot_id|><|start_header_id|>user<|end_header_id|>
Translate the following Luxembourgish input text into English.
Do not include any additional information or unrelated content.

{text_block}
"""},
    ]
    prompt = tokenizer_trans.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer_trans.encode(prompt, add_special_tokens=False, return_tensors="pt")
    outputs = model_trans.generate(input_ids=inputs.to(model_trans.device), max_new_tokens=len(text_block)*2) # Generous token limit

    translation_text = tokenizer_trans.decode(outputs[0]).split('<start_of_turn>model')[1]
    cleaned_text = translation_text.replace('<eos>', '').replace('<end_of_turn>', '').strip()
    return cleaned_text

full_luxembourgish_text = "\n".join(df_initial['sentence_lb'])
full_english_translation = translate_full_block(full_luxembourgish_text)

print("Full-text translation complete.")

# --- Cleanup Translation model from GPU ---
del model_trans, tokenizer_trans
torch.cuda.empty_cache()
gc.collect()
print("Cleaned up Translation model from memory.")


### =======================================================================
### 4. SEMANTIC ALIGNMENT
### =======================================================================
print("Step 4: Starting semantic alignment...")

# Split translated block into sentences
translated_sentences_en = re.split(r'(?<=[.!?])\s+', full_english_translation.strip())
original_sentences_lb = df_initial['sentence_lb'].tolist()

print(f"Aligning {len(original_sentences_lb)} original sentences with {len(translated_sentences_en)} translated sentences.")

# Load multilingual embedding model
embedding_model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2', device=device)
embeddings_lb = embedding_model.encode(original_sentences_lb, convert_to_tensor=True)
embeddings_en = embedding_model.encode(translated_sentences_en, convert_to_tensor=True)

# Calculate similarity
similarity_matrix = cosine_similarity(embeddings_lb.cpu().numpy(), embeddings_en.cpu().numpy())

# --- Cleanup Embedding model from GPU ---
del embedding_model, embeddings_lb, embeddings_en
torch.cuda.empty_cache()
gc.collect()
print("Cleaned up Embedding model from memory.")


# --- STAGE 4A: Handle 1-to-1 and 1-to-Many (Splits) ---
print("Stage 4a: Handling 1-to-1 and 1-to-many (split) mappings...")
best_parent_indices = np.argmax(similarity_matrix, axis=0)
df_en = pd.DataFrame({'sentence_en': translated_sentences_en, 'parent_index': best_parent_indices})
grouped_en = df_en.groupby('parent_index')['sentence_en'].apply(' '.join).reset_index()

# Merge results back. Rows that were part of a merge will have NaN here.
df_aligned = df_initial.merge(grouped_en, left_index=True, right_on='parent_index', how='left').drop(columns=['parent_index'])
df_splits_handled = df_aligned[df_aligned['sentence_en'].notna()].copy()


# --- STAGE 4B: Handle Many-to-1 (Merges) ---
print("Stage 4b: Handling many-to-1 (merge) mappings...")
unmapped_df = df_aligned[df_aligned['sentence_en'].isna()]
merged_rows = []

if not unmapped_df.empty:
    unmapped_indices = unmapped_df.index
    # For each unmapped LB sentence, find its best EN match
    best_target_for_unmapped = np.argmax(similarity_matrix[unmapped_indices, :], axis=1)

    unmapped_df['target_en_idx'] = best_target_for_unmapped

    # Group unmapped LB sentences by the EN sentence they point to
    grouped_merges = unmapped_df.groupby('target_en_idx')

    for target_idx, group_df in grouped_merges:
        merged_rows.append({
            'sentence_lb': ' || '.join(group_df['sentence_lb']),
            'start_time': group_df['start_time'].min(),
            'end_time': group_df['end_time'].max(),
            'sentence_en': translated_sentences_en[target_idx]
        })

df_merges_handled = pd.DataFrame(merged_rows)


### =======================================================================
### 5. FINAL COMBINATION AND SAVING
### =======================================================================
print("Step 5: Combining results and generating final files...")

# Combine the two dataframes
df_final = pd.concat([df_splits_handled, df_merges_handled], ignore_index=True)
df_final.sort_values(by='start_time', inplace=True)
df_final.reset_index(drop=True, inplace=True)

# Save to CSV
df_final.to_csv(output_csv_path, encoding='utf-8', index=False)
print(f"Results saved to {output_csv_path}")


### =======================================================================
### 6. SRT FILE GENERATION
### =======================================================================
def format_time(seconds):
    """Converts seconds to SRT time format HH:MM:SS,ms"""
    milliseconds = int((seconds - int(seconds)) * 1000)
    seconds = int(seconds)
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"

if os.path.exists(output_srt_path):
    os.remove(output_srt_path)

with open(output_srt_path, 'a', encoding='utf-8') as srt_file:
    for index, row in df_final.iterrows():
        srt_file.write(str(index + 1) + '\n')
        srt_file.write(f"{format_time(row['start_time'])} --> {format_time(row['end_time'])}\n")
        srt_file.write(row['sentence_en'] + '\n\n')

print(f"SRT subtitle file saved to {output_srt_path}")
print("\nProcedure finished successfully! 🎉")
