import os
import gc
import torch
import librosa
import pandas as pd
import nemo.collections.asr as nemo_asr
from sentence_transformers import SentenceTransformer, util
from audio_separator.separator import Separator
from transformers import AutoModelForCausalLM, AutoTokenizer

###############################################################
###                        VARIABLES                        ###
###############################################################

input_audio = 'От_большого_ума.opus'
output_csv_path = 'Yanka.csv'
output_srt_path = 'Yanka.srt'

###############################################################
###                     VOCAL ISOLATION                     ###
###############################################################

separator = Separator(output_single_stem='vocals')
separator.load_model(model_filename='model_bs_roformer_ep_317_sdr_12.9755.ckpt')
output_names = {'Vocals': 'isolated_vocals'}
vocal_audio = separator.separate(input_audio, output_names)
sample, sr = librosa.load(vocal_audio[0], sr=16000)

# Clean up the first model
del separator
torch.cuda.empty_cache()
gc.collect()

###############################################################
###                     TRANSCRIPTION                       ###
###############################################################

asr_model = nemo_asr.models.ASRModel.from_pretrained(model_name='nvidia/parakeet-tdt-0.6b-v3')

transcript = asr_model.transcribe(sample, timestamps=True)
word_timestamps = transcript[0].timestamp['word']
raw_text = transcript[0].timestamp['segment'][0]['segment']
#char_timestamps = transcript[0].timestamp['char']
print(raw_text)

# Clean up the first model
del asr_model
torch.cuda.empty_cache()
gc.collect()

################################################################
###                  SENTENCE CHUNKING                       ###
################################################################

def sentence_chunking(word_timestamps, hard_stops, soft_stops, min_words_soft=5, gap_threshold=2.0):

    sentences = []
    current_sentence = []
    current_start = None
    current_end = None
    
    for i, stamp in enumerate(word_timestamps):
        word = stamp['word']
        
        # Check for significant time gap from previous word
        if (current_end is not None and 
            stamp['start'] - current_end > gap_threshold and 
            current_sentence):
            # Finalize current sentence due to time gap
            sentence_text = ' '.join(current_sentence)
            sentences.append((sentence_text, current_start, current_end))
            current_sentence = []
            current_start = None
            current_end = None
        
        # Start new sentence if needed
        if current_start is None:
            current_start = stamp['start']
        
        current_sentence.append(word)
        current_end = stamp['end']
        
        # Check if this word should end the sentence
        should_end = False
        is_last_word = (i == len(word_timestamps) - 1)
        
        # HARD stops always end the sentence
        ends_with_hard = any(word.endswith(stop) for stop in hard_stops)
        if ends_with_hard:
            should_end = True
        
        # SOFT stops only end if sentence is long enough
        elif not should_end:  # Only check if we haven't already decided to end
            ends_with_soft = any(word.endswith(stop) for stop in soft_stops)
            if ends_with_soft and len(current_sentence) >= min_words_soft:
                should_end = True
        
        # Last word always ends the sentence
        if is_last_word:
            should_end = True
        
        if should_end:
            sentence_text = ' '.join(current_sentence)
            sentences.append((sentence_text, current_start, current_end))
            current_sentence = []
            current_start = None
            current_end = None
    
    return sentences

# Punctuation categories
HARD_STOPS = ('。', '．', '.', '！', '!', '?', '？')
SOFT_STOPS = (',', '，', ':' , ';', '；', '—', '–')

sentences = sentence_chunking(
    word_timestamps, 
    HARD_STOPS, 
    SOFT_STOPS, 
    min_words_soft=5,
    gap_threshold=2.0
)

df = pd.DataFrame(sentences, columns = ['Text', 'Start', 'End'])

### =======================================================================
### 3. ROBUST TRANSLATION WITH LINE-COUNT MATCHING AND ITERATIVE REFINEMENT
### =======================================================================
print("Step 3: Starting robust translation with line-count matching and iterative refinement...")

def rematch_multiple_translations(df_rus, translations, threshold=0.1):
    """
    Apply line rematching to multiple translations using semantic similarity.
    
    Args:
        df_rus: DataFrame with Russian sentences
        translations: List of translation strings (each with same number of lines as df_rus)
        threshold: Minimum improvement in cosine similarity to justify a swap
    
    Returns:
        List of rematched translation strings
    """
    
    print("Rematching multiple translations using semantic similarity...")
    
    # Load the multilingual embedding model
    embedder = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
    
    # Get embeddings for all Russian sentences
    ru_embeddings = embedder.encode(df_rus['Text'].tolist(), convert_to_tensor=True)
    
    rematched_translations = []
    
    for i, translation in enumerate(translations):
        print(f"Rematching translation {i+1}/{len(translations)}...")
        
        # Split the translation into lines
        english_lines = [line.strip() for line in translation.split('\n') if line.strip()]
        
        if len(english_lines) != len(df_rus):
            print(f"Warning: Translation {i+1} has {len(english_lines)} lines, expected {len(df_rus)}")
            rematched_translations.append(translation)
            continue
        
        # Get embeddings for English sentences
        en_embeddings = embedder.encode(english_lines, convert_to_tensor=True)
        
        # Calculate current similarity scores
        current_similarities = []
        for j in range(len(df_rus)):
            similarity = util.pytorch_cos_sim(ru_embeddings[j], en_embeddings[j]).item()
            current_similarities.append(similarity)
        
        total_current_similarity = sum(current_similarities)
        
        # Try swapping consecutive lines to see if it improves similarity
        improved = True
        iteration = 0
        max_iterations = len(df_rus)
        
        current_lines = english_lines.copy()
        current_embeddings = en_embeddings.clone()
        
        while improved and iteration < max_iterations:
            improved = False
            iteration += 1
            
            for j in range(len(df_rus) - 1):
                # Calculate similarity if we swap lines j and j+1
                original_sim_j = util.pytorch_cos_sim(ru_embeddings[j], current_embeddings[j]).item()
                original_sim_j1 = util.pytorch_cos_sim(ru_embeddings[j+1], current_embeddings[j+1]).item()
                
                swapped_sim_j = util.pytorch_cos_sim(ru_embeddings[j], current_embeddings[j+1]).item()
                swapped_sim_j1 = util.pytorch_cos_sim(ru_embeddings[j+1], current_embeddings[j]).item()
                
                original_total = original_sim_j + original_sim_j1
                swapped_total = swapped_sim_j + swapped_sim_j1
                
                # Check if swapping improves similarity beyond threshold
                if swapped_total > original_total + threshold:
                    # Swap the English sentences and their embeddings
                    current_lines[j], current_lines[j+1] = current_lines[j+1], current_lines[j]
                    current_embeddings[j], current_embeddings[j+1] = current_embeddings[j+1].clone(), current_embeddings[j].clone()
                    
                    improved = True
                    break  # Restart checking from the beginning
        
        # Calculate final similarity scores
        final_similarities = []
        for j in range(len(df_rus)):
            similarity = util.pytorch_cos_sim(ru_embeddings[j], current_embeddings[j]).item()
            final_similarities.append(similarity)
        
        total_final_similarity = sum(final_similarities)
        improvement = total_final_similarity - total_current_similarity
        
        print(f"Translation {i+1}: Improvement {improvement:.4f}")
        
        # Add the rematched translation
        rematched_translations.append("\n".join(current_lines))
    
    return rematched_translations

def translate_with_line_count_matching(df_rus):
    """
    Generates translations until we have 6 with matching line counts,
    then uses Hunyuan-MT-Chimera-7B to refine until we get a matching refined version.
    Applies semantic rematching to all intermediate and final translations.
    """
    print("=== Loading Translation Model: Hunyuan-MT-7B ===")
    
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained("tencent/Hunyuan-MT-7B", trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        "tencent/Hunyuan-MT-7B",
        device_map="auto",
        dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    # Prepare the full Russian text
    full_russian_text = "\n".join(df_rus['Text'].tolist())
    num_russian_lines = len(df_rus)
    
    print(f"Russian text has {num_russian_lines} lines. Collecting matching translations...")
    
    # Generate translations until we have 6 with matching line counts
    raw_matching_translations = []
    attempt_count = 0
    max_attempts = 21  # Prevent infinite loops
    
    while len(raw_matching_translations) < 6 and attempt_count < max_attempts:
        attempt_count += 1
        print(f"Generation attempt {attempt_count}...")
        
        # Use the good prompt that worked well
        prompt = (
            f"Translate the following Russian song lyrics to English. "
            f"The translation should match the orginal verse by verse and line by line. "
            f"Preserve the meaning as accurately as possible:\n\n"
            f"{full_russian_text}\n\n"
            f"English translation:"
        )
        
        messages = [{"role": "user", "content": prompt}]
        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(model.device)

        # Use varied generation parameters
        gen_params = {
            "max_new_tokens": 1024,
            "do_sample": True,
            "top_k": 30,
            "top_p": 0.85,
            "repetition_penalty": 1.1,
            "temperature": 0.3 + (attempt_count % 5 * 0.1),  # Vary temperature
        }

        with torch.inference_mode():
            outputs = model.generate(inputs, **gen_params)
        
        generated_tokens = outputs[0][inputs.shape[1]:]
        translation = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
        
        # Check if line count matches
        english_lines = [line.strip() for line in translation.split('\n') if line.strip()]
        print(english_lines)
        if len(english_lines) == num_russian_lines:
            raw_matching_translations.append(translation)
            print(f"✓ Found matching translation {len(raw_matching_translations)}/6")
        else:
            print(f"✗ Translation has {len(english_lines)} lines, expected {num_russian_lines}")
    
    # Clean up the first model
    del model, tokenizer
    torch.cuda.empty_cache()
    gc.collect()
    
    if not raw_matching_translations:
        print("ERROR: Could not find any translations with matching line count.")
        # Fallback to simple translation
        return simple_translation_fallback(df_rus)
    
    print(f"Collected {len(raw_matching_translations)} matching translations. Now applying semantic rematching...")
    
    # Apply semantic rematching to all intermediate translations
    matching_translations = rematch_multiple_translations(df_rus, raw_matching_translations)
    
    print("Now refining with Chimera...")
    
    # Load the Chimera model for refinement
    chimera_tokenizer = AutoTokenizer.from_pretrained("tencent/Hunyuan-MT-Chimera-7B", trust_remote_code=True)
    chimera_model = AutoModelForCausalLM.from_pretrained(
        "tencent/Hunyuan-MT-Chimera-7B",
        device_map="auto",
        dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    
    # Refine until we get a matching translation
    refined_translation = None
    refinement_attempts = 0
    max_refinement_attempts = 6
    
    while refined_translation is None and refinement_attempts < max_refinement_attempts:
        refinement_attempts += 1
        print(f"Refinement attempt {refinement_attempts}...")
        
        # Prepare the prompt for the Chimera model
        translations_text = "\n\n".join([f"Translation {i+1}:\n{trans}" for i, trans in enumerate(matching_translations)])
        
        chimera_prompt = (
            f"I need you to refine an English translation of a Russian song. "
            f"Below are several translations of the same Russian song. "
            f"Please create a refined version that combines the best elements of each translation "
            f"while maintaining the exact same line structure as the original Russian.\n\n"
            f"Original Russian text (has {num_russian_lines} lines):\n{full_russian_text}\n\n"
            f"Available translations:\n{translations_text}\n\n"
            f"CRITICAL: Your refined translation MUST have exactly {num_russian_lines} lines, "
            f"matching the original Russian structure.\n\n"
            f"Refined English translation:"
        )
        
        chimera_messages = [{"role": "user", "content": chimera_prompt}]
        chimera_inputs = chimera_tokenizer.apply_chat_template(
            chimera_messages,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(chimera_model.device)
        
        chimera_gen_params = {
            "max_new_tokens": 1024,
            "do_sample": True,
            "top_k": 30,
            "top_p": 0.85,
            "repetition_penalty": 1.1,
            "temperature": 0.3,
        }
        
        with torch.inference_mode():
            chimera_outputs = chimera_model.generate(chimera_inputs, **chimera_gen_params)
        
        chimera_generated_tokens = chimera_outputs[0][chimera_inputs.shape[1]:]
        candidate_translation = chimera_tokenizer.decode(chimera_generated_tokens, skip_special_tokens=True).strip()
        
        # Check if the refined translation matches the line count
        english_lines = [line.strip() for line in candidate_translation.split('\n') if line.strip()]
        if len(english_lines) == num_russian_lines:
            refined_translation = candidate_translation
            print("✓ Refined translation matches line count!")
        else:
            print(f"✗ Refined translation has {len(english_lines)} lines, expected {num_russian_lines}")
    
    # Clean up the Chimera model
    del chimera_model, chimera_tokenizer
    torch.cuda.empty_cache()
    gc.collect()
    
    # Apply semantic rematching to the refined translation
    if refined_translation:
        print("Applying semantic rematching to refined translation...")
        refined_translations = rematch_multiple_translations(df_rus, [refined_translation])
        refined_translation = refined_translations[0]
        english_lines = [line.strip() for line in refined_translation.split('\n') if line.strip()]
        print("✓ Refined translation rematched!")
    else:
        print("Using best matching translation from initial set.")
        # Use the translation with the highest similarity score as fallback
        english_lines = select_best_translation(matching_translations, full_russian_text)
    
    df_rus['Translation'] = english_lines
    print("Translation process completed successfully.")
    return df_rus

def simple_translation_fallback(df_rus):
    """Fallback function if no matching translations are found"""
    print("Using simple translation fallback...")
    
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained("tencent/Hunyuan-MT-7B", trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        "tencent/Hunyuan-MT-7B",
        device_map="auto",
        dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    
    # Prepare the full Russian text
    full_russian_text = "\n".join(df_rus['Text'].tolist())
    
    # Use the good prompt
    prompt = (
        f"Translate the following Russian song lyrics to English. "
        f"Maintain the same line structure and preserve the meaning as accurately as possible:\n\n"
        f"{full_russian_text}\n\n"
        f"English translation:"
    )
    
    messages = [{"role": "user", "content": prompt}]
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)

    gen_params = {
        "max_new_tokens": 1024,
        "do_sample": True,
        "top_k": 30,
        "top_p": 0.85,
        "repetition_penalty": 1.1,
        "temperature": 0.3,
    }

    with torch.inference_mode():
        outputs = model.generate(inputs, **gen_params)
    
    generated_tokens = outputs[0][inputs.shape[1]:]
    translation = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
    
    # Clean up
    del model, tokenizer
    torch.cuda.empty_cache()
    gc.collect()
    
    # Split into lines
    english_lines = [line.strip() for line in translation.split('\n') if line.strip()]
    
    # If line count doesn't match, just use what we have with padding
    if len(english_lines) != len(df_rus):
        print("Warning: Line count still doesn't match after fallback")
        if len(english_lines) < len(df_rus):
            # Pad with empty strings
            english_lines.extend([""] * (len(df_rus) - len(english_lines)))
        else:
            # Truncate
            english_lines = english_lines[:len(df_rus)]
    
    df_rus['Translation'] = english_lines
    return df_rus

def select_best_translation(translations, russian_text):
    """Select the best translation using embedding similarity"""
    
    # Use the multilingual model
    embedder = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
    
    # Get embeddings for Russian text and all translations
    ru_embedding = embedder.encode(russian_text, convert_to_tensor=True)
    en_embeddings = embedder.encode(translations, convert_to_tensor=True)
    
    # Compute similarities
    similarities = util.pytorch_cos_sim(ru_embedding, en_embeddings)[0]
    
    # Find the best matching translation
    best_idx = torch.argmax(similarities).item()
    best_translation = translations[best_idx]
    
    # Split into lines
    return [line.strip() for line in best_translation.split('\n') if line.strip()]

# Replace the translation function call with:
df = translate_with_line_count_matching(df)

# Now the lengths will always match
print(f"Russian sentences: {len(df)}")
print(f"English sentences: {len(df['Translation'])}")

# Save to CSV
df.to_csv(output_csv_path, encoding='utf-8', index=False)
print(f"Results saved to {output_csv_path}")

### =======================================================================
### 5. SIMPLE NON-OVERLAPPING SRT GENERATION
### =======================================================================
def format_time(seconds):
    """Converts seconds to SRT time format HH:MM:SS,ms"""
    milliseconds = int((seconds - int(seconds)) * 1000)
    seconds = int(seconds)
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"

# Get audio duration
audio_duration = librosa.get_duration(filename=vocal_audio[0])
print(f"Audio duration: {audio_duration:.2f} seconds")

if os.path.exists(output_srt_path):
    os.remove(output_srt_path)

with open(output_srt_path, 'a', encoding='utf-8') as srt_file:
    for index, row in df.iterrows():
        # Calculate end time with guaranteed gap
        if index < len(df) - 1:
            next_start = df.iloc[index + 1]['Start']
            # End 0.3 seconds before next subtitle starts
            end_time = next_start - 0.3
            # Ensure minimum duration of 1.0 second
            if end_time - row['Start'] < 1.0:
                end_time = row['Start'] + 1.0
            # Don't let it overlap even with minimum duration
            if end_time >= next_start:
                end_time = next_start - 0.1  # Smallest possible gap
        else:
            end_time = audio_duration  # Last subtitle
        
        srt_file.write(str(index + 1) + '\n')
        srt_file.write(f"{format_time(row['Start'])} --> {format_time(end_time)}\n")
        srt_file.write(row['Translation'] + '\n\n')
        
        print(f"Subtitle {index+1}: {row['Start']:.2f}s -> {end_time:.2f}s "
              f"(duration: {end_time-row['Start']:.2f}s)")

print(f"SRT subtitle file saved to {output_srt_path}")
