### =======================================================================
### 0. SETUP AND IMPORTS
### =======================================================================
import gc
import os
import re
from pathlib import Path
import pandas as pd
import librosa
import torch
import torchaudio
from scipy import signal
from transformers import WhisperForConditionalGeneration, WhisperProcessor, AutoModelForCausalLM, AutoTokenizer, pipeline
from demucs.pretrained import get_model
from demucs.apply import apply_model

# --- Configuration ---
original_audio = 'Ekaterina.opus'
audio_path = 'extracted_vocals.wav'  # Replace with your audio file
output_csv_path = 'Ekaterina.csv'
output_srt_path = 'Ekaterina.srt'

### =======================================================================
### 0. EXTRACT VOCALS WITH DEMUCS
### =======================================================================

class VocalExtractor:
    def __init__(self, model_name='htdemucs'):
        """
        Initialize the vocal extractor with specified model
        
        Args:
            model_name (str): Name of Demucs model to use
                             Options: 'htdemucs', 'hdemucs_mmi', 'demucs'
        """
        self.model = get_model(model_name)
        self.model.eval()
        
        # Move model to appropriate device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
    def extract_vocals(self, input_path, output_path, sr=44100):
        """
        Extract vocals from audio file and save result
        
        Args:
            input_path (str): Path to input audio file
            output_path (str): Path to save extracted vocals
            sr (int): Sample rate for processing
        """
        try:
            # Load and preprocess audio
            audio, orig_sr = librosa.load(input_path, sr=sr, mono=False)
            
            if audio.ndim == 1:
                audio = audio.unsqueeze(0)  # Convert to stereo if mono
            
            audio_tensor = torch.from_numpy(audio).float().unsqueeze(0).to(self.device)
            
            # Separate sources
            with torch.no_grad():
                sources = apply_model(self.model, audio_tensor)
            
            # Extract vocals (typically index 3 for htdemucs: [drums, bass, other, vocals])
            vocals = sources[0, 3].cpu()
            
            # Save extracted vocals
            os.makedirs(Path(output_path).parent, exist_ok=True)
            torchaudio.save(output_path, vocals, sr)
            
            print(f"Vocals extracted successfully to {output_path}")
            
        except Exception as e:
            print(f"Error processing {input_path}: {str(e)}")
    
    def batch_extract(self, input_dir, output_dir, extensions=['.mp3', '.wav', '.flac']):
        """
        Extract vocals from all audio files in a directory
        
        Args:
            input_dir (str): Input directory containing audio files
            output_dir (str): Output directory for extracted vocals
            extensions (list): File extensions to process
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        audio_files = []
        for ext in extensions:
            audio_files.extend(input_path.glob(f'*{ext}'))
        
        for audio_file in audio_files:
            output_file = output_path / f"{audio_file.stem}_vocals.wav"
            self.extract_vocals(str(audio_file), str(output_file))

# Usage example
if __name__ == "__main__":
    extractor = VocalExtractor('htdemucs')
    
    # Extract vocals from single file
    extractor.extract_vocals(original_audio, audio_path)
    
    # Extract vocals from all audio files in directory
    # extractor.batch_extract('input_directory/', 'output_directory/')

### =======================================================================
### 1. AUTOMATIC SPEECH RECOGNITION (ASR)
### =======================================================================
print("Step 1: Starting speech-to-text transcription...")

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_id_asr = "openai/whisper-large-v3"

model_asr = WhisperForConditionalGeneration.from_pretrained(
    model_id_asr,
    dtype=torch_dtype,
    low_cpu_mem_usage=True,
    use_safetensors=True,
    attn_implementation="eager"  # Force eager attention for compatibility :cite[1]
)
model_asr.to(device)

processor_asr = WhisperProcessor.from_pretrained(model_id_asr)

pipe_asr = pipeline(
    "automatic-speech-recognition",
    model=model_asr,
    tokenizer=processor_asr.tokenizer,
    feature_extractor=processor_asr.feature_extractor,
    torch_dtype=torch_dtype,
    device=device,
    return_timestamps="word",
)

'''
generate_kwargs = {
    "max_new_tokens": 256,  # Increased for longer musical phrases
    #"num_beams": 1,
    #"condition_on_prev_tokens": False,
    #"compression_ratio_threshold": 2.8,  # Higher threshold for music
    #"temperature": (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
    #"logprob_threshold": -1.2,  # Lower threshold to capture more uncertain speech
    #"no_speech_threshold": 0.3,  # Much lower threshold to not miss speech in music
    "language": "russian",
}
'''

generate_kwargs = {
    "max_new_tokens": 445,
    "num_beams": 1,
    "condition_on_prev_tokens": False,
    "compression_ratio_threshold": 1.35,  # zlib compression ratio threshold (in token space)
    "temperature": (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
    "logprob_threshold": -1.0,
    "no_speech_threshold": 0.6,
    "return_timestamps": True,
    "language": "russian",
}

# Load and preprocess audio
sample, sr = librosa.load(audio_path, sr=16000, mono=True)
duration = len(sample) / sr
print(f"Audio duration: {duration:.2f} seconds")

result = pipe_asr(sample, generate_kwargs=generate_kwargs)
chunks = result['chunks']

print(result['text'])
print("Transcription complete.")

# --- Cleanup ASR model from GPU ---
del model_asr, processor_asr, pipe_asr
torch.cuda.empty_cache()
gc.collect()
print("Cleaned up ASR model from memory.")

### =======================================================================
### 2. SENTENCE SPLITTING (REVISED)
### =======================================================================

import pandas as pd
import re

### =======================================================================
### 2. SENTENCE SPLITTING (REVISED)
### =======================================================================

import pandas as pd
import re

def final_sentence_chunker(data, max_pause_duration=2.0, min_words_for_split=5):
    """
    Splits ASR chunks into semantically meaningful sentences for translation.
    Uses a combination of punctuation, pause duration, and semantic cues.

    Args:
        data (list): Word-level timestamp chunks from Whisper.
        max_pause_duration (float): Maximum pause duration to consider for splitting.
        min_words_for_split (int): Minimum words required for a valid sentence.
    """
    if not data:
        return pd.DataFrame()

    # =================
    # Stage 1: Initial grouping with smarter splitting
    # =================
    lines = []
    current_line = []
    
    for i, item in enumerate(data):
        text = item['text'].strip()
        current_line.append(item)
        
        # Check if this is likely a sentence end
        is_sentence_end = False
        
        # Check for sentence-ending punctuation
        if re.search(r'[.!?…]\s*$', text):
            is_sentence_end = True
        
        # Check for longer pauses (but only if we have enough words)
        if i < len(data) - 1:
            next_start = data[i+1]['timestamp'][0]
            current_end = item['timestamp'][1]
            pause_duration = next_start - current_end
            
            # Longer pause indicates potential sentence boundary
            if pause_duration > max_pause_duration and len(current_line) >= min_words_for_split:
                is_sentence_end = True
        
        # Check for semantic cues (conjunctions, prepositions starting next sentence)
        if i < len(data) - 1:
            next_text = data[i+1]['text'].strip().lower()
            # If next word starts with a conjunction or preposition, it might be a new sentence
            semantic_cues = ['и', 'а', 'но', 'что', 'который', 'где', 'когда', 'если', 'чтобы']
            if next_text in semantic_cues and len(current_line) >= min_words_for_split:
                is_sentence_end = True
        
        # If we've determined this is a sentence end, finalize the line
        if is_sentence_end:
            lines.append(current_line)
            current_line = []
    
    # Add any remaining words
    if current_line:
        lines.append(current_line)

    # =================
    # Stage 2: Merge very short lines with previous ones
    # =================
    merged_lines = []
    for line in lines:
        if not merged_lines:
            merged_lines.append(line)
            continue
            
        # Calculate number of words in current and previous line
        prev_words = sum(len(item['text'].split()) for item in merged_lines[-1])
        curr_words = sum(len(item['text'].split()) for item in line)
        
        # If current line is very short, merge it with previous
        if curr_words < 3 and prev_words + curr_words < 12:
            merged_lines[-1].extend(line)
        else:
            merged_lines.append(line)

    # =================
    # Stage 3: Create final sentences
    # =================
    final_sentences = []
    for line in merged_lines:
        sentence_text = ' '.join(item['text'] for item in line).strip()
        # Clean up punctuation spacing
        sentence_text = re.sub(r'\s([,.!?;—])', r'\1', sentence_text)
        sentence_text = re.sub(r'([(])\s', r'\1', sentence_text)
        sentence_text = re.sub(r'\s([)])', r'\1', sentence_text)
        
        start_time = line[0]['timestamp'][0]
        end_time = line[-1]['timestamp'][1]
        
        final_sentences.append({
            'sentence_ru': sentence_text,
            'start_time': start_time,
            'end_time': end_time
        })
    
    return pd.DataFrame(final_sentences)

df = final_sentence_chunker(chunks)
print("Step 2: Improved sentence splitting complete.")
print(f"Found {len(df)} sentence chunks.")
print(df)

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
    from sentence_transformers import SentenceTransformer, util
    
    print("Rematching multiple translations using semantic similarity...")
    
    # Load the multilingual embedding model
    embedder = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
    
    # Get embeddings for all Russian sentences
    ru_embeddings = embedder.encode(df_rus['sentence_ru'].tolist(), convert_to_tensor=True)
    
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
    full_russian_text = "\n".join(df_rus['sentence_ru'].tolist())
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
    
    df_rus['sentence_en'] = english_lines
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
    full_russian_text = "\n".join(df_rus['sentence_ru'].tolist())
    
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
    
    df_rus['sentence_en'] = english_lines
    return df_rus

def select_best_translation(translations, russian_text):
    """Select the best translation using embedding similarity"""
    from sentence_transformers import SentenceTransformer, util
    
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
print(f"English sentences: {len(df['sentence_en'])}")

# Save to CSV
df.to_csv(output_csv_path, encoding='utf-8', index=False)
print(f"Results saved to {output_csv_path}")

### =======================================================================
### 5. SRT FILE GENERATION
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
    for index, row in df.iterrows():
        srt_file.write(str(index + 1) + '\n')
        srt_file.write(f"{format_time(row['start_time'])} --> {format_time(row['end_time'])}\n")
        srt_file.write(row['sentence_en'] + '\n\n')

print(f"SRT subtitle file saved to {output_srt_path}")
print("\nProcedure finished successfully! 🎉")
