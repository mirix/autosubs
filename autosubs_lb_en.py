import pandas as pd
import torch
import gc
import os
import librosa
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, AutoModelForCausalLM, AutoTokenizer, pipeline

audio_path = "letz.mp3"

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_id = "unilux/whisper-medium-v1-luxembourgish"

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
)
model.to(device)

processor = AutoProcessor.from_pretrained(model_id)

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
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
    "no_speech_threshold": 0.9,  # Increased from 0.6
    "return_timestamps": "word",
    "language": "luxembourgish",
}

sample, sr = librosa.load(audio_path, sr=16_000, mono=True)

result = pipe(sample, generate_kwargs=generate_kwargs)

results = result['chunks']

def split_sentences(data):
    primary_stops = ['。', '．', '.', '！', '!', '?', '？']
    secondary_stops = [':', ';']
    
    sentences = []
    start_times = []
    end_times = []
    
    current_sentence = []
    current_start = None
    current_end = None
    
    for i, item in enumerate(data):
        text = item['text']
        start, end = item['timestamp']
        
        if current_start is None:
            current_start = start
            
        current_sentence.append(text)
        current_end = end
        
        # Check if the current token ends with any primary stop character
        ends_with_primary = any(text.endswith(stop) for stop in primary_stops)
        
        # Check if the current token ends with any secondary stop character
        ends_with_secondary = any(text.endswith(stop) for stop in secondary_stops)
        
        # Determine if we are at the last token
        is_last_token = i == len(data) - 1
        
        # Check the next token if it exists
        next_token = data[i + 1] if i + 1 < len(data) else None
        if next_token:
            next_text = next_token['text'].strip()  # Strip leading/trailing spaces
            # Check if next token starts with uppercase letter OR a digit
            next_starts_upper_or_digit = next_text and (next_text[0].isupper() or next_text[0].isdigit())
        else:
            next_starts_upper_or_digit = False
        
        current_sentence_str = ''.join(current_sentence)
        current_length = len(current_sentence_str)
        
        # Split conditions:
        # 1. If it's the last token, always split.
        # 2. If the token ends with a primary stop and the next token starts with uppercase or digit, split.
        # 3. If the sentence is longer than 44 characters and the token ends with a secondary stop, split.
        if is_last_token:
            sentences.append(current_sentence_str)
            start_times.append(current_start)
            end_times.append(current_end)
            current_sentence = []
            current_start = None
            current_end = None
        elif ends_with_primary and next_starts_upper_or_digit:
            sentences.append(current_sentence_str)
            start_times.append(current_start)
            end_times.append(current_end)
            current_sentence = []
            current_start = None
            current_end = None
        elif current_length > 44 and ends_with_secondary:
            sentences.append(current_sentence_str)
            start_times.append(current_start)
            end_times.append(current_end)
            current_sentence = []
            current_start = None
            current_end = None
            
    # Handle any remaining tokens in current_sentence after loop
    if current_sentence:
        sentences.append(''.join(current_sentence))
        start_times.append(current_start)
        end_times.append(current_end)
        
    return {
        'sentences': sentences,
        'start_times': start_times,
        'end_times': end_times
    }

sentences = split_sentences(results)
df = pd.DataFrame(sentences)

del model
torch.cuda.empty_cache()
gc.collect()

### TRANSLATE ###

tmodel_id = "etamin/Letz-MT-gemma2-2b-lb-en"
dtype = torch.bfloat16

tokenizer = AutoTokenizer.from_pretrained(tmodel_id)
tmodel = AutoModelForCausalLM.from_pretrained(
    tmodel_id,
    device_map="cuda",
    torch_dtype=dtype,)

def translate_sentences(row):
    chat = [
        { "role": "user", "content": f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a helpful AI assistant for translation.<|eot_id|>
<|start_header_id|>user<|end_header_id|>

Translate the Luxembourgish input text into English.
Do not include any additional information or unrelated content.

{row}
""" },
    ]
    prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt")
    outputs = tmodel.generate(input_ids=inputs.to(tmodel.device), max_new_tokens=150)
    
    # Make parsing slightly more robust by stripping whitespace
    translation_text = tokenizer.decode(outputs[0]).split('<start_of_turn>model')[1].replace('<eos>', '').replace('<end_of_turn>', '').replace('\n', '').strip()
    return translation_text

# Apply translation
df['english'] = df['sentences'].apply(translate_sentences)

# Save results
df.to_csv('letz.csv', encoding='utf-8', index=False)

# Cleanup (correct order and objects)
del tmodel, tokenizer
torch.cuda.empty_cache()
gc.collect()

sep = ' --> '
sub_name = 'letz.srt'

def format_time(seconds):
    milliseconds = int((seconds - int(seconds)) * 1000)
    seconds = int(seconds)
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"

if os.path.exists(sub_name): os.remove(sub_name)

with open(sub_name, 'a') as srt:
    for index, row in df.iterrows():
        srt.write(str(index+1) + '\n')
        srt.write(format_time(row['start_times']) + sep + format_time(row['end_times']) + '\n')
        srt.write(row['english'] + '\n\n')
