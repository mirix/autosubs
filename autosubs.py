import os
import gc
import pandas as pd
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

import stable_whisper

# 1. Find out audio format: ffmpeg -i Istorii_Despre_Vlad_Voievod_Draculea_2015.mp4
# 2. Extract audio from video:
#ffmpeg -i Istorii_Despre_Vlad_Voievod_Draculea_2015.mp4 -acodec copy vlad.aac

audio_file = 'vlad.aac'

# List of supported languages for NLLB (and their corresponding FLORES code)
# (~200 supported)
# https://huggingface.co/spaces/UNESCO/nllb/blob/d0a2f64cdae2fae119a127dba13609cb1d0b7542/flores.py

src_lang = "ron_Latn"
tgt_lang = "eng_Latn"

token = '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'
model_id = "facebook/nllb-200-3.3B"


### TRANSCRIPTION ###
# audio2txt

# List of supported languages for Whisper transcription
# (~100 supported of which ~50 meet the quality standards)
# https://platform.openai.com/docs/guides/speech-to-text

language = 'Romanian'

base_name = audio_file.split('.')[0]
sub_name = base_name + '_en.srt'

model = stable_whisper.load_model('large-v3')
result = model.transcribe(audio_file, language=language, regroup='sp=.* /。/?/？/．/!/！')
#vad=True, vad_threshold=0.35, denoiser="demucs"

results = result.to_dict()['segments']

### Sentence splitting ###

word_list = []
start_list = []
end_list =[]
for segment in results:
    for word in segment['words']:
        word_list.append(word['word'].replace('...', ''))
        start_list.append(word['start'])
        end_list.append(word['end'])

full_text = ''.join([str(i) for i in word_list])

# Max sentence length (approx)
max_length = 44
# Sentences shorter than 50 words are split on the following characters
stops = ('。', '．', '.', '！', '!', '?', '？')
# For longer sentences, the comma is also a splitting mark
extra_stops = (',', '，')
# The following abbreviations are excluded
abbre = ('Dr.', 'Mr.', 'Mrs.', 'Ms.', 'vs.', 'Prof.', 'i.e.')

chunk_list = []
for i, word in enumerate(full_text.split()):
    if i == 0:
        start0 = start_list[i]
        end0 = end_list[i]
        word0 = word
        if word0.endswith(stops) and not word0.endswith(abbre):
            chunk_list.append((word0, start0, end0))
    else:
        if len(word0.split()) <= max_length:
            if not word0.endswith(stops) or word0.endswith(abbre):
                word1 = word
                word0 = word0 + ' ' + word1
                start0 = start0
                end0 = end_list[i]
                if word0.endswith(stops) and not word0.endswith(abbre):
                    chunk_list.append((word0, start0, end0))
            else:
                word0 = word
                start0 = start_list[i]
                end0 = end_list[i]
                if word0.endswith(stops) and not word0.endswith(abbre):
                    chunk_list.append((word0, start0, end0))
        if len(word0.split()) > max_length:
            if not word0.endswith(stops + extra_stops) or word0.endswith(abbre):
                word1 = word
                word0 = word0 + ' ' + word1
                start0 = start0
                end0 = end_list[i]
                if word0.endswith(stops + extra_stops) and not word0.endswith(abbre):
                    chunk_list.append((word0, start0, end0))
            else:
                word0 = word
                start0 = start_list[i]
                end0 = end_list[i]
                if word0.endswith(stops + extra_stops) and not word0.endswith(abbre):
                    chunk_list.append((word0, start0, end0))

df = pd.DataFrame(chunk_list, columns = ['Text', 'Start', 'End'])

del model
gc.collect()

### TRANSLATION ###
# txt2txt

tokenizer = AutoTokenizer.from_pretrained(model_id, token=token, src_lang=src_lang, device_map="cuda:0")
model_trans = AutoModelForSeq2SeqLM.from_pretrained(model_id, token=token, device_map="cuda:0")

def trans_sent(text):

    inputs = tokenizer(text, return_tensors="pt").to('cuda:0')

    translated_tokens = model_trans.generate(
        **inputs, forced_bos_token_id=tokenizer.convert_tokens_to_ids(tgt_lang), max_length=256
    ).to('cuda:0')

    output = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]

    return output

df['English'] = df['Text'].apply(trans_sent)

#df.to_csv('test_srt.csv', encoding='utf-8', index=False)
#df = pd.read_csv('test_srt.csv')

sep = ' --> '

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
        srt.write(format_time(row['Start']) + sep + format_time(row['End']) + '\n')
        srt.write(row['English'] + '\n\n')

#result.to_srt_vtt('vlad2.srt', word_level=False)
