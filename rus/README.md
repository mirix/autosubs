I have migrated the automated Russian subtitling and translation system entirely from Parakeet (multilingual) to GigaAM (Russian only).

I am using the following GigaAM fork (currently, it does not work with the official code):

https://github.com/Maksim-Goncharovskiy/GigaAM

Word precision has improved substantially.

However, GigaAM does not seem to have been trained on songs and still struggles with overlapping music and voice, overlapping voices and polyphony.

A few test examples can be found here:

https://rumble.com/c/c-7816716

INSTALL

You need to have both Python scripts on the same folder and install the dependencies.

I may provide a requirements file a some point.

USAGE

You can run the main script as:

python rus_subs_gigaam_auto.py <Project Name> <Youtube ID or "URL"> <0, 1 or Nothing>

The third argument is optional:

0 - Do not run vocal isolation (discouraged).
1 - Run vocal isolation (Default, recommended).

REQUIREMENTS

GigaAM from the aforementioned repo only.

Transformers, sentence_transformers, torch, audio_separator, yt_dlp, fmpeg-python, librosa, soundfile, Pandas, lxml, zipfile. Plus cuda, ffmeg, etc.

The current code runs confortably on 8 GB of VRAM, perhaps even 6 GB. 

You can further reduce the VRAM footprint by switching the translation models to 4-bit (currently 8-bit).

