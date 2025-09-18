# autosubs

Automatically generates subtitles in SRT format for a video.

You need to extract the audio beforehand (for instance, with ffmeg, check the script for an example).

Work in progress, very precarious but functional at the moment.

Transcribes audio in any language supported by Whisper (ca. 100 in total of which ca. 50 with acceptable quality).

Translates the subtitles to any language supported by No Language Left Behind (NLLB); ca. 200 languages.

The speed in a RTX 3090 is about 5 minutes per hour of audio.

Additional script specifically to Luxembourgish audio -> English subtitles. 
