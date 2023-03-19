import re
import json
from pathlib import Path
from pydub import AudioSegment
from yt_dlp import YoutubeDL
import torch
import whisper
from huggingface_hub import notebook_login
from pyannote.audio import Pipeline
import locale
import whisper
import torch


def millisec(timeStr):
    spl = timeStr.split(":")
    s = (int)((int(spl[0]) * 60 * 60 +
               int(spl[1]) * 60 + float(spl[2])) * 1000)
    return s


Source = 'Youtube'
video_url = "https://youtu.be/NSp2fEQ6wyA"
output_path = "/content/"
output_path = str(Path(output_path))

Path(output_path).mkdir(parents=True, exist_ok=True)
video_title = ""
video_id = ""

if Source == "Youtube":
    with YoutubeDL() as ydl:
        info_dict = ydl.extract_info(video_url, download=False)
        video_title = info_dict.get('title', None)
        video_id = info_dict.get('id', None)
        print("Title: " + video_title)

if Source == "Youtube":
    audio_output_file = f"{str(output_path)}/input.wav"
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '192',
        }],
        'outtmpl': audio_output_file
    }
    with YoutubeDL(ydl_opts) as ydl:
        ydl.download([video_url])

spacermilli = 2000
spacer = AudioSegment.silent(duration=spacermilli)

audio = AudioSegment.from_wav("input.wav")
audio = spacer.append(audio, crossfade=0)
audio.export('input_prep.wav', format='wav')

access_token = "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"
if not (access_token):
    notebook_login()

pipeline = Pipeline.from_pretrained(
    'pyannote/speaker-diarization', use_auth_token=(access_token) or True)

DEMO_FILE = {'uri': 'blabla', 'audio': 'input_prep.wav'}
dz = pipeline(DEMO_FILE)

with open("diarization.txt", "w") as text_file:
    text_file.write(str(dz))

print(*list(dz.itertracks(yield_label=True))[:10], sep="\n")

dzs = open('diarization.txt').read().splitlines()
groups = []
g = []
lastend = 0

for d in dzs:
    if g and (g[0].split()[-1] != d.split()[-1]):
        groups.append(g)
        g = []

    g.append(d)

    end = re.findall('[0-9]+:[0-9]+:[0-9]+\.[0-9]+', string=d)[1]
    end = millisec(end)
    if (lastend > end):
        groups.append(g)
        g = []
    else:
        lastend = end
if g:
    groups.append(g)
print(*groups, sep='\n')

audio = AudioSegment.from_wav("input_prep.wav")
gidx = -1
for g in groups:
    start = re.findall('[0-9]+:[0-9]+:[0-9]+\.[0-9]+', string=g[0])[0]
    end = re.findall('[0-9]+:[0-9]+:[0-9]+\.[0-9]+', string=g[-1])[1]
    start = millisec(start)
    end = millisec(end)  # - spacermilli
    gidx += 1
    audio[start:end].export(str(gidx) + '.wav', format='wav')
    print(f"group {gidx}: {start}--{end}")

locale.getpreferredencoding = lambda: "UTF-8"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = whisper.load_model('large', device=device)

for i in range(len(groups)):
    audiof = str(i) + '.wav'
    # , initial_prompt=result.get('text', ""))
    result = model.transcribe(
        audio=audiof, language='en', word_timestamps=True)
    with open(str(i)+'.json', "w") as outfile:
        json.dump(result, outfile, indent=4)
