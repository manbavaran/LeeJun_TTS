import torch
from transformers import VitsModel, VitsTokenizer
import soundfile as sf
import os

base_dir = os.path.dirname(os.path.abspath(__file__))
audio_dir = os.path.abspath(os.path.join(base_dir, "..", "audios"))

audio_path = os.path.join(audio_dir, "mms_tts_korean.wav")

if not os.path.exists(audio_dir):
    os.makedirs(audio_dir)

# 1. 모델/토크나이저 다운로드 및 불러오기
model = VitsModel.from_pretrained("facebook/mms-tts-kor")
tokenizer = VitsTokenizer.from_pretrained("facebook/mms-tts-kor")

# 2. 텍스트 토크나이즈
inputs = tokenizer("안녕하세요! 오늘 날씨 좋네요.", return_tensors="pt")
inputs["input_ids"] = inputs["input_ids"].long()

# 3. 음성 생성
with torch.no_grad():
    audio = model(**inputs).waveform  # [batch, sample]

# 4. wav 파일로 저장 (16kHz)
sf.write(audio_path, audio.cpu().numpy()[0], 16000)



