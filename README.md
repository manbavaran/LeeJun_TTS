# LeeJun Memorial TTS

A simple Korean Text-to-Speech (TTS) tool for generating narration for the July 14th memorial video of Martyr Lee Jun.  
This project uses [facebook/mms-tts-kor](https://huggingface.co/facebook/mms-tts-kor) for natural-sounding Korean speech synthesis via the Hugging Face Transformers library.

---

## ✨ Project Overview

This repository provides a minimal Python TTS pipeline that synthesizes Korean audio for memorial or narration use cases.  
It was developed for use in a tribute video for Lee Jun, but can be adapted for other Korean TTS projects.

---

## 🚀 Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/your-username/leejun-memorial-tts.git
cd leejun-memorial-tts/src
```

### 2. Set up the virtual environment

```bash
python -m venv ../venv310
../venv310/Scripts/activate  # On Windows
```

### 3. Install dependencies

```bash
pip install -r ../requirements.txt
```

### 4. Run the TTS script

```bash
python mms_tts_korean.py
```

- The resulting audio will be saved as `../audios/mms_tts_korean.wav`.

---

## 📝 Example Code

Below is the full example code used in this repository:

```python
import torch
from transformers import VitsModel, VitsTokenizer
import soundfile as sf
import os

base_dir = os.path.dirname(os.path.abspath(__file__))
audio_dir = os.path.abspath(os.path.join(base_dir, "..", "audios"))
audio_path = os.path.join(audio_dir, "mms_tts_korean.wav")

if not os.path.exists(audio_dir):
    os.makedirs(audio_dir)

# 1. Load model and tokenizer
model = VitsModel.from_pretrained("facebook/mms-tts-kor")
tokenizer = VitsTokenizer.from_pretrained("facebook/mms-tts-kor")

# 2. Tokenize input text
inputs = tokenizer("안녕하세요! 오늘 날씨 좋네요.", return_tensors="pt")
inputs["input_ids"] = inputs["input_ids"].long()

# 3. Generate speech waveform
with torch.no_grad():
    audio = model(**inputs).waveform

# 4. Save audio to file (16kHz)
sf.write(audio_path, audio.cpu().numpy()[0], 16000)
```

---

## 📁 Project Structure

```
leejun-memorial-tts/
│
├─ src/
│   ├─ mms_tts_korean.py
│   └─ requirements.txt
│
├─ audios/
│   └─ (generated .wav files)
│
├─ README.md
└─ .gitignore
```

---

## ⚠️ Notes

- The `audios/` folder and generated `.wav` files are not tracked by git (see `.gitignore`).
- The first run will automatically download the TTS model weights from Hugging Face.
- If you encounter Windows symlink warnings, it will not affect the TTS results.

---

## 📄 License

- **Code and documentation:** MIT License  
- **TTS model:** [facebook/mms-tts-kor](https://huggingface.co/facebook/mms-tts-kor)  
  - License: [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0)

Users must comply with the model and code licenses above, including proper attribution and license notices for any public or commercial use.

---

## 🙏 Acknowledgements

- [Martyr Lee Jun – Wikipedia (KR)](https://ko.wikipedia.org/wiki/%EC%9D%B4%EC%A4%80_(%EC%9A%B0%EC%9D%98%EC%82%AC))
- [facebook/mms-tts-kor on Hugging Face](https://huggingface.co/facebook/mms-tts-kor)
- [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0)

---

## ✉️ Contact & Contributing

Questions or contributions are welcome via GitHub issues or pull requests.
