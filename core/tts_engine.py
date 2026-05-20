import os
import time
import uuid
from gtts import gTTS


class TTSEngine:
    def __init__(self, output_dir="static/audio"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def generate(self, text: str) -> str:
        filename = f"{uuid.uuid4()}.mp3"
        filepath = os.path.join(self.output_dir, filename)
        tts = gTTS(text=text, lang='ko')
        tts.save(filepath)
        return f"/audio/{filename}"

    def cleanup_old_files(self, max_age_seconds=60):
        now = time.time()
        for fname in os.listdir(self.output_dir):
            fpath = os.path.join(self.output_dir, fname)
            if os.path.isfile(fpath) and (now - os.path.getmtime(fpath)) > max_age_seconds:
                try:
                    os.remove(fpath)
                except OSError:
                    pass
