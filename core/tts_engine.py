import asyncio
import hashlib
import os
import time

import edge_tts


class TTSEngine:
    VOICE = "ko-KR-SunHiNeural"

    def __init__(self, output_dir="static/audio"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self._cache: dict[str, str] = {}

    def generate(self, text: str) -> str:
        if text in self._cache:
            filepath = os.path.join(self.output_dir, self._cache[text])
            if os.path.exists(filepath):
                return f"/audio/{self._cache[text]}"

        filename = hashlib.md5(text.encode()).hexdigest() + ".mp3"
        filepath = os.path.join(self.output_dir, filename)

        if not os.path.exists(filepath):
            loop = asyncio.new_event_loop()
            loop.run_until_complete(edge_tts.Communicate(text, self.VOICE).save(filepath))
            loop.close()

        self._cache[text] = filename
        return f"/audio/{filename}"

    def cleanup_old_files(self, max_age_seconds=300):
        cached = set(self._cache.values())
        now = time.time()
        for fname in os.listdir(self.output_dir):
            if fname in cached:
                continue
            fpath = os.path.join(self.output_dir, fname)
            if os.path.isfile(fpath) and (now - os.path.getmtime(fpath)) > max_age_seconds:
                try:
                    os.remove(fpath)
                except OSError:
                    pass
