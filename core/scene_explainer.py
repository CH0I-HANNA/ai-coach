import os

import cv2
from google import genai
from google.genai import types


class SceneExplainer:
    QUESTION_PROMPT = (
        "당신은 시각장애인의 눈을 대신하는 AI 도우미입니다. "
        "이 사진을 보고 다음 질문에 답해주세요: {question} "
        "2~3문장으로 짧게 한국어로 답변하세요."
    )

    def __init__(self):
        self.client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    def _resize(self, frame):
        h, w = frame.shape[:2]
        if h > 480:
            frame = cv2.resize(frame, (int(w * 480 / h), 480))
        return frame

    def _encode(self, frame) -> bytes:
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return buffer.tobytes()

    def _stream(self, frame, prompt: str):
        image_bytes = self._encode(self._resize(frame))
        for chunk in self.client.models.generate_content_stream(
            model="gemini-2.5-flash",
            contents=[
                types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
                prompt,
            ],
        ):
            if chunk.text:
                yield chunk.text

    def ask_stream(self, frame, question: str):
        return self._stream(frame, self.QUESTION_PROMPT.format(question=question))
