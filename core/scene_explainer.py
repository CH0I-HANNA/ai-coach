import base64
import os

import cv2
from google import genai
from google.genai import types


class SceneExplainer:
    SCENE_PROMPT = (
        "당신은 시각장애인의 눈을 대신하는 AI 도우미입니다. "
        "지금 눈앞의 상황을 설명해주세요. "
        "위험 요소가 있으면 먼저 말하고, 2~3문장으로 짧게 한국어로 설명하세요."
    )
    QUESTION_PROMPT = (
        "당신은 시각장애인의 눈을 대신하는 AI 도우미입니다. "
        "이 사진을 보고 다음 질문에 답해주세요: {question} "
        "2~3문장으로 짧게 한국어로 답변하세요."
    )

    def __init__(self):
        self.client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    def _generate(self, frame, prompt: str) -> str:
        _, buffer = cv2.imencode('.jpg', frame)
        image_bytes = buffer.tobytes()
        response = self.client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[
                types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
                prompt,
            ],
        )
        return response.text

    def explain(self, frame) -> str:
        return self._generate(frame, self.SCENE_PROMPT)

    def ask(self, frame, question: str) -> str:
        return self._generate(frame, self.QUESTION_PROMPT.format(question=question))
