# 길벗 (Gilbeot)

> 길을 함께 걷는 친구 — 시각장애인을 위한 실시간 보행 도우미

## 개요

길벗은 스마트폰 카메라와 AI를 결합해 시각장애인의 안전한 보행을 돕는 웹 앱입니다.  
YOLOv8 객체 감지, Gemini Vision AI 장면 설명, gTTS 한국어 음성 안내를 실시간으로 제공합니다.

**배경**
- 경기연구원 2022년 조사: 시각장애인 252명 중 52.8%가 버스를 가장 불편한 교통수단으로 꼽음
- 서울시 비신호 횡단보도 25,509개 — 신호등·음향신호기 없이 혼자 건너기 위험

---

## 주요 기능

### 모드 시스템

| 모드 | 전환 방법 |
|------|-----------|
| 대기 | 앱 시작 시 기본값 |
| 횡단보도 모드 | 싱글탭 |
| 보도 보행 모드 | 싱글탭 |
| 장면 설명 | 꾹 누름 (어느 모드에서나) |

### 횡단보도 모드
- 신호등 빨강/초록 감지 및 음성 안내
- 비신호 횡단보도에서 차량 접근 거리 추정 경고

### 보도 보행 모드
- 킥보드, 자전거, 사람 등 장애물 감지
- 차량 돌발 접근 경고

### 장면 설명 (더블탭)
- Gemini Vision AI가 현재 화면을 분석해 2~3문장으로 상황 설명
- 한국어 TTS 음성 출력

---

## 기술 스택

| 구분 | 기술 |
|------|------|
| 백엔드 | Python 3.11, Flask 3.0 |
| 객체 감지 | YOLOv8n (Ultralytics) |
| 장면 설명 | Gemini 2.5 Flash (Vision API) |
| 음성 안내 | gTTS (한국어) |
| 프론트엔드 | HTML/JS (모바일 브라우저 최적화) |

---

## 설치 및 실행

### 1. 의존성 설치

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. 환경변수 설정

[Google AI Studio](https://aistudio.google.com/apikey)에서 API 키를 발급받아 설정합니다.

```bash
export GEMINI_API_KEY="your_api_key_here"
```

### 3. 실행

```bash
python app.py
```

브라우저에서 `http://localhost:5000` 접속 (모바일은 같은 Wi-Fi에서 IP 주소로 접속)

---

## 프로젝트 구조

```
gilbeot/
├── app.py                  # Flask 진입점 및 API 엔드포인트
├── requirements.txt
├── core/
│   ├── detector.py         # YOLOv8 객체 감지 및 거리 추정
│   ├── traffic_light.py    # 신호등 색상 분석 (HSV)
│   ├── scene_explainer.py  # Gemini Vision 장면 설명
│   └── tts_engine.py       # gTTS 음성 생성
├── static/
│   ├── js/main.js          # 웹캠, 터치 제스처, TTS 큐, UI 제어
│   └── css/style.css       # 모바일 최적화 스타일
└── templates/
    └── index.html          # 메인 UI
```

---

## API 엔드포인트

| 메서드 | 경로 | 설명 |
|--------|------|------|
| GET | `/` | 메인 페이지 |
| POST | `/analyze` | 프레임 분석 (실시간 객체 감지) |
| POST | `/explain` | 장면 설명 (Gemini Vision) |
| GET | `/audio/<filename>` | TTS mp3 파일 제공 |

---

## 경고 쿨다운 정책

| 유형 | 쿨다운 |
|------|--------|
| 일반 경고 (장애물 등) | 3초 |
| 차량 접근 경고 | 1초 |
| 신호등 | 상태 변경 시에만 |
