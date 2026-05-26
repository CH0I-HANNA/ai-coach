import time
import cv2
import numpy as np
from ultralytics import YOLO
from .traffic_light import TrafficLightAnalyzer


class BowDetector:
    TARGET_CLASSES = {
        0: "person",
        1: "bicycle",
        2: "car",
        3: "motorcycle",
        5: "bus",
        7: "truck",
        9: "traffic light",
    }

    VEHICLE_CLASSES = {2, 3, 5, 7}

    SIGNAL_CONFIRM_FRAMES = 3

    def __init__(self):
        self.model = YOLO("yolov8s.pt")
        self.last_alert_time: dict[str, float] = {}
        self.last_traffic_state: str = "none"
        self.tl_analyzer = TrafficLightAnalyzer()
        self._frame_idx = 0
        self._last_result: dict | None = None
        self._signal_history: list[str] = []

    def _resize_for_inference(self, frame):
        h, w = frame.shape[:2]
        if w > 640:
            frame = cv2.resize(frame, (640, int(h * 640 / w)))
        return frame

    def detect(self, frame, mode: str) -> dict:
        self._frame_idx += 1
        if self._frame_idx % 2 == 0 and self._last_result is not None:
            return self._last_result

        results = self.model.track(self._resize_for_inference(frame), persist=True, tracker="bytetrack.yaml", verbose=False)[0]
        objects = []

        for box in results.boxes:
            cls_id = int(box.cls[0])
            if cls_id not in self.TARGET_CLASSES:
                continue
            conf = float(box.conf[0])
            if conf < 0.4:
                continue
            bbox = box.xyxy[0].tolist()
            distance = self._estimate_distance(bbox, frame.shape)
            objects.append({
                "class": self.TARGET_CLASSES[cls_id],
                "class_id": cls_id,
                "confidence": round(conf, 2),
                "bbox": [round(v, 1) for v in bbox],
                "distance": distance,
            })

        alert, alert_type, traffic_light = self._build_alert(frame, objects, mode)

        self._last_result = {
            "objects": objects,
            "alert": alert,
            "alert_type": alert_type,
            "traffic_light": traffic_light,
        }
        return self._last_result

    def _color_detect_traffic_light(self, frame) -> str:
        h, w = frame.shape[:2]
        # 화면 중앙 40% 폭, 상단 35% 높이만 검사 — 간판·차량 등 측면 노이즈 제거
        x1, x2 = int(w * 0.30), int(w * 0.70)
        y2 = int(h * 0.35)
        roi = frame[:y2, x1:x2]
        if roi.size == 0:
            return "none"

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        total = roi.shape[0] * roi.shape[1]

        red_mask = cv2.bitwise_or(
            cv2.inRange(hsv, np.array([0, 120, 120]), np.array([10, 255, 255])),
            cv2.inRange(hsv, np.array([160, 120, 120]), np.array([180, 255, 255])),
        )
        green_mask = cv2.inRange(hsv, np.array([45, 120, 120]), np.array([75, 255, 255]))

        if cv2.countNonZero(red_mask) / total > 0.05:
            return "red"
        if cv2.countNonZero(green_mask) / total > 0.05:
            return "green"
        return "none"

    def _build_alert(self, frame, objects, mode: str):
        traffic_light = "none"

        if mode == "crosswalk":
            tl_objects = [o for o in objects if o["class_id"] == 9]

            if tl_objects:
                tl = max(tl_objects, key=lambda o: (o["bbox"][2] - o["bbox"][0]) * (o["bbox"][3] - o["bbox"][1]))
                traffic_light = self.tl_analyzer.analyze(frame, tl["bbox"])
            else:
                traffic_light = self._color_detect_traffic_light(frame)

            self._signal_history.append(traffic_light)
            if len(self._signal_history) > self.SIGNAL_CONFIRM_FRAMES:
                self._signal_history.pop(0)

            confirmed = (
                len(self._signal_history) == self.SIGNAL_CONFIRM_FRAMES
                and len(set(self._signal_history)) == 1
                and traffic_light in ("red", "green")
            )

            if confirmed:
                if traffic_light != self.last_traffic_state:
                    self.last_traffic_state = traffic_light
                    if traffic_light == "green":
                        return "초록불입니다. 건너세요", "safe", traffic_light
                    elif traffic_light == "red":
                        return "빨간불입니다. 기다리세요", "danger", traffic_light
            else:
                # 신호등 미감지 → 차량 확인
                if self._should_alert("no_tl", 5.0):
                    return "비신호 횡단보도입니다. 차량을 확인합니다", "info", traffic_light

                vehicles = [o for o in objects if o["class_id"] in self.VEHICLE_CLASSES]
                # 횡단보도 위험 차량 = 화면 하단 2/3에 위치한 차량 (측면에서 접근 중)
                frame_h = frame.shape[0]
                approaching = [
                    v for v in vehicles
                    if v["distance"] in ("very_close", "close")
                    and (v["bbox"][1] + v["bbox"][3]) / 2 > frame_h * 0.33
                ]

                if approaching and self._should_alert("vehicle_approach", 1.0):
                    return "차량이 접근하고 있습니다. 기다리세요", "danger", traffic_light

                if not vehicles and self._should_alert("no_vehicle", 5.0):
                    return "차량이 없습니다. 건너셔도 됩니다", "safe", traffic_light

        elif mode == "sidewalk":
            vehicles = [o for o in objects if o["class_id"] in self.VEHICLE_CLASSES]
            very_close_vehicles = [v for v in vehicles if v["distance"] == "very_close"]
            if very_close_vehicles and self._should_alert("vehicle_sudden", 1.0):
                return "차량이 접근합니다. 멈추세요", "danger", traffic_light

            kickboards = [o for o in objects if o["class_id"] == 3]
            if kickboards and self._should_alert("kickboard", 3.0):
                return "앞에 킥보드가 있습니다. 주의하세요", "info", traffic_light

            obstacles = [o for o in objects
                         if o["class_id"] in (0, 1)
                         and o["distance"] in ("very_close", "close")]
            if obstacles and self._should_alert("obstacle", 3.0):
                return "앞에 장애물이 있습니다", "info", traffic_light

        return "", "info", traffic_light

    def _estimate_distance(self, bbox, frame_shape) -> str:
        frame_area = frame_shape[0] * frame_shape[1]
        bbox_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        ratio = bbox_area / frame_area if frame_area > 0 else 0

        if ratio >= 0.10:
            return "very_close"
        elif ratio >= 0.05:
            return "close"
        return "far"

    def _should_alert(self, alert_key: str, cooldown: float) -> bool:
        now = time.time()
        last = self.last_alert_time.get(alert_key, 0)
        if now - last >= cooldown:
            self.last_alert_time[alert_key] = now
            return True
        return False
