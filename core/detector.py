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

    def __init__(self):
        self.model = YOLO("yolov8s.pt")
        self.last_alert_time: dict[str, float] = {}
        self.last_traffic_state: str = "none"
        self.tl_analyzer = TrafficLightAnalyzer()

    def detect(self, frame, mode: str) -> dict:
        results = self.model.track(frame, persist=True, tracker="bytetrack.yaml", verbose=False)[0]
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

        return {
            "objects": objects,
            "alert": alert,
            "alert_type": alert_type,
            "traffic_light": traffic_light,
        }

    def _color_detect_traffic_light(self, frame) -> str:
        h = frame.shape[0]
        roi = frame[:int(h * 0.4), :]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        total = roi.shape[0] * roi.shape[1]

        red_mask = cv2.bitwise_or(
            cv2.inRange(hsv, np.array([0, 100, 100]), np.array([10, 255, 255])),
            cv2.inRange(hsv, np.array([160, 100, 100]), np.array([180, 255, 255])),
        )
        green_mask = cv2.inRange(hsv, np.array([40, 100, 100]), np.array([80, 255, 255]))

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
                tl = tl_objects[0]
                traffic_light = self.tl_analyzer.analyze(frame, tl["bbox"])
            else:
                traffic_light = self._color_detect_traffic_light(frame)

            if traffic_light in ("red", "green"):
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
                close_vehicles = [v for v in vehicles if v["distance"] in ("very_close", "close")]

                if close_vehicles and self._should_alert("vehicle_approach", 1.0):
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
