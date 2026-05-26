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
        self._vehicle_tracks: dict[int, tuple[float, float]] = {}  # track_id -> (cx, area)

    def _resize_for_inference(self, frame):
        h, w = frame.shape[:2]
        if w > 640:
            frame = cv2.resize(frame, (640, int(h * 640 / w)))
        return frame

    def detect(self, frame, mode: str) -> dict:
        if mode != getattr(self, '_last_mode', None):
            self._signal_history.clear()
            self.last_traffic_state = "none"
            self._last_mode = mode

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
            track_id = int(box.id[0]) if box.id is not None and len(box.id) > 0 else None
            objects.append({
                "class": self.TARGET_CLASSES[cls_id],
                "class_id": cls_id,
                "confidence": round(conf, 2),
                "bbox": [round(v, 1) for v in bbox],
                "distance": distance,
                "track_id": track_id,
            })

        alert, alert_type, traffic_light = self._build_alert(frame, objects, mode)

        # _build_alert 이후 차량 위치 업데이트 (다음 프레임의 방향 감지용)
        current_ids = set()
        for obj in objects:
            if obj["class_id"] in self.VEHICLE_CLASSES and obj["track_id"] is not None:
                bbox = obj["bbox"]
                cx = (bbox[0] + bbox[2]) / 2
                area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                self._vehicle_tracks[obj["track_id"]] = (cx, area)
                current_ids.add(obj["track_id"])
        self._vehicle_tracks = {k: v for k, v in self._vehicle_tracks.items() if k in current_ids}

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

    def _detect_environment(self, frame) -> str:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        v_mean = float(np.mean(hsv[:, :, 2]))
        s_mean = float(np.mean(hsv[:, :, 1]))
        if v_mean < 60:
            return "night"
        if s_mean < 30 and v_mean > 150:
            return "fog"
        return "normal"

    def _detect_crosswalk(self, frame) -> bool:
        h, w = frame.shape[:2]
        roi = frame[int(h * 0.7):, :]
        if roi.size == 0:
            return False
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        white_mask = cv2.inRange(hsv, np.array([0, 0, 180]), np.array([180, 50, 255]))
        row_sums = np.sum(white_mask, axis=1) / (w * 255)
        threshold = 0.25
        bands, in_white = 0, row_sums[0] > threshold
        for val in row_sums[1:]:
            now_white = val > threshold
            if now_white != in_white:
                bands += 1
                in_white = now_white
        return bands >= 4

    def _get_approach_direction(self, obj, frame_w: int) -> str:
        track_id = obj.get("track_id")
        if track_id is None:
            return "front"
        prev = self._vehicle_tracks.get(track_id)
        if prev is None:
            return "front"
        bbox = obj["bbox"]
        cx = (bbox[0] + bbox[2]) / 2
        area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        prev_cx, prev_area = prev
        if area <= prev_area * 1.1:  # 면적이 10% 이상 커져야 접근으로 판단
            return ""
        dx = cx - prev_cx
        if dx < -frame_w * 0.05:
            return "left"
        elif dx > frame_w * 0.05:
            return "right"
        return "front"

    def _build_alert(self, frame, objects, mode: str):
        traffic_light = "none"
        frame_w = frame.shape[1]
        env = self._detect_environment(frame)

        if mode == "crosswalk":
            tl_objects = [o for o in objects if o["class_id"] == 9]
            yolo_detected = bool(tl_objects)

            if tl_objects:
                tl = max(tl_objects, key=lambda o: (o["bbox"][2] - o["bbox"][0]) * (o["bbox"][3] - o["bbox"][1]))
                traffic_light = self.tl_analyzer.analyze(frame, tl["bbox"])
            else:
                traffic_light = self._color_detect_traffic_light(frame)

            color_detected = traffic_light in ("red", "green")

            # 색상이 확인된 경우에만 히스토리에 추가 (unknown은 쌓지 않음)
            if color_detected:
                self._signal_history.append(traffic_light)
                if len(self._signal_history) > self.SIGNAL_CONFIRM_FRAMES:
                    self._signal_history.pop(0)
            elif not yolo_detected:
                self._signal_history.clear()

            confirmed = (
                len(self._signal_history) == self.SIGNAL_CONFIRM_FRAMES
                and len(set(self._signal_history)) == 1
                and traffic_light in ("red", "green")
            )

            if confirmed:
                if traffic_light != self.last_traffic_state:
                    self.last_traffic_state = traffic_light
                    night_suffix = " 야간이니 주의하세요" if env == "night" else ""
                    if traffic_light == "green":
                        return f"초록불입니다. 건너세요{night_suffix}", "safe", traffic_light
                    elif traffic_light == "red":
                        return f"빨간불입니다. 기다리세요{night_suffix}", "danger", traffic_light

            elif yolo_detected and not color_detected:
                # 신호등은 보이지만 색상 판별 불가 → 사용자에게 알림
                if self._should_alert("tl_detected", 5.0):
                    return "신호등이 보입니다. 색상을 확인하세요", "info", traffic_light

            elif not yolo_detected and not color_detected:
                if self._detect_crosswalk(frame) and self._should_alert("crosswalk", 10.0):
                    return "횡단보도입니다. 신호를 확인하세요", "info", traffic_light

                if self._should_alert("no_tl", 5.0):
                    return "비신호 횡단보도입니다. 차량을 확인합니다", "info", traffic_light

                vehicles = [o for o in objects if o["class_id"] in self.VEHICLE_CLASSES]
                frame_h = frame.shape[0]
                approaching = [
                    v for v in vehicles
                    if v["distance"] in ("very_close", "close")
                    and (v["bbox"][1] + v["bbox"][3]) / 2 > frame_h * 0.33
                ]

                if approaching and self._should_alert("vehicle_approach", 1.0):
                    direction = self._get_approach_direction(approaching[0], frame_w)
                    if direction == "left":
                        return "왼쪽에서 차량이 접근합니다. 기다리세요", "danger", traffic_light
                    elif direction == "right":
                        return "오른쪽에서 차량이 접근합니다. 기다리세요", "danger", traffic_light
                    return "차량이 접근하고 있습니다. 기다리세요", "danger", traffic_light

                if not vehicles and self._should_alert("no_vehicle", 5.0):
                    return "차량이 없습니다. 건너셔도 됩니다", "safe", traffic_light

        elif mode == "sidewalk":
            vehicles = [o for o in objects if o["class_id"] in self.VEHICLE_CLASSES]
            very_close_vehicles = [v for v in vehicles if v["distance"] == "very_close"]
            if very_close_vehicles and self._should_alert("vehicle_sudden", 1.0):
                direction = self._get_approach_direction(very_close_vehicles[0], frame_w)
                if direction == "left":
                    return "왼쪽에서 차량이 접근합니다. 멈추세요", "danger", traffic_light
                elif direction == "right":
                    return "오른쪽에서 차량이 접근합니다. 멈추세요", "danger", traffic_light
                return "차량이 접근합니다. 멈추세요", "danger", traffic_light

            kickboards = [o for o in objects if o["class_id"] == 3]
            if kickboards and self._should_alert("kickboard", 3.0):
                return "앞에 킥보드가 있습니다. 주의하세요", "info", traffic_light

            obstacles = [o for o in objects
                         if o["class_id"] in (0, 1)
                         and o["distance"] in ("very_close", "close")]
            if obstacles and self._should_alert("obstacle", 3.0):
                return "앞에 장애물이 있습니다", "info", traffic_light

        # 환경 경고 — 다른 경고 없을 때만 출력
        if env == "night" and self._should_alert("env_night", 30.0):
            return "야간입니다. 주의하며 이동하세요", "info", traffic_light
        if env == "fog" and self._should_alert("env_fog", 30.0):
            return "시야가 흐립니다. 주의하세요", "info", traffic_light

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
