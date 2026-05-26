import cv2
import numpy as np


class TrafficLightAnalyzer:
    def analyze(self, frame, traffic_light_bbox) -> str:
        top_region, bottom_region = self._crop_signal_region(frame, traffic_light_bbox)

        red_count = self._count_color(top_region, "red")
        green_count = self._count_color(bottom_region, "green")

        top_area = top_region.shape[0] * top_region.shape[1] if top_region.size > 0 else 1
        bot_area = bottom_region.shape[0] * bottom_region.shape[1] if bottom_region.size > 0 else 1

        red_ratio = red_count / top_area
        green_ratio = green_count / bot_area

        if red_ratio > 0.03 and red_ratio > green_ratio:
            return "red"
        elif green_ratio > 0.03 and green_ratio > red_ratio:
            return "green"
        return "unknown"

    def _crop_signal_region(self, frame, bbox):
        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)

        height = y2 - y1
        top_region = frame[y1:y1 + int(height * 0.4), x1:x2]
        bottom_region = frame[y2 - int(height * 0.4):y2, x1:x2]
        return top_region, bottom_region

    def _count_color(self, region, color: str) -> int:
        if region.size == 0:
            return 0

        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)

        if color == "red":
            mask1 = cv2.inRange(hsv,
                np.array([0, 100, 100]), np.array([10, 255, 255]))
            mask2 = cv2.inRange(hsv,
                np.array([160, 100, 100]), np.array([180, 255, 255]))
            mask = cv2.bitwise_or(mask1, mask2)
        elif color == "green":
            mask = cv2.inRange(hsv,
                np.array([40, 100, 100]), np.array([80, 255, 255]))
        else:
            return 0

        return int(np.sum(mask > 0))
