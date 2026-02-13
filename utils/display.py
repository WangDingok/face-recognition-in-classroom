from collections import deque
from typing import Dict, Optional
import cv2

class FPSDisplay:
    def __init__(
        self,
        buffer_size: int = 10,
        metrics: Optional[Dict[str, str]] = None,
        pos_x_ratio: float = 0.01,     # 1% from left
        pos_y_ratio: float = 0.15,     # 15% from top
        box_width_ratio: float = 0.2,  # 20% of frame width
        line_spacing_ratio: float = 0.035,  # ~3.5% of frame height
        padding_ratio: float = 0.015,       # ~1.5% of frame height
        background_alpha: float = 0.4
    ):
        """
        Args:
            buffer_size (int): Number of frames used for smoothing FPS values.
            metrics (Dict[str, str], optional): Internal key to display label mapping.
            pos_x_ratio (float): X position as ratio of frame width.
            pos_y_ratio (float): Y position as ratio of frame height.
            box_width_ratio (float): Width of the FPS box relative to frame width.
            line_spacing_ratio (float): Line spacing as ratio of frame height.
            padding_ratio (float): Padding as ratio of frame height.
            background_alpha (float): Transparency of background.
        """
        if metrics is None:
            metrics = {
                "detect": "Detection + Tracking",
                "embed": "Embedding",
                "query": "Query",
                "total": "Total Processing"
            }

        self.metrics = metrics
        self.buffer_size = buffer_size
        self.fps_history = {k: deque(maxlen=buffer_size) for k in metrics}

        # Ratio-based layout parameters
        self.pos_x_ratio = pos_x_ratio
        self.pos_y_ratio = pos_y_ratio
        self.box_width_ratio = box_width_ratio
        self.line_spacing_ratio = line_spacing_ratio
        self.padding_ratio = padding_ratio
        self.background_alpha = background_alpha

        self.fps_video = 0  # Set externally later

    def update(self, **kwargs: float) -> None:
        for key, value in kwargs.items():
            if key in self.fps_history:
                self.fps_history[key].append(value)

    def draw(self, frame):
        h, w = frame.shape[:2]

        x = int(self.pos_x_ratio * w)
        y = int(self.pos_y_ratio * h)
        box_width = int(self.box_width_ratio * w)
        line_spacing = int(self.line_spacing_ratio * h)
        padding = int(self.padding_ratio * h)

        total_lines = len(self.metrics) + 1  # +1 for original video FPS
        box_height = total_lines * line_spacing + 2 * padding

        # Draw semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (x, y), (x + box_width, y + box_height), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, self.background_alpha, frame, 1 - self.background_alpha, 0)

        # Draw FPS for original video
        text_video = f"FPS (Original Video): {self.fps_video}"
        start_y = y + padding + int(line_spacing * 0.8)
        cv2.putText(frame, text_video, (x + padding, start_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Draw other metrics
        for i, (key, label) in enumerate(self.metrics.items()):
            values = self.fps_history[key]
            avg_fps = sum(values) / len(values) if values else 0.0
            text = f"FPS ({label}): {avg_fps:.2f}"
            cv2.putText(frame, text, (x + padding, start_y + (i + 1) * line_spacing),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        return frame
