"""Eye and iris feature extraction module for EyeControl AI (Phase 2).

Extracts geometric features from facial landmarks detected by MediaPipe:
- Left and right eye contours and landmarks
- Iris landmarks and iris centers
- Normalized iris position within eye boundaries
- Eye Aspect Ratio (EAR)
- Eye width and height
- Basic head reference features (inter-eye distance and face center)
"""

from dataclasses import dataclass
import math
from typing import List, Optional, Tuple, Any
import cv2
import numpy as np


# ---------------------------------------------------------------------------
# MediaPipe Face Landmarker Indices (478-landmark topology)
# ---------------------------------------------------------------------------

# Iris landmarks (5 landmarks per iris: center + 4 perimeter points)
# Note: MediaPipe anatomical naming:
# Left iris is the subject's left eye (appears on the right side of camera image).
# Right iris is the subject's right eye (appears on the left side of camera image).
LEFT_IRIS_INDICES = [473, 474, 475, 476, 477]
RIGHT_IRIS_INDICES = [468, 469, 470, 471, 472]

# Full eye boundary contour indices (from FaceLandmarksConnections)
LEFT_EYE_CONTOUR = [
    249, 263, 362, 373, 374, 380, 381, 382,
    384, 385, 386, 387, 388, 390, 398, 466
]
RIGHT_EYE_CONTOUR = [
    7, 33, 133, 144, 145, 153, 154, 155,
    157, 158, 159, 160, 161, 163, 173, 246
]

# Key landmarks for EAR calculation and bounding geometry:
# Left eye (subject's left):
LEFT_EYE_OUTER_CORNER = 263      # Lateral canthus
LEFT_EYE_INNER_CORNER = 362      # Medial canthus
LEFT_EYE_TOP_1 = 385            # Upper eyelid point 1
LEFT_EYE_BOTTOM_1 = 380         # Lower eyelid point 1
LEFT_EYE_TOP_2 = 387            # Upper eyelid point 2
LEFT_EYE_BOTTOM_2 = 373         # Lower eyelid point 2

# Right eye (subject's right):
RIGHT_EYE_OUTER_CORNER = 33      # Lateral canthus
RIGHT_EYE_INNER_CORNER = 133     # Medial canthus
RIGHT_EYE_TOP_1 = 160           # Upper eyelid point 1
RIGHT_EYE_BOTTOM_1 = 144        # Lower eyelid point 1
RIGHT_EYE_TOP_2 = 158           # Upper eyelid point 2
RIGHT_EYE_BOTTOM_2 = 153        # Lower eyelid point 2

# Nose tip landmark (useful for face center approximation)
NOSE_TIP = 1


@dataclass
class EyeFeatures:
    """Dataclass holding extracted eye and iris geometric features."""

    # Raw iris center in normalized image coordinates [0.0, 1.0]
    left_iris_x: float
    left_iris_y: float
    right_iris_x: float
    right_iris_y: float

    # Normalized iris position inside eye socket [0.0, 1.0]
    # (0.0 -> one side, 0.5 -> center, 1.0 -> other side)
    left_normalized_x: float
    left_normalized_y: float
    right_normalized_x: float
    right_normalized_y: float

    # Eye Aspect Ratio (EAR) measuring eye openness
    left_ear: float
    right_ear: float
    avg_ear: float

    # Eye dimensions (normalized image coordinates)
    eye_width_left: float
    eye_height_left: float
    eye_width_right: float
    eye_height_right: float

    # Basic head reference information
    face_center_x: float
    face_center_y: float
    inter_eye_distance: float


class EyeFeatureExtractor:
    """Extracts eye and iris features from MediaPipe facial landmarks."""

    @staticmethod
    def _euclidean_distance(p1: Any, p2: Any) -> float:
        """Calculate 2D Euclidean distance between two normalized landmarks.

        Formula: sqrt((x2 - x1)^2 + (y2 - y1)^2)
        """
        return math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)

    @staticmethod
    def _calculate_iris_center(landmarks: List[Any], indices: List[int]) -> Tuple[float, float]:
        """Calculate the center of the iris by averaging its landmark coordinates.

        Formula:
            center_x = sum(x_i) / N
            center_y = sum(y_i) / N
        """
        avg_x = sum(landmarks[i].x for i in indices) / len(indices)
        avg_y = sum(landmarks[i].y for i in indices) / len(indices)
        return avg_x, avg_y

    @staticmethod
    def _calculate_ear(
        landmarks: List[Any],
        outer_corner: int,
        inner_corner: int,
        top_1: int,
        bottom_1: int,
        top_2: int,
        bottom_2: int,
    ) -> float:
        """Calculate Eye Aspect Ratio (EAR) based on Soukupová & Čech formula.

        EAR = (||top_1 - bottom_1|| + ||top_2 - bottom_2||) / (2 * ||outer - inner||)
        """
        dist_v1 = EyeFeatureExtractor._euclidean_distance(landmarks[top_1], landmarks[bottom_1])
        dist_v2 = EyeFeatureExtractor._euclidean_distance(landmarks[top_2], landmarks[bottom_2])
        dist_h = EyeFeatureExtractor._euclidean_distance(landmarks[outer_corner], landmarks[inner_corner])

        if dist_h < 1e-6:
            return 0.0

        return (dist_v1 + dist_v2) / (2.0 * dist_h)

    @staticmethod
    def _calculate_normalized_position(
        iris_x: float,
        iris_y: float,
        contour_landmarks: List[Any],
    ) -> Tuple[float, float, float, float]:
        """Calculate relative iris position inside the eye bounding box.

        Formula:
            normalized_x = (iris_x - min_x) / (max_x - min_x)
            normalized_y = (iris_y - min_y) / (max_y - min_y)

        Returns:
            Tuple of (norm_x, norm_y, eye_width, eye_height).
        """
        xs = [lm.x for lm in contour_landmarks]
        ys = [lm.y for lm in contour_landmarks]

        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)

        eye_width = max(max_x - min_x, 1e-6)
        eye_height = max(max_y - min_y, 1e-6)

        norm_x = (iris_x - min_x) / eye_width
        norm_y = (iris_y - min_y) / eye_height

        # Clamp to [0.0, 1.0] to guard against minor landmark jitter
        norm_x = max(0.0, min(1.0, norm_x))
        norm_y = max(0.0, min(1.0, norm_y))

        return norm_x, norm_y, eye_width, eye_height

    def extract_features(self, landmarks: List[Any]) -> Optional[EyeFeatures]:
        """Extract all eye, iris, and reference features from 478 face landmarks.

        Args:
            landmarks: List of 478 NormalizedLandmarks from MediaPipe.

        Returns:
            EyeFeatures instance, or None if landmarks are insufficient.
        """
        if landmarks is None or len(landmarks) < 478:
            return None

        # 1. Calculate Iris Centers (average of iris points)
        left_iris_x, left_iris_y = self._calculate_iris_center(landmarks, LEFT_IRIS_INDICES)
        right_iris_x, right_iris_y = self._calculate_iris_center(landmarks, RIGHT_IRIS_INDICES)

        # 2. Extract eye contour landmarks for bounding calculation
        left_eye_pts = [landmarks[i] for i in LEFT_EYE_CONTOUR]
        right_eye_pts = [landmarks[i] for i in RIGHT_EYE_CONTOUR]

        # 3. Calculate Normalized Iris Positions and eye dimensions
        left_norm_x, left_norm_y, left_w, left_h = self._calculate_normalized_position(
            left_iris_x, left_iris_y, left_eye_pts
        )
        right_norm_x, right_norm_y, right_w, right_h = self._calculate_normalized_position(
            right_iris_x, right_iris_y, right_eye_pts
        )

        # 4. Calculate Eye Aspect Ratio (EAR) for openness
        left_ear = self._calculate_ear(
            landmarks,
            LEFT_EYE_OUTER_CORNER,
            LEFT_EYE_INNER_CORNER,
            LEFT_EYE_TOP_1,
            LEFT_EYE_BOTTOM_1,
            LEFT_EYE_TOP_2,
            LEFT_EYE_BOTTOM_2,
        )
        right_ear = self._calculate_ear(
            landmarks,
            RIGHT_EYE_OUTER_CORNER,
            RIGHT_EYE_INNER_CORNER,
            RIGHT_EYE_TOP_1,
            RIGHT_EYE_BOTTOM_1,
            RIGHT_EYE_TOP_2,
            RIGHT_EYE_BOTTOM_2,
        )
        avg_ear = (left_ear + right_ear) / 2.0

        # 5. Basic Head Reference Info
        face_center_x = landmarks[NOSE_TIP].x
        face_center_y = landmarks[NOSE_TIP].y
        inter_eye_distance = math.sqrt(
            (left_iris_x - right_iris_x) ** 2 + (left_iris_y - right_iris_y) ** 2
        )

        return EyeFeatures(
            left_iris_x=left_iris_x,
            left_iris_y=left_iris_y,
            right_iris_x=right_iris_x,
            right_iris_y=right_iris_y,
            left_normalized_x=left_norm_x,
            left_normalized_y=left_norm_y,
            right_normalized_x=right_norm_x,
            right_normalized_y=right_norm_y,
            left_ear=left_ear,
            right_ear=right_ear,
            avg_ear=avg_ear,
            eye_width_left=left_w,
            eye_height_left=left_h,
            eye_width_right=right_w,
            eye_height_right=right_h,
            face_center_x=face_center_x,
            face_center_y=face_center_y,
            inter_eye_distance=inter_eye_distance,
        )

    def draw_eye_overlay(
        self,
        frame: np.ndarray,
        landmarks: List[Any],
        features: EyeFeatures,
    ) -> np.ndarray:
        """Draw visual markers for eye boundaries, iris points, and iris centers.

        Args:
            frame: OpenCV BGR image to draw on.
            landmarks: 478 MediaPipe landmarks.
            features: Extracted EyeFeatures dataclass.

        Returns:
            Annotated OpenCV frame.
        """
        h, w = frame.shape[:2]

        # Draw Eye Contours (cyan dots)
        for idx in LEFT_EYE_CONTOUR + RIGHT_EYE_CONTOUR:
            lm = landmarks[idx]
            pt = (int(lm.x * w), int(lm.y * h))
            cv2.circle(frame, pt, 1, (255, 255, 0), -1)

        # Draw Iris Landmarks (green dots)
        for idx in LEFT_IRIS_INDICES + RIGHT_IRIS_INDICES:
            lm = landmarks[idx]
            pt = (int(lm.x * w), int(lm.y * h))
            cv2.circle(frame, pt, 2, (0, 255, 0), -1)

        # Draw Left Iris Center (distinct magenta/red marker with crosshair)
        lx = int(features.left_iris_x * w)
        ly = int(features.left_iris_y * h)
        cv2.circle(frame, (lx, ly), 3, (0, 0, 255), -1)
        cv2.line(frame, (lx - 5, ly), (lx + 5, ly), (0, 0, 255), 1)
        cv2.line(frame, (lx, ly - 5), (lx, ly + 5), (0, 0, 255), 1)

        # Draw Right Iris Center (distinct magenta/red marker with crosshair)
        rx = int(features.right_iris_x * w)
        ry = int(features.right_iris_y * h)
        cv2.circle(frame, (rx, ry), 3, (0, 0, 255), -1)
        cv2.line(frame, (rx - 5, ry), (rx + 5, ry), (0, 0, 255), 1)
        cv2.line(frame, (rx, ry - 5), (rx, ry + 5), (0, 0, 255), 1)

        return frame
