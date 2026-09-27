
"""Face landmarks detector module using MediaPipe Tasks API.

Uses the pre-trained MediaPipe Face Landmarker model to extract
478 3D facial landmarks and facial transformation matrices.
"""

from pathlib import Path
from typing import List, Optional, Any, Dict

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


class FaceLandmarksDetector:
    """Detect facial landmarks and optional head-pose information."""

    def __init__(self, model_path: Optional[str] = None) -> None:
        """Initialize MediaPipe Face Landmarker."""

        if model_path is None:
            base_dir = Path(__file__).resolve().parent.parent.parent
            model_path = str(base_dir / "models" / "face_landmarker.task")

        self.model_path = Path(model_path)

        if not self.model_path.is_file():
            raise FileNotFoundError(
                f"\n[MediaPipe Error] Model file not found at: {self.model_path}\n"
                "MediaPipe Tasks API requires the pre-trained "
                "'face_landmarker.task' bundle.\n"
                "Please download it from Google's official storage:\n"
                "https://storage.googleapis.com/mediapipe-models/"
                "face_landmarker/face_landmarker/float16/1/"
                "face_landmarker.task\n"
                f"and place it at: {self.model_path}\n"
            )

        self._video_timestamp_ms = 0

        # Latest valid facial transformation matrix.
        # This does NOT change the existing process_frame() return value.
        self._last_transformation_matrix: Optional[np.ndarray] = None

        try:
            base_options = python.BaseOptions(
                model_asset_path=str(self.model_path)
            )

            options = vision.FaceLandmarkerOptions(
                base_options=base_options,
                running_mode=vision.RunningMode.VIDEO,
                num_faces=1,
                min_face_detection_confidence=0.5,
                min_face_presence_confidence=0.5,
                min_tracking_confidence=0.5,
                output_face_blendshapes=False,
                output_facial_transformation_matrixes=True,
            )

            self.landmarker = vision.FaceLandmarker.create_from_options(
                options
            )

        except Exception as e:
            raise RuntimeError(
                f"Failed to initialize MediaPipe Face Landmarker: {e}"
            ) from e

    def process_frame(
        self,
        frame: np.ndarray
    ) -> Optional[List[Any]]:
        """Process an OpenCV BGR frame.

        The return value remains compatible with the previous project:
        a list of facial landmarks or None.

        The latest facial transformation matrix is stored internally
        and can be retrieved with get_last_transformation_matrix().
        """

        # Clear the previous frame's transformation matrix.
        self._last_transformation_matrix = None

        if frame is None or frame.size == 0:
            return None

        if frame.ndim != 3 or frame.shape[2] != 3:
            return None

        # Convert BGR -> RGB for MediaPipe.
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if rgb_frame.dtype != np.uint8:
            rgb_frame = rgb_frame.astype(np.uint8)

        rgb_frame = np.ascontiguousarray(rgb_frame)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame,
        )

        detection_result = self.landmarker.detect_for_video(
            mp_image,
            self._video_timestamp_ms,
        )

        # Keep the existing timestamp behavior for now.
        # Timestamp/FPS will be audited separately.
        self._video_timestamp_ms += 1

        if not detection_result.face_landmarks:
            return None

        # Store the transformation matrix if MediaPipe returned one.
        matrices = detection_result.facial_transformation_matrixes

        if matrices is not None and len(matrices) > 0:
            matrix = np.asarray(
                matrices[0],
                dtype=np.float64,
            )

            if matrix.size == 16:
                matrix = matrix.reshape(4, 4)

            if (
                matrix.shape == (4, 4)
                and np.all(np.isfinite(matrix))
            ):
                self._last_transformation_matrix = matrix

        # IMPORTANT:
        # Keep returning landmarks exactly as before.
        return detection_result.face_landmarks[0]

    def get_last_transformation_matrix(
        self,
    ) -> Optional[np.ndarray]:
        """Return the latest valid 4x4 facial transformation matrix.

        Returns:
            A copy of the latest valid matrix, or None.
        """

        if self._last_transformation_matrix is None:
            return None

        return self._last_transformation_matrix.copy()

    def get_head_pose(self) -> Dict[str, Any]:
        """Return diagnostic head-pose rotation information.

        Returns:
            Dictionary containing:
                valid
                yaw
                pitch
                roll
                rotation_matrix
        """

        matrix = self._last_transformation_matrix

        if matrix is None or matrix.shape != (4, 4):
            return {
                "valid": False,
                "yaw": None,
                "pitch": None,
                "roll": None,
                "rotation_matrix": None,
            }

        rotation = matrix[:3, :3]

        if not np.all(np.isfinite(rotation)):
            return {
                "valid": False,
                "yaw": None,
                "pitch": None,
                "roll": None,
                "rotation_matrix": None,
            }

        try:
            # Normalize the rotation matrix.
            u, _, vh = np.linalg.svd(rotation)
            rotation = u @ vh

            # Ensure a proper rotation matrix.
            if np.linalg.det(rotation) < 0:
                u[:, -1] *= -1
                rotation = u @ vh

            sy = float(
                np.sqrt(
                    rotation[0, 0] ** 2
                    + rotation[1, 0] ** 2
                )
            )

            singular = sy < 1e-6

            if not singular:
                pitch = np.arctan2(
                    -rotation[2, 0],
                    sy,
                )

                yaw = np.arctan2(
                    rotation[1, 0],
                    rotation[0, 0],
                )

                roll = np.arctan2(
                    rotation[2, 1],
                    rotation[2, 2],
                )

            else:
                pitch = np.arctan2(
                    -rotation[2, 0],
                    sy,
                )

                yaw = np.arctan2(
                    -rotation[0, 1],
                    rotation[1, 1],
                )

                roll = 0.0

            return {
                "valid": True,
                "yaw": float(np.degrees(yaw)),
                "pitch": float(np.degrees(pitch)),
                "roll": float(np.degrees(roll)),
                "rotation_matrix": rotation.copy(),
            }

        except (
            np.linalg.LinAlgError,
            ValueError,
            FloatingPointError,
        ):
            return {
                "valid": False,
                "yaw": None,
                "pitch": None,
                "roll": None,
                "rotation_matrix": None,
            }

    def draw_face_mesh(
        self,
        frame: np.ndarray,
        landmarks: List[Any],
        color: tuple = (0, 200, 0),
        radius: int = 1,
    ) -> np.ndarray:
        """Draw face landmarks onto the frame."""

        h, w = frame.shape[:2]

        for lm in landmarks:
            px = int(lm.x * w)
            py = int(lm.y * h)

            if 0 <= px < w and 0 <= py < h:
                cv2.circle(
                    frame,
                    (px, py),
                    radius,
                    color,
                    -1,
                )

        return frame

    def close(self) -> None:
        """Release MediaPipe landmarker resources."""

        if (
            hasattr(self, "landmarker")
            and self.landmarker is not None
        ):
            self.landmarker.close()
            self.landmarker = None

    def __enter__(self) -> "FaceLandmarksDetector":
        return self

    def __exit__(
        self,
        exc_type,
        exc_val,
        exc_tb,
    ) -> None:
        self.close()

