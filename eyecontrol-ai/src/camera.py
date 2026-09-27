"""Camera capture module using OpenCV.

Provides a clean and reusable Camera class to open, read from,
and release the system webcam safely.
"""

from typing import Optional, Tuple
import cv2
import numpy as np


class Camera:
    """Manages webcam capture lifecycle using OpenCV."""

    def __init__(self, device_index: int = 0) -> None:
        """Initialize the camera with the specified device index.

        Args:
            device_index: Integer index of the camera (default 0 for primary webcam).
        """
        self.device_index = device_index
        self.cap: Optional[cv2.VideoCapture] = None
        self.failed_reads = 0
        self._open()

    def _open(self) -> None:
        """Attempt to open the video capture device using the Windows DSHOW backend."""
        self.cap = cv2.VideoCapture(self.device_index, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(self.device_index)

    def is_opened(self) -> bool:
        """Check if the camera is currently opened and available.

        Returns:
            True if camera is opened, False otherwise.
        """
        return self.cap is not None and self.cap.isOpened()

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read a single frame from the camera.

        Returns:
            Tuple of (success_flag, frame). Frame is None if reading fails.
        """
        if not self.is_opened():
            return False, None

        ret, frame = self.cap.read()
        if not ret or frame is None:
            self.failed_reads += 1
            return False, None

        self.failed_reads = max(0, self.failed_reads - 1)
        return True, frame

    def release(self) -> None:
        """Release the camera capture device safely."""
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()
        self.cap = None

    def __enter__(self) -> "Camera":
        """Support for Python context manager."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Ensure camera resources are released on context exit."""
        self.release()
