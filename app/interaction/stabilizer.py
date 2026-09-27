"""Direction stabilization module for EyeControl AI (Issue #1).

Filters raw discrete gaze predictions from the classifier and provides a
stable, noise-resilient directional state for the mouse controller.
"""

from collections import deque
from typing import Deque, FrozenSet, List, Optional


class DirectionStabilizer:
    """Stabilizes discrete gaze direction predictions using state confirmation.

    This component filters out momentary classification noise, rapid jitter,
    and eye-blink artifacts without introducing global state or external
    hardware dependencies.
    """

    NEUTRAL_STATES: FrozenSet[str] = frozenset(
        {
            "FORWARD",
            "EYES_OPEN",
            "EYES_CLOSED",
            "NO_FACE",
        }
    )

    DIRECTIONAL_STATES: FrozenSet[str] = frozenset(
        {
            "LEFT",
            "RIGHT",
            "UP",
            "DOWN",
        }
    )

    CENTER_STATE: str = "CENTER"

    def __init__(
        self,
        confirmation_count: int = 2,
        history_size: int = 4,
    ) -> None:
        """Initialize the DirectionStabilizer.

        Args:
            confirmation_count: Number of consecutive identical predictions
                required before switching active direction.
            history_size: Maximum number of recent raw predictions retained in
                the sliding window history.
        """
        if confirmation_count < 1:
            raise ValueError("confirmation_count must be at least 1.")
        if history_size < 1:
            raise ValueError("history_size must be at least 1.")

        self._confirmation_count: int = confirmation_count
        self._history_size: int = history_size
        self._history: Deque[str] = deque(maxlen=history_size)

        self._current_direction: str = self.CENTER_STATE
        self._candidate_direction: Optional[str] = None
        self._candidate_count: int = 0
        self._last_timestamp: Optional[float] = None

    @property
    def current_direction(self) -> str:
        """Return the current confirmed active direction ('CENTER', 'LEFT', etc.)."""
        return self._current_direction

    @property
    def candidate_direction(self) -> Optional[str]:
        """Return the candidate direction currently pending confirmation."""
        return self._candidate_direction

    @property
    def candidate_count(self) -> int:
        """Return the number of consecutive occurrences of the candidate direction."""
        return self._candidate_count

    @property
    def confirmation_count(self) -> int:
        """Return the required confirmation frame threshold."""
        return self._confirmation_count

    @property
    def history(self) -> List[str]:
        """Return a snapshot of recent raw predictions in the history buffer."""
        return list(self._history)

    @property
    def is_neutral(self) -> bool:
        """Return True if current confirmed direction is the neutral CENTER state."""
        return self._current_direction == self.CENTER_STATE

    def update(self, raw_prediction: str, timestamp: float) -> str:
        """Process a raw prediction frame and return the confirmed stable direction.

        Rules:
        1. Neutral states (FORWARD, EYES_OPEN, EYES_CLOSED, NO_FACE) immediately
           reset active direction to CENTER without waiting for confirmation.
        2. A directional prediction identical to the current active direction
           maintains that direction and resets candidate tracking.
        3. A new directional prediction becomes a candidate and is only confirmed
           after appearing consecutively for `confirmation_count` frames.
        4. Jitter / alternating predictions reset candidate counts and are rejected.

        Args:
            raw_prediction: Raw class label from Gaze Estimator.
            timestamp: Frame timestamp in seconds.

        Returns:
            The confirmed active direction string.
        """
        self._history.append(raw_prediction)
        self._last_timestamp = timestamp

        # -------------------------------------------------------------
        # 1. Neutral / Dead-Zone states -> Instant return to CENTER
        # -------------------------------------------------------------
        if (
            raw_prediction in self.NEUTRAL_STATES
            or raw_prediction not in self.DIRECTIONAL_STATES
        ):
            self._current_direction = self.CENTER_STATE
            self._candidate_direction = None
            self._candidate_count = 0
            return self._current_direction

        # -------------------------------------------------------------
        # 2. Raw prediction matches current active direction
        # -------------------------------------------------------------
        if raw_prediction == self._current_direction:
            self._candidate_direction = None
            self._candidate_count = 0
            return self._current_direction

        # -------------------------------------------------------------
        # 3. Candidate direction tracking
        # -------------------------------------------------------------
        if raw_prediction == self._candidate_direction:
            self._candidate_count += 1
        else:
            self._candidate_direction = raw_prediction
            self._candidate_count = 1

        # -------------------------------------------------------------
        # 4. Confirmation threshold check
        # -------------------------------------------------------------
        if self._candidate_count >= self._confirmation_count:
            self._current_direction = self._candidate_direction
            self._candidate_direction = None
            self._candidate_count = 0
            return self._current_direction

        # If not yet confirmed, retain the existing active direction
        return self._current_direction

    def reset(self) -> None:
        """Reset internal stabilizer state to initial values.

        Clears active direction, candidate tracking, counters, and history.
        """
        self._current_direction = self.CENTER_STATE
        self._candidate_direction = None
        self._candidate_count = 0
        self._history.clear()
        self._last_timestamp = None
