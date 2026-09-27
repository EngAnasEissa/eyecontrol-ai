"""Unit tests for DirectionStabilizer (Issue #1: Prediction Filtering and Direction Stabilization).

These tests define the expected behavior of the direction stabilizer before
implementing the production code (Test-Driven Development).

Expected Interface:
    DirectionStabilizer(confirmation_count=2, history_size=4)
    - stabilizer.update(raw_prediction: str, timestamp: float) -> str
    - stabilizer.reset() -> None
    - stabilizer.current_direction -> str
"""

import unittest

try:
    from app.interaction.stabilizer import DirectionStabilizer
except ImportError:
    DirectionStabilizer = None


class TestDirectionStabilizer(unittest.TestCase):
    """Test suite for DirectionStabilizer logic and state transitions."""

    def setUp(self) -> None:
        """Initialize a fresh stabilizer before each test."""
        if DirectionStabilizer is None:
            self.fail(
                "DirectionStabilizer is not implemented yet. "
                "Expected location: app/interaction/stabilizer.py"
            )

        self.stabilizer = DirectionStabilizer(
            confirmation_count=2,
            history_size=4,
        )
        self.now = 1000.0

    def step(self, raw_prediction: str, dt: float = 0.02) -> str:
        """Helper to advance time and update stabilizer."""
        self.now += dt
        return self.stabilizer.update(raw_prediction, self.now)

    # -----------------------------------------------------------------------
    # 1. Basic Confirmation: LEFT then LEFT
    # -----------------------------------------------------------------------
    def test_left_then_left_confirms_direction(self) -> None:
        """First LEFT is candidate only; second consecutive LEFT confirms."""
        # Frame 1: First appearance of LEFT
        result_1 = self.step("LEFT")
        self.assertEqual(
            result_1,
            "CENTER",
            "Single unconfirmed LEFT must not trigger movement (should be CENTER).",
        )
        self.assertEqual(
            self.stabilizer.current_direction,
            "CENTER",
        )

        # Frame 2: Second consecutive LEFT confirms the direction
        result_2 = self.step("LEFT")
        self.assertEqual(
            result_2,
            "LEFT",
            "Second consecutive LEFT must confirm active direction to LEFT.",
        )
        self.assertEqual(
            self.stabilizer.current_direction,
            "LEFT",
        )

    # -----------------------------------------------------------------------
    # 2. Isolated Noise: LEFT then RIGHT
    # -----------------------------------------------------------------------
    def test_isolated_right_does_not_change_stable_left(self) -> None:
        """Single isolated RIGHT must not switch away from established LEFT."""
        # Establish stable LEFT
        self.step("LEFT")
        self.step("LEFT")
        self.assertEqual(self.stabilizer.current_direction, "LEFT")

        # Frame 3: Isolated noise frame
        result = self.step("RIGHT")
        self.assertEqual(
            result,
            "LEFT",
            "Single isolated RIGHT must not alter active stable direction from LEFT.",
        )
        self.assertEqual(
            self.stabilizer.current_direction,
            "LEFT",
        )

    # -----------------------------------------------------------------------
    # 3. Switching: LEFT -> LEFT -> RIGHT -> RIGHT
    # -----------------------------------------------------------------------
    def test_direction_switch_requires_two_consecutive_frames(self) -> None:
        """Switching from LEFT to RIGHT requires 2 consecutive RIGHT frames."""
        # Establish stable LEFT
        self.step("LEFT")
        self.step("LEFT")
        self.assertEqual(self.stabilizer.current_direction, "LEFT")

        # Frame 1 of new direction (Candidate only)
        result_switch_1 = self.step("RIGHT")
        self.assertEqual(
            result_switch_1,
            "LEFT",
            "First RIGHT must keep active direction as LEFT until confirmed.",
        )

        # Frame 2 of new direction (Confirmed)
        result_switch_2 = self.step("RIGHT")
        self.assertEqual(
            result_switch_2,
            "RIGHT",
            "Second consecutive RIGHT must switch active direction to RIGHT.",
        )
        self.assertEqual(
            self.stabilizer.current_direction,
            "RIGHT",
        )

    # -----------------------------------------------------------------------
    # 4. Rapid Oscillation / Jitter: LEFT, RIGHT, LEFT, RIGHT
    # -----------------------------------------------------------------------
    def test_rapid_oscillation_jitter_rejected(self) -> None:
        """Rapid alternating predictions must never confirm any new direction."""
        stream = ["LEFT", "RIGHT", "LEFT", "RIGHT"]

        for item in stream:
            result = self.step(item)
            self.assertEqual(
                result,
                "CENTER",
                f"Oscillating frame '{item}' must not confirm any direction.",
            )

        self.assertEqual(
            self.stabilizer.current_direction,
            "CENTER",
            "Active direction must remain CENTER throughout jitter.",
        )

    # -----------------------------------------------------------------------
    # 5. Immediate Neutral Passthrough: Direction -> FORWARD
    # -----------------------------------------------------------------------
    def test_forward_instantly_returns_to_center(self) -> None:
        """FORWARD must immediately revert direction to CENTER without delay."""
        # Establish stable LEFT
        self.step("LEFT")
        self.step("LEFT")
        self.assertEqual(self.stabilizer.current_direction, "LEFT")

        # Looking straight ahead (FORWARD)
        result = self.step("FORWARD")
        self.assertEqual(
            result,
            "CENTER",
            "FORWARD is neutral dead-zone and must immediately return CENTER.",
        )
        self.assertEqual(
            self.stabilizer.current_direction,
            "CENTER",
        )

    # -----------------------------------------------------------------------
    # 6. Lost Face Safety: Direction -> NO_FACE
    # -----------------------------------------------------------------------
    def test_no_face_instantly_returns_to_center(self) -> None:
        """NO_FACE must immediately revert direction to CENTER for safety."""
        # Establish stable UP
        self.step("UP")
        self.step("UP")
        self.assertEqual(self.stabilizer.current_direction, "UP")

        # Face lost
        result = self.step("NO_FACE")
        self.assertEqual(
            result,
            "CENTER",
            "NO_FACE must instantly stop directional movement and return CENTER.",
        )
        self.assertEqual(
            self.stabilizer.current_direction,
            "CENTER",
        )

    # -----------------------------------------------------------------------
    # 7. Dead Zones: EYES_OPEN and EYES_CLOSED
    # -----------------------------------------------------------------------
    def test_eyes_open_is_treated_as_neutral(self) -> None:
        """EYES_OPEN is a neutral state and produces CENTER."""
        self.step("EYES_OPEN")
        result = self.step("EYES_OPEN")
        self.assertEqual(result, "CENTER")

    def test_eyes_closed_instantly_neutralizes_active_direction(self) -> None:
        """Closing eyes (blink or click hold) must immediately neutralize movement."""
        # Establish stable RIGHT
        self.step("RIGHT")
        self.step("RIGHT")
        self.assertEqual(self.stabilizer.current_direction, "RIGHT")

        # User closes eyes to click or blink
        result = self.step("EYES_CLOSED")
        self.assertEqual(
            result,
            "CENTER",
            "EYES_CLOSED must immediately return CENTER to prevent cursor drift.",
        )
        self.assertEqual(
            self.stabilizer.current_direction,
            "CENTER",
        )

    # -----------------------------------------------------------------------
    # 8. Reset Functionality
    # -----------------------------------------------------------------------
    def test_reset_clears_all_internal_state(self) -> None:
        """reset() must clear active direction, candidate direction, and history."""
        # Establish stable LEFT
        self.step("LEFT")
        self.step("LEFT")
        self.assertEqual(self.stabilizer.current_direction, "LEFT")

        # Call reset
        self.stabilizer.reset()
        self.assertEqual(
            self.stabilizer.current_direction,
            "CENTER",
            "reset() must restore current_direction to CENTER.",
        )

        # After reset, next single LEFT must again be unconfirmed
        result_after_reset = self.step("LEFT")
        self.assertEqual(
            result_after_reset,
            "CENTER",
            "After reset(), a single LEFT must require re-confirmation.",
        )

    # -----------------------------------------------------------------------
    # 9. Vertical Transitions: UP then DOWN
    # -----------------------------------------------------------------------
    def test_vertical_up_to_down_transition(self) -> None:
        """Vertical transitions work identically to horizontal transitions."""
        # Establish UP
        self.step("UP")
        result_up = self.step("UP")
        self.assertEqual(result_up, "UP")

        # First DOWN (Candidate only)
        result_down_1 = self.step("DOWN")
        self.assertEqual(result_down_1, "UP")

        # Second DOWN (Confirmed)
        result_down_2 = self.step("DOWN")
        self.assertEqual(result_down_2, "DOWN")
        self.assertEqual(self.stabilizer.current_direction, "DOWN")

    # -----------------------------------------------------------------------
    # 10. Glitch Recovery: LEFT -> Noise (UP) -> LEFT
    # -----------------------------------------------------------------------
    def test_isolated_noise_frame_preserves_stable_direction(self) -> None:
        """A single glitch frame does not reset or derail active direction."""
        # Establish LEFT
        self.step("LEFT")
        self.step("LEFT")
        self.assertEqual(self.stabilizer.current_direction, "LEFT")

        # Single noise frame UP
        result_glitch = self.step("UP")
        self.assertEqual(result_glitch, "LEFT")

        # Immediately back to LEFT
        result_back = self.step("LEFT")
        self.assertEqual(
            result_back,
            "LEFT",
            "Returning to LEFT after 1 glitch frame must remain stable LEFT.",
        )


if __name__ == "__main__":
    unittest.main()
