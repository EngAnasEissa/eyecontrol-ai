

import cv2
import numpy as np

from app.vision.face_landmarks import FaceLandmarksDetector



def calculate_head_axes(matrix):
    """
    Extract two diagnostic head-motion angles directly
    from the rotation matrix.

    horizontal = left/right head rotation
    vertical   = up/down head rotation
    """

    rotation = matrix[:3, :3]

    # Horizontal rotation:
    # Uses the X/Z relationship in the rotation matrix.
    horizontal = np.degrees(
        np.arctan2(
            rotation[0, 2],
            rotation[2, 2],
        )
    )

    # Vertical rotation:
    # Uses the Y/Z relationship in the rotation matrix.
    vertical = np.degrees(
        np.arctan2(
            -rotation[2, 1],
            rotation[1, 1],
        )
    )

    return float(horizontal), float(vertical)


def main():
    print("=" * 60)
    print("HEAD AXES DIAGNOSTIC")
    print("=" * 60)
    print()
    print("Controls:")
    print("  S = capture current values")
    print("  Q = quit")
    print()
    print("Test these positions:")
    print("  1. Straight")
    print("  2. Turn RIGHT")
    print("  3. Turn LEFT")
    print("  4. Look UP")
    print("  5. Look DOWN")
    print()

    detector = FaceLandmarksDetector()
    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print("ERROR: Could not open camera.")
        detector.close()
        return

    try:
        while True:
            success, frame = camera.read()

            if not success:
                print("ERROR: Could not read frame.")
                break

            landmarks = detector.process_frame(frame)

            if landmarks is not None:
                detector.draw_face_mesh(
                    frame,
                    landmarks,
                    radius=1,
                )

                matrix = detector.get_last_transformation_matrix()

                if matrix is not None:
                    horizontal, vertical = calculate_head_axes(matrix)

                    cv2.putText(
                        frame,
                        f"Horizontal: {horizontal:7.2f} deg",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.75,
                        (0, 255, 0),
                        2,
                    )

                    cv2.putText(
                        frame,
                        f"Vertical:   {vertical:7.2f} deg",
                        (20, 75),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.75,
                        (0, 255, 0),
                        2,
                    )

            else:
                cv2.putText(
                    frame,
                    "NO FACE",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2,
                )

            cv2.putText(
                frame,
                "S = capture | Q = quit",
                (20, frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
            )

            cv2.imshow(
                "Head Axes Diagnostic",
                frame,
            )

            key = cv2.waitKey(1) & 0xFF

            if key == ord("s"):
                matrix = detector.get_last_transformation_matrix()

                if matrix is None:
                    print("\nNo valid matrix.")
                    continue

                horizontal, vertical = calculate_head_axes(matrix)

                print("\n" + "=" * 50)
                print("CAPTURED HEAD AXES")
                print("=" * 50)
                print(f"Horizontal = {horizontal:.2f} deg")
                print(f"Vertical   = {vertical:.2f} deg")
                print("=" * 50)

            elif key == ord("q"):
                break

    finally:
        camera.release()
        cv2.destroyAllWindows()
        detector.close()

        print("\nDiagnostic finished.")


if __name__ == "__main__":
    main()

