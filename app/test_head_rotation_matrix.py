
import cv2
import numpy as np

from app.vision.face_landmarks import FaceLandmarksDetector


def print_rotation_matrix(detector):
    """Print the raw 3x3 rotation matrix and current Euler values."""

    matrix = detector.get_last_transformation_matrix()

    if matrix is None:
        print("\nNo transformation matrix available.")
        return

    rotation = matrix[:3, :3]

    print("\n" + "=" * 50)
    print("RAW ROTATION MATRIX")
    print("=" * 50)

    print(
        f"[ {rotation[0, 0]: .6f}  "
        f"{rotation[0, 1]: .6f}  "
        f"{rotation[0, 2]: .6f} ]"
    )

    print(
        f"[ {rotation[1, 0]: .6f}  "
        f"{rotation[1, 1]: .6f}  "
        f"{rotation[1, 2]: .6f} ]"
    )

    print(
        f"[ {rotation[2, 0]: .6f}  "
        f"{rotation[2, 1]: .6f}  "
        f"{rotation[2, 2]: .6f} ]"
    )

    print("\nCurrent Euler interpretation:")

    pose = detector.get_head_pose()

    if pose["valid"]:
        print(f"Yaw   : {pose['yaw']:.2f} deg")
        print(f"Pitch : {pose['pitch']:.2f} deg")
        print(f"Roll  : {pose['roll']:.2f} deg")
    else:
        print("Euler values are INVALID.")

    print("=" * 50)


def main():
    print("Head Rotation Matrix Diagnostic")
    print("--------------------------------")
    print("Controls:")
    print("  S = capture/print current matrix")
    print("  Q = quit")
    print()
    print("Keep your head still for a moment before pressing S.")
    print("Capture these positions separately:")
    print("  1. Straight")
    print("  2. Head turned RIGHT")
    print("  3. Head turned LEFT")
    print("  4. Head tilted UP")
    print("  5. Head tilted DOWN")
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
                print("ERROR: Could not read camera frame.")
                break

            landmarks = detector.process_frame(frame)

            if landmarks is not None:
                detector.draw_face_mesh(
                    frame,
                    landmarks,
                    radius=1,
                )

                pose = detector.get_head_pose()

                if pose["valid"]:
                    text = (
                        f"Y:{pose['yaw']:.1f} "
                        f"P:{pose['pitch']:.1f} "
                        f"R:{pose['roll']:.1f}"
                    )

                    cv2.putText(
                        frame,
                        text,
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
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
                "S = capture matrix | Q = quit",
                (20, frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
            )

            cv2.imshow(
                "Head Rotation Matrix Diagnostic",
                frame,
            )

            key = cv2.waitKey(1) & 0xFF

            if key == ord("s"):
                print_rotation_matrix(detector)

            elif key == ord("q"):
                break

    finally:
        camera.release()
        cv2.destroyAllWindows()
        detector.close()

        print("\nDiagnostic finished.")


if __name__ == "__main__":
    main()

