
import cv2
import numpy as np

from app.vision.face_landmarks import FaceLandmarksDetector
from app.vision.eye_features import EyeFeatureExtractor


def calculate_head_axes(matrix):
    """Calculate horizontal and vertical head rotation."""

    rotation = matrix[:3, :3]

    horizontal = np.degrees(
        np.arctan2(
            rotation[0, 2],
            rotation[2, 2],
        )
    )

    vertical = np.degrees(
        np.arctan2(
            -rotation[2, 1],
            rotation[1, 1],
        )
    )

    return float(horizontal), float(vertical)

def print_features(features, horizontal, vertical):
    print(f"Head horizontal : {horizontal:8.2f} deg")
    print(f"Head vertical   : {vertical:8.2f} deg")

    if features is None:
        print("Eye features   : None")
        return

    print(f"Left normalized X  : {features.left_normalized_x:.6f}")
    print(f"Left normalized Y  : {features.left_normalized_y:.6f}")
    print(f"Right normalized X : {features.right_normalized_x:.6f}")
    print(f"Right normalized Y : {features.right_normalized_y:.6f}")

    print(f"Left iris X        : {features.left_iris_x:.6f}")
    print(f"Left iris Y        : {features.left_iris_y:.6f}")
    print(f"Right iris X       : {features.right_iris_x:.6f}")
    print(f"Right iris Y       : {features.right_iris_y:.6f}")

    print(f"Left EAR           : {features.left_ear:.6f}")
    print(f"Right EAR          : {features.right_ear:.6f}")
    print(f"Average EAR        : {features.avg_ear:.6f}")

    print(f"Face center X      : {features.face_center_x:.6f}")
    print(f"Face center Y      : {features.face_center_y:.6f}")

    print(f"Inter-eye distance : {features.inter_eye_distance:.6f}")


def main():
    print("=" * 65)
    print("EYE + HEAD FEATURE DIAGNOSTIC")
    print("=" * 65)
    print()
    print("Controls:")
    print("  S = capture current features")
    print("  Q = quit")
    print()
    print("Test these situations:")
    print()
    print("A) Head straight + eyes looking CENTER")
    print("B) Head straight + eyes LEFT")
    print("C) Head straight + eyes RIGHT")
    print("D) Head straight + eyes UP")
    print("E) Head straight + eyes DOWN")
    print()
    print("Then:")
    print()
    print("F) Eyes CENTER + head LEFT")
    print("G) Eyes CENTER + head RIGHT")
    print("H) Eyes CENTER + head UP")
    print("I) Eyes CENTER + head DOWN")
    print()

    detector = FaceLandmarksDetector()
    extractor = EyeFeatureExtractor()

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

                features = extractor.extract_features(
                    landmarks
                )

                matrix = detector.get_last_transformation_matrix()

                if matrix is not None:

                    head_horizontal, head_vertical = (
                        calculate_head_axes(matrix)
                    )

                    cv2.putText(
                        frame,
                        f"Head H: {head_horizontal:7.2f}",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2,
                    )

                    cv2.putText(
                        frame,
                        f"Head V: {head_vertical:7.2f}",
                        (20, 75),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2,
                    )

            else:
                features = None
                matrix = None

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
                "Eye + Head Feature Diagnostic",
                frame,
            )

            key = cv2.waitKey(1) & 0xFF

            if key == ord("s"):

                if features is None or matrix is None:
                    print("\nCannot capture: face or matrix unavailable.")
                    continue

                head_horizontal, head_vertical = (
                    calculate_head_axes(matrix)
                )

                print_features(
                    features,
                    head_horizontal,
                    head_vertical,
                )

            elif key == ord("q"):
                break

    finally:
        camera.release()
        cv2.destroyAllWindows()
        detector.close()

        print("\nDiagnostic finished.")


if __name__ == "__main__":
    main()

