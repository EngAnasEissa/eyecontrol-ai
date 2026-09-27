
import cv2

from app.vision.face_landmarks import FaceLandmarksDetector


def main():
    print("Starting head-pose test...")
    print("Press Q to quit.")

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

            # Detect face landmarks.
            landmarks = detector.process_frame(frame)

            # Get the latest head-pose information.
            pose = detector.get_head_pose()

            if landmarks is not None and pose["valid"]:
                yaw = pose["yaw"]
                pitch = pose["pitch"]
                roll = pose["roll"]

                cv2.putText(
                    frame,
                    f"Yaw:   {yaw:7.2f} deg",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                )

                cv2.putText(
                    frame,
                    f"Pitch: {pitch:7.2f} deg",
                    (20, 75),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                )

                cv2.putText(
                    frame,
                    f"Roll:  {roll:7.2f} deg",
                    (20, 110),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                )

                # Draw the face landmarks.
                detector.draw_face_mesh(
                    frame,
                    landmarks,
                    radius=1,
                )

            else:
                cv2.putText(
                    frame,
                    "Head pose: INVALID / NO FACE",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2,
                )

            cv2.imshow("Head Pose Test", frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

    finally:
        camera.release()
        cv2.destroyAllWindows()
        detector.close()

        print("Head-pose test finished.")


if __name__ == "__main__":
    main()

