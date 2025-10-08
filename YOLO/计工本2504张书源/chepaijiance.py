from ultralytics import YOLO
import cv2


class LicensePlateDetector:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def detect_image(self, image_path, conf=0.5):
        results = self.model.predict(
            source=image_path,
            conf=conf,
            save=True
        )
        return results

    def detect_video(self, video_path, conf=0.5):
        results = self.model.predict(
            source=video_path,
            conf=conf,
            save=True
        )
        return results

    def detect_webcam(self, conf=0.5):
        cap = cv2.VideoCapture(0)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = self.model(frame, conf=conf, verbose=False)
            annotated_frame = results[0].plot()
            cv2.imshow('License Plate Detection', annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


def main():
    detector = LicensePlateDetector('runs/detect/license_plate_v1/weights/best.pt')

    choice = input("模式 (1:图片, 2:视频, 3:摄像头): ")

    if choice == '1':
        image_path = input("图片路径: ")
        detector.detect_image(image_path)
    elif choice == '2':
        video_path = input("视频路径: ")
        detector.detect_video(video_path)
    elif choice == '3':
        detector.detect_webcam()


if __name__ == "__main__":
    main()