from ultralytics import YOLO
import cv2
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont


class LicensePlateRecognition:
    def __init__(self):
        self.output_dir = 'results'
        os.makedirs(self.output_dir, exist_ok=True)

        # 加载模型
        self.detector = YOLO('runs/detect/license_plate_v1/weights/best.pt')

        # 加载OCR
        try:
            from paddleocr import PaddleOCR
            self.ocr = PaddleOCR(use_gpu=False, show_log=False)
        except:
            self.ocr = None

        # 中文字体
        self.font_path = self._find_chinese_font()

    def _find_chinese_font(self):
        # 查找中文字体
        font_paths = [
            "C:/Windows/Fonts/simhei.ttf",
            "C:/Windows/Fonts/simsun.ttc",
            "C:/Windows/Fonts/msyh.ttc",
        ]
        for font_path in font_paths:
            if os.path.exists(font_path):
                return font_path
        return None

    def _put_chinese_text(self, image, text, position, font_size=20, color=(0, 0, 0)):
        # 绘制中文
        try:
            image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(image_pil)
            if self.font_path:
                font = ImageFont.truetype(self.font_path, font_size)
            else:
                font = ImageFont.load_default()
            draw.text(position, text, font=font, fill=color)
            return cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
        except:
            cv2.putText(image, text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            return image

    def recognize_image(self, image_path):
        # 识别图片
        if self.ocr is None:
            return None, []

        image = cv2.imread(image_path)
        if image is None:
            return None, []

        # 检测车牌
        results = self.detector(image, conf=0.5, verbose=False)
        plates_info = []

        for result in results:
            if len(result.boxes) == 0:
                return image, []

            boxes = result.boxes.xyxy.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()

            for box, conf in zip(boxes, confidences):
                x1, y1, x2, y2 = map(int, box)
                plate_region = image[y1:y2, x1:x2]

                if plate_region.size == 0:
                    continue

                # OCR识别
                plate_text = self._ocr_plate(plate_region)

                plate_info = {
                    'bbox': (x1, y1, x2, y2),
                    'confidence': conf,
                    'text': plate_text
                }
                plates_info.append(plate_info)

                # 绘制结果
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                text_bg_y = max(0, y1 - 30)
                cv2.rectangle(image, (x1, text_bg_y), (x2, y1), (0, 255, 0), -1)
                result_text = f"{plate_text} ({conf:.2f})"
                image = self._put_chinese_text(image, result_text, (x1, y1 - 25))

        return image, plates_info

    def process_video(self, video_path):
        # 处理视频
        if self.ocr is None:
            return

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return

        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        output_path = os.path.join(self.output_dir, "video_result.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = self.detector(frame, conf=0.5, verbose=False)

            for result in results:
                if len(result.boxes) > 0:
                    boxes = result.boxes.xyxy.cpu().numpy()
                    confidences = result.boxes.conf.cpu().numpy()

                    for box, conf in zip(boxes, confidences):
                        x1, y1, x2, y2 = map(int, box)
                        plate_region = frame[y1:y2, x1:x2]

                        if plate_region.size == 0:
                            continue

                        plate_text = self._ocr_plate(plate_region)

                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        text_bg_y = max(0, y1 - 30)
                        cv2.rectangle(frame, (x1, text_bg_y), (x2, y1), (0, 255, 0), -1)
                        result_text = f"{plate_text}"
                        frame = self._put_chinese_text(frame, result_text, (x1, y1 - 25))

            out.write(frame)
            cv2.imshow('Preview', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        out.release()
        cv2.destroyAllWindows()

    def _ocr_plate(self, plate_image):
        # OCR识别
        try:
            result = self.ocr.ocr(plate_image, cls=True)
            if not result:
                return "识别失败"

            texts = []
            for line in result:
                for word_info in line:
                    text = word_info[1][0]
                    confidence = word_info[1][1]
                    if confidence > 0.1:
                        texts.append(text)

            final_text = ''.join(texts)
            final_text = ''.join(filter(lambda x: x.isalnum() or '\u4e00' <= x <= '\u9fff', final_text))
            return final_text if final_text else "识别失败"
        except:
            return "识别错误"


def main():
    recognizer = LicensePlateRecognition()

    if recognizer.ocr is None:
        print("OCR初始化失败")
        return

    choice = input("模式 (1:图片, 2:视频): ")

    if choice == '1':
        image_path = input("图片路径: ")
        result_image, plates = recognizer.recognize_image(image_path)

        if result_image is not None:
            cv2.imshow('结果', result_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            filename = os.path.basename(image_path)
            output_path = os.path.join(recognizer.output_dir, f"result_{filename}")
            cv2.imwrite(output_path, result_image)

    elif choice == '2':
        video_path = input("视频路径: ")
        recognizer.process_video(video_path)


if __name__ == "__main__":
    main()