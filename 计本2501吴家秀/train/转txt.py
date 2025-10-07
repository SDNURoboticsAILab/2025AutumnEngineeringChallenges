import os
import cv2


def ccpd_to_yolo(ccpd_img_dir, yolo_txt_dir, class_id=0):
    os.makedirs(yolo_txt_dir, exist_ok=True)
    for img_name in os.listdir(ccpd_img_dir):
        if not img_name.endswith(('.jpg', '.png')):
            continue
        img_path = os.path.join(ccpd_img_dir, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue
        img_h, img_w = img.shape[:2]

        name_prefix = img_name.split('.')[0]
        segments = name_prefix.split('-')  # 按 - 分割文件名前缀

        try:
            # 假设坐标信息在 segments[2] 和 segments[3]，需根据实际文件名调整
            coord_segment1 = segments[2]
            coord_segment2 = segments[3]

            # 处理 coord_segment1，先按 _ 分割，再按 & 分割，确保得到纯数字
            part1 = coord_segment1.split('_')[0].split('&')
            lt_x, lt_y = map(int, part1)

            # 处理 coord_segment2，先按 _ 分割，再按 & 分割，确保得到纯数字
            part2 = coord_segment2.split('_')[0].split('&')
            rb_x, rb_y = map(int, part2)
        except (IndexError, ValueError) as e:
            print(f"文件名 {img_name} 格式异常，错误信息：{e}，跳过该文件")
            continue

        cx = (lt_x + rb_x) / 2 / img_w
        cy = (lt_y + rb_y) / 2 / img_h
        w = (rb_x - lt_x) / img_w
        h = (rb_y - lt_y) / img_h

        txt_name = name_prefix + '.txt'
        with open(os.path.join(yolo_txt_dir, txt_name), 'w') as f:
            f.write(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")


ccpd_img_dir = "D:/yolo/train/images"
yolo_txt_dir = "D:/yolo/train/labels"
ccpd_to_yolo(ccpd_img_dir, yolo_txt_dir)
print("Conversion completed!")