import os
folder_path = "D:/yolo/labels"
for filename in os.listdir(folder_path):
    if filename.endswith(".txt"):
        file_path = os.path.join(folder_path, filename)
        # 读取文件内容
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        # 在所有"0."前添加空格
        new_content = content.replace("0.", " 0.")
        # 写入修改后的内容
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(new_content)

print("批量处理完成！")