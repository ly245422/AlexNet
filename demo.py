import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageDraw, ImageTk
from models import AlexNet
from utils import plot
import torchvision.transforms as transforms

def detect(image_path, result_save_path):
    # 打开图像
    image = Image.open(image_path)
    
    # 将图像 resize 成 224x224
    image = image.resize((224, 224), Image.ANTIALIAS)
    
    # 创建可编辑的图像副本
    draw = ImageDraw.Draw(image)

    model = AlexNet(pretrained_weights_path="Carboard Detection/best.pth")
    transform = transforms.Compose([
        transforms.ToTensor(),  # 转换为张量
    ])
    tensor_image = transform(image).unsqueeze(0)  # 添加批次维度
    # 调用模型预测结果
    result = model(tensor_image)
    recs = plot(result, image_path)

    for i in range(len(recs)):
        x1 = recs[i][0]
        y1 = recs[i][1]
        x2 = recs[i][2]
        y2 = recs[i][3]
        draw.rectangle([x1, y1, x2, y2], outline="blue", width=1)

        # 显示类别为 "defect"
        draw.text((x1 + 10, y1 + 10), "defect", fill="blue")

    root.after(3000)
    result_image = ImageTk.PhotoImage(image)
    result_label.configure(image=result_image)
    result_label.image = result_image

    result_label.update_idletasks()

    # 保存结果图片
    if result_save_path:
        image.save(result_save_path)

def select_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
    if file_path:
        root.after(2000, lambda: detect(file_path, result_save_entry.get()))

def clear_result():
    # 清空结果显示
    result_label.configure(image=None)
    result_label.image = None

# 创建主窗口
root = tk.Tk()
root.title("Demo")

# 设置窗口初始大小
root.geometry("630x280")

# 左侧：结果图片显示标签
result_label = tk.Label(root)
result_label.place(x=30, y=20)

# 右侧：选择图片按钮
select_button = tk.Button(text="Select Image", command=select_image)
select_button.place(x=300, y=40)

# 右侧：输入文本框用于填写结果图片保存地址
result_save_label = tk.Label(text="Result Save Path:")
result_save_label.place(x=300, y=90)

result_save_entry = tk.Entry(width=40)
result_save_entry.place(x=300, y=140)

# 右侧：清空结果按钮
clear_button = tk.Button(text="Clear Result", command=clear_result)
clear_button.place(x=300, y=190)


# 运行主循环
root.mainloop()
