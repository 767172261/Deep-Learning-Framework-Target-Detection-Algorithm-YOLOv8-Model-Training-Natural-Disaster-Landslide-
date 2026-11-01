深度学习框架目标检测算法YOLOv8模型 训练自然灾害滑坡落石塌方检测数据集 通过训练出的权重推理 识别检测滑坡 倒树塌方 落石检测识别
@[toc]
以下文字及代码仅供参考。学习，同学。
自然灾害滑坡落石塌方检测数据集
 4 类：
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/a2e3ee4dceea47ffa5af3e55ec740950.png)

- `landslide`（滑坡）：7818 个标注框  
- `fallen tree`（倒树）：11037 个标注框  
- `stone`（落石）：25155 个标注框  
- `road collapse`（道路塌陷）：6416 个标注框  
- 总图像数量：21357 张  

---![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/27bcb1727e5242ea9cab2e6fb4e93e0c.png)


## 📊 一、类别数量统计表格

| 类别 ID | 类别名称         | 标注框数量 |
|---------|------------------|------------|
| 0       | landslide        | 7818       |
| 1       | fallen tree      | 11037      |
| 2       | stone            | 25155      |
| 3       | road collapse    | 6416       |

总标注框数：**49,426**

---![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/282cc6e8280e4093bd2e7c1dc570bb2e.png)


## 🧱 二、YOLOv8 训练这套自然灾害数据集
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/ed26463c3302458d822dee2d0112a80f.png)

### ✅ 1. 数据准备
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/43a402134527480ea0d843ba879c9100.png)

#### ✅ 1.1 数据结构要求

YOLOv8   YOLO 格式（`.txt` 文件 + `.jpg/.png` 图像），每个图像对应一个标签文件。每行内容如下：
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/2f55f26b1f7046b697155898022f3ce2.png)

```
class_id x_center y_center width height
```

- 所有值都是相对于图像宽度和高度的归一化坐标（0~1）
- class_id 对应类别索引（从 0 开始）

#### ✅ 1.2 划分训练集 / 验证集 / 测试集

建议按照 **8:1:1** 的比例划分：

```bash
Total images = 21357
Train: 17086
Val: 2136
Test: 2135
```

你可以使用以下代码来划分数据集：

```python
import os
import random
from shutil import copyfile

# 设置路径
image_dir = 'path/to/images'
label_dir = 'path/to/labels'

output_dir = 'dataset_natural_disaster'
os.makedirs(output_dir, exist_ok=True)
for split in ['train', 'val', 'test']:
    os.makedirs(os.path.join(output_dir, 'images', split), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'labels', split), exist_ok=True)

# 获取所有图像列表
image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg') or f.endswith('.png')]
random.shuffle(image_files)

# 按照 8:1:1 分割
train_split = int(0.8 * len(image_files))
val_split = int(0.9 * len(image_files))

train_files = image_files[:train_split]
val_files = image_files[train_split:val_split]
test_files = image_files[val_split:]

def copy_data(files, src_img_dir, src_label_dir, dst_dir, split):
    for file in files:
        base_name = os.path.splitext(file)[0]
        img_src = os.path.join(src_img_dir, file)
        label_src = os.path.join(src_label_dir, base_name + '.txt')
        
        img_dst = os.path.join(dst_dir, 'images', split, file)
        label_dst = os.path.join(dst_dir, 'labels', split, base_name + '.txt')

        copyfile(img_src, img_dst)
        if os.path.exists(label_src):
            copyfile(label_src, label_dst)

copy_data(train_files, image_dir, label_dir, output_dir, 'train')
copy_data(val_files, image_dir, label_dir, output_dir, 'val')
copy_data(test_files, image_dir, label_dir, output_dir, 'test')

print("✅ Dataset split completed.")
```

---

### ✅ 2. 创建 YOLOv8 数据配置文件 `data.yaml`

```yaml
train: dataset_natural_disaster/images/train/
val: dataset_natural_disaster/images/val/
test: dataset_natural_disaster/images/test/

nc: 4
names: ['landslide', 'fallen tree', 'stone', 'road collapse']
```

保存为 `data.yaml`。

---

### ✅ 3. 安装 YOLOv8 环境

```bash
# 克隆仓库
git clone https://github.com/ultralytics/ultralytics.git
cd ultralytics
pip install -r requirements.txt
```

---

### ✅ 4. 模型选择与训练

YOLOv8 可支持多种模型结构：`yolov8n`, `yolov8s`, `yolov8m`, `yolov8l`, `yolov8x`

推荐使用 `yolov8s` 或 `yolov8m` 进行训练。

#### 🚀 单卡训练命令：

```bash
yolo train data=data.yaml model=yolov8s.pt epochs=100 batch=16 imgsz=640
```

参数说明：

| 参数 | 含义 |
|------|------|
| `data` | 数据配置文件路径 |
| `model` | 使用的预训练模型权重 |
| `epochs` | 总训练轮次 |
| `batch` | 批大小 |
| `imgsz` | 输入图像尺寸 |

---

### ✅ 5. 可视化训练过程

训练过程中会自动在 `runs/train/exp` 下生成可视化日志和模型权重文件。

可以使用 TensorBoard 查看训练曲线：

```bash
tensorboard --logdir=runs
```

---

### ✅ 6. 推理代码

#### 🔍 单张图片推理：

```python
from ultralytics import YOLO

# 加载训练好的模型
model = YOLO('runs/train/exp/weights/best.pt')

# 推理单张图片
results = model('test.jpg')

# 显示结果
for r in results:
    print(r.boxes)  # 输出预测框信息
    annotated_img = r.plot()
    cv2.imshow('Detection', annotated_img)
    cv2.waitKey(0)
```

#### 📦 批量推理：

```python
results = model(['test1.jpg', 'test2.jpg'])

for result in results:
    boxes = result.boxes
    print(f"Detected {len(boxes)} objects")
```

---

### ✅ 7. 性能评估

验证模型性能（mAP 等指标）：

```bash
yolo val data=data.yaml model=runs/train/exp/weights/best.pt imgsz=640
```

输出结果包括：
- mAP@0.5
- mAP@0.5:0.95
- Precision / Recall
- F1-score 等

---

### ✅ 8. 导出 ONNX 模型

部署到边缘设备或 C++/Java 环境中：

```bash
yolo export model=runs/train/exp/weights/best.pt format=onnx
```

导出后得到 `best.onnx` 模型文件。

---

## 📌 三、总结

| 步骤 | 命令 / 工具 |
|------|-------------|
| 数据划分 | 自定义 Python 脚本 |
| 数据配置 | `data.yaml` |
| 环境搭建 | Ultralytics GitHub + pip |
| 模型训练 | `yolo train ...` |
| 推理 | `model.predict()` |
| 性能评估 | `yolo val ...` |
| 模型导出 | `yolo export ...` |

---


