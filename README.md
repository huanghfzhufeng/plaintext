# 多模态分类：基于文本与图像的联合学习模型

本项目实现了一个多模态分类系统，结合文本和图像特征，通过深度学习方法实现高效的分类任务。该模型集成了预训练的 BERT 模型用于文本编码，以及 ResNet50 模型用于图像特征提取，适用于多模态数据的分类场景。

---

## 项目特点

- **文本特征提取**：基于预训练的 BERT 模型（如 `chinese-roberta-wwm-ext`）进行高质量的文本表示。
- **图像特征提取**：采用 ResNet50 模型处理图像数据，并支持对部分网络层进行冻结以优化训练效率。
- **多模态特征融合**：通过特征拼接和自定义全连接层，实现文本与图像特征的联合建模。
- **不平衡问题处理**：通过 Focal Loss 减少类别不平衡对训练的影响。
- **数据增强**：使用多种图像增强技术（如随机裁剪、翻转、颜色抖动等），提升模型的泛化能力。
- **动态学习率调度**：基于 `ReduceLROnPlateau` 动态调整学习率，提高训练稳定性。
- **早停机制**：在模型性能停止提升时，自动终止训练以避免过拟合。

---

## 文件结构

```plaintext
├── data/                     # 数据存放目录
│   ├── Part.xlsx             # 数据表，包含文本、图像路径和标签
│   └── images/               # 图像文件目录
├── models/                   # 预训练模型目录
│   └── chinese-roberta-wwm-ext  # 本地的中文 BERT 模型
├── save/                     # 模型权重保存目录
│   ├── best_model.pth        # 验证集性能最优模型
│   └── last_model.pth        # 最后一次训练保存的模型
├── train.py                  # 主训练脚本
└── README.md                 # 项目说明文件

环境依赖
建议使用以下配置进行开发和运行：

Python: >= 3.8
PyTorch: >= 1.10
Transformers: >= 4.0
其他依赖:
torchvision
pandas
scikit-learn
tqdm
Pillow
安装依赖

pip install torch torchvision transformers pandas scikit-learn tqdm pillow

数据准备
文本和图像信息
将数据整理成 Excel 文件（如 data/Part.xlsx），需包含以下列：

text: 文本数据
images_list: 图像文件名，使用制表符 \t 分隔多个文件
label: 标签（如分类任务中的 0 或 1）
如果 label 列缺失，代码会默认赋值为 0。

图像文件存储
图像数据应存放在 data/images/ 目录下，并与 Excel 文件中的 images_list 字段对应。

预训练模型
下载中文 BERT 模型（如 chinese-roberta-wwm-ext），并将其存放在 models/ 目录下。

快速开始
训练模型
运行以下命令开始训练：

python train.py

评估模型
训练过程中，模型会在验证集上自动评估，并保存最佳模型至 save/best_model.pth。

结果展示
在二分类任务中，该多模态分类模型通过融合文本和图像特征，取得了以下性能指标（以示例数据为准）：

准确率（Accuracy）: 90%+
F1 分数: 90%+
详细的分类报告和混淆矩阵会在验证阶段自动生成。

贡献指南
欢迎您提出问题（Issue）或提交代码（Pull Request）。如果您有任何改进建议，请随时与我们联系！
