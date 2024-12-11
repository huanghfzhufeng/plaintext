import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from transformers import BertTokenizer, BertModel
from sklearn.metrics import f1_score, classification_report, accuracy_score
from tqdm import tqdm
from PIL import Image
from torchvision.models import resnet50

# 设置随机种子以确保可重复性
torch.manual_seed(42)
np.random.seed(42)

# 数据集类定义
class MultiModalDataset(Dataset):
    def __init__(self, excel_file, img_dir, tokenizer, max_len=256, transform=None):
        self.data = pd.read_excel(excel_file)
        self.img_dir = img_dir
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.transform = transform

        if 'label' not in self.data.columns:
            self.data['label'] = 0

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        text = str(item['text'])
        label = torch.tensor(item['label'], dtype=torch.long)

        # 文本处理：tokenization + padding
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors='pt'
        )

        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)

        # 图像处理
        images_list = item['images_list'].split('\t') if pd.notna(item['images_list']) else []
        images = []

        for img_name in images_list:
            img_path = os.path.join(self.img_dir, img_name)
            try:
                img = Image.open(img_path).convert('RGB')
            except (OSError, FileNotFoundError):
                img = Image.new('RGB', (224, 224), (255, 255, 255))

            if self.transform:
                img = self.transform(img)

            images.append(img)

        if len(images) > 1:
            images = torch.stack(images)
            images = images.mean(dim=0)
        elif len(images) == 1:
            images = images[0]
        else:
            images = torch.zeros(3, 224, 224)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'images': images,
            'label': label
        }


# 模型定义
# 文本编码器
class TextEncoder(nn.Module):
    def __init__(self, model_name='/models--hfl--chinese-roberta-wwm-ext'):
        super(TextEncoder, self).__init__()
        # 加载本地模型
        self.bert = BertModel.from_pretrained(model_name, local_files_only=True)
        self.dropout = nn.Dropout(0.3)
        self.output_dim = self.bert.config.hidden_size  # 通常为768

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        cls_output = outputs.last_hidden_state[:, 0, :]  # [batch_size, hidden_size]
        cls_output = self.dropout(cls_output)
        return cls_output
# 图像编码器
class ImageEncoder(nn.Module):
    def __init__(self, output_dim=512):
        super(ImageEncoder, self).__init__()
        # 使用ResNet50作为图像编码器
        resnet = resnet50(pretrained=True)
        # 冻结 ResNet 的前几层
        for name, param in resnet.named_parameters():
            if "layer4" not in name and "fc" not in name:
                param.requires_grad = False
        num_features = resnet.fc.in_features
        resnet.fc = nn.Linear(num_features, output_dim)
        self.model = resnet
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.model(x)
        x = self.dropout(x)
        return x


# 多模态分类器
class MultimodalClassifier(nn.Module):
    def __init__(self, num_classes, text_output_dim=768, image_output_dim=512, hidden_dim=512):
        super(MultimodalClassifier, self).__init__()
        self.text_encoder = TextEncoder()
        self.image_encoder = ImageEncoder(output_dim=image_output_dim)
        # 使用自注意力机制融合特征
        self.fc1 = nn.Linear(text_output_dim + image_output_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, input_ids, attention_mask, images):
        text_features = self.text_encoder(input_ids, attention_mask)  # [batch_size, text_output_dim]
        image_features = self.image_encoder(images)  # [batch_size, image_output_dim]
        
        combined_features = torch.cat((text_features, image_features), dim=1)  # [batch_size, text_output_dim + image_output_dim]
        x = self.fc1(combined_features)
        x = self.relu(x)
        x = self.dropout(x)
        logits = self.classifier(x)  # [batch_size, num_classes]
        return logits
# Focal Loss
class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=0.25, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')
    
    def forward(self, inputs, targets):
        logp = self.ce_loss(inputs, targets)
        p = torch.exp(-logp)
        loss = self.alpha * (1 - p) ** self.gamma * logp
        if self.reduction == 'mean':
            return loss.mean()
        else:
            return loss.sum()
#  训练器类
class Trainer:
    def __init__(self, model, train_dataloader, val_dataloader, optimizer, scheduler, criterion, num_epochs, device, patience=5, grad_clip=1.0):
        self.device = device
        self.model = model.to(device)
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.num_epochs = num_epochs
        self.patience = patience
        self.grad_clip = grad_clip

        self.best_accuracy = 0
        self.epochs_no_improve = 0
        self.scaler = GradScaler()

        if torch.cuda.device_count() > 1:
            self.model = torch.nn.DataParallel(self.model)
    def evaluate(self):
        self.model.eval()
        all_labels = []
        all_preds = []

        with torch.no_grad():
            for batch in self.val_dataloader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                images = batch['images'].to(self.device)
                labels = batch['label'].to(self.device)

                outputs = self.model(input_ids, attention_mask, images)
                _, preds = torch.max(outputs, dim=1)

                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())

        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='weighted')
        print(f"Validation Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}")
        print(classification_report(all_labels, all_preds))
        return accuracy, f1
    def train(self):
        for epoch in range(self.num_epochs):
            self.model.train()
            total_loss = 0
            pbar = tqdm(self.train_dataloader, desc=f"Epoch {epoch+1}/{self.num_epochs}")

            for batch in pbar:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                images = batch['images'].to(self.device)
                labels = batch['label'].to(self.device)

                self.optimizer.zero_grad()

                with autocast():
                    outputs = self.model(input_ids, attention_mask, images)
                    loss = self.criterion(outputs, labels)

                self.scaler.scale(loss).backward()
                # 梯度裁剪
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                # scheduler step will be called after validation
                total_loss += loss.item()
                avg_loss = total_loss / (pbar.n + 1)
                pbar.set_postfix({'Loss': f'{avg_loss:.4f}'})

            # Validation after each epoch
            accuracy, f1 = self.evaluate()
            self.scheduler.step(1.0 - accuracy)  # 假设 scheduler 是 ReduceLROnPlateau，需要传递 metric

            # 检查是否有提升
            if accuracy > self.best_accuracy:
                self.best_accuracy = accuracy
                self.epochs_no_improve = 0
                torch.save(self.model.state_dict(), './save/best_model.pth')
                print(f"New best model saved with accuracy: {self.best_accuracy:.4f}")
            else:
                self.epochs_no_improve += 1
                print(f"No improvement for {self.epochs_no_improve} epoch(s).")

            # 保存最后一个模型
            torch.save(self.model.state_dict(), './save/last_model.pth')

            # 早停
            if self.epochs_no_improve >= self.patience:
                print("Early stopping triggered.")
                break

#  数据预处理和加载
data_path = '/data/Part.xlsx'
model_path = '/models--hfl--chinese-roberta-wwm-ext'
image_path = '/data/Part'


# 加载本地的 tokenizer
tokenizer = BertTokenizer.from_pretrained(model_path, local_files_only=True)

# 图像数据增强和预处理
transform = transforms.Compose([
    transforms.Resize((256, 256)), 
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomRotation(10),
    transforms.RandomAffine(0, shear=10, scale=(0.8, 1.2)),
    transforms.RandomPerspective(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

dataset = MultiModalDataset(
    excel_file=data_path,
    img_dir=image_path,
    tokenizer=tokenizer,
    transform=transform
)

# 划分训练集和验证集
train_size = int(0.95 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(
    train_dataset,
    batch_size=16,  # 增加批量大小
    shuffle=True,
    num_workers=8,
    pin_memory=True
)
val_loader = DataLoader(
    val_dataset,
    batch_size=16,  # 增加批量大小
    shuffle=False,
    num_workers=8,
    pin_memory=True
)

# 模型训练
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = MultimodalClassifier(num_classes=2)

optimizer = optim.AdamW([
    {'params': model.text_encoder.bert.parameters(), 'lr': 2e-5},  # 调整学习率
    {'params': model.image_encoder.model.parameters(), 'lr': 2e-5},  # 调整学习率
    {'params': model.fc1.parameters(), 'lr': 1e-4},
    {'params': model.classifier.parameters(), 'lr': 1e-4},
], weight_decay=1e-5)

num_epochs = 50  # 增加训练轮数

scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2, verbose=True)

criterion = FocalLoss(gamma=2, alpha=0.25)

trainer = Trainer(
    model=model,
    train_dataloader=train_loader,
    val_dataloader=val_loader,
    optimizer=optimizer,
    scheduler=scheduler,
    criterion=criterion,
    num_epochs=num_epochs,
    device=device,
    patience=5,  # 早停耐心
    grad_clip=1.0  # 梯度裁剪
)

trainer.train()
