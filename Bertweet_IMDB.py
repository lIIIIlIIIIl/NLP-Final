import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from datasets import load_dataset
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import time
import os

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(f"Using device: {device}")
# 加载 IMDb 数据集
dataset = load_dataset('imdb')
# 分词函数
def tokenize_function(example):
    return tokenizer(example['text'], padding='max_length', truncation=True, max_length=128)
# 加载 Bertweet 的分词器和模型
tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-base", use_fast=True)
model = AutoModelForSequenceClassification.from_pretrained("vinai/bertweet-base", num_labels=2)
model.to(device)
# 应用分词
tokenized_datasets = dataset.map(tokenize_function, batched=True)
tokenized_datasets = tokenized_datasets.remove_columns(['text']).rename_column("label", "labels")
tokenized_datasets.set_format('torch')
# 创建 DataLoader
train_dataset = tokenized_datasets['train']
test_dataset = tokenized_datasets['test']
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64)

# 定义优化器和学习率调度器
optimizer = AdamW(model.parameters(), lr=2e-5)
num_epochs = 100  # 设置较小的epoch以减少训练时间
num_training_steps = num_epochs * len(train_loader)
lr_scheduler = get_linear_schedule_with_warmup(optimizer,
                                               num_warmup_steps=0,
                                               num_training_steps=num_training_steps)
# 创建模型保存目录
save_dir = './saved_models'
os.makedirs(save_dir, exist_ok=True)
# 训练模型
training_loss = []
model.train()
for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs}")
    epoch_loss = 0
    start_time = time.time()
    for batch in train_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], labels=batch['labels'])
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        epoch_loss += loss.item()
    avg_epoch_loss = epoch_loss / len(train_loader)
    training_loss.append(avg_epoch_loss)
    end_time = time.time()
    print(f"Loss: {avg_epoch_loss:.4f} | Time: {end_time - start_time:.2f}s")
    model_save_path = os.path.join(save_dir, f'bertweet_epoch_{epoch + 1}.pt')
    tokenizer_save_path = os.path.join(save_dir, f'bertweet_tokenizer_epoch_{epoch + 1}')
    torch.save(model.state_dict(), model_save_path)
    tokenizer.save_pretrained(tokenizer_save_path)
    print(f"Model saved to {model_save_path}")

# 绘制训练损失图
plt.figure(figsize=(8, 6))
plt.plot(range(1, num_epochs + 1), training_loss, marker='o')
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid()
plt.show()

# 评估模型
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for batch in test_loader:
        batch = {k: v.to(device) for k, v in batch.items()}

        outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
        logits = outputs.logits
        preds = torch.argmax(logits, dim=-1).cpu().numpy()
        labels = batch['labels'].cpu().numpy()

        all_preds.extend(preds)
        all_labels.extend(labels)

# 计算混淆矩阵
cm = confusion_matrix(all_labels, all_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Negative', 'Positive'])

# 绘制混淆矩阵
fig, ax = plt.subplots(figsize=(6, 6))
disp.plot(ax=ax)
plt.title('Confusion Matrix')
plt.show()
