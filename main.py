import os
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import BertModel, BertTokenizer
from torchvision import transforms, models
from sklearn.model_selection import train_test_split
from model import MultimodalDataset, MultimodalModel  # 假设模型和数据集类已经在 model.py 中定义
from PIL import Image  # 需要导入PIL库用于处理图像

# 参数设置
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 1e-5
MAX_LENGTH = 128
NUM_CLASSES = 3

# 数据和模型路径
# DATA_DIR = r"e:\Learning_material\junior\AI\5_FinalLab\dataset"
DATA_DIR = "/kaggle/working/dataset"

TRAIN_FILE = os.path.join(DATA_DIR, "train.txt")
TEST_FILE = os.path.join(DATA_DIR, "test_without_label.txt")
IMG_DIR = os.path.join(DATA_DIR, "data")

# 设置BERT和图像预处理
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 加载数据
def load_data():
    with open(TRAIN_FILE, 'r') as file:
        lines = file.readlines()[1:]
    
    train_data, val_data = train_test_split(lines, test_size=0.2)
    
    train_dataset = MultimodalDataset(train_data, IMG_DIR, tokenizer, transform, MAX_LENGTH, IMG_DIR)
    val_dataset = MultimodalDataset(val_data, IMG_DIR, tokenizer, transform, MAX_LENGTH, IMG_DIR)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    return train_loader, val_loader

# 初始化模型
def initialize_model():
    text_model = BertModel.from_pretrained("bert-base-uncased")
    img_model = models.resnet50(pretrained=True)
    img_model.fc = torch.nn.Identity()  # 去掉ResNet的最后一层
    
    model = MultimodalModel(text_model, img_model, NUM_CLASSES)
    
    return model

# 训练过程
def train(model, train_loader, val_loader):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    best_val_accuracy = 0
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        
        # 包装 train_loader 以显示进度条
        for input_ids, attention_mask, img, label in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} - Training", unit="batch"):
            optimizer.zero_grad()
            output = model(input_ids, attention_mask, img)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {running_loss/len(train_loader)}")
        
        # 验证集评估
        val_accuracy = evaluate(model, val_loader)
        print(f"Validation Accuracy: {val_accuracy}%")
        
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), "best_model.pth")

# 验证过程
def evaluate(model, val_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        # 包装 val_loader 以显示进度条
        for input_ids, attention_mask, img, label in tqdm(val_loader, desc="Validation", unit="batch"):
            output = model(input_ids, attention_mask, img)
            _, predicted = torch.max(output, 1)
            total += label.size(0)
            correct += (predicted == label).sum().item()
    
    return 100 * correct / total

# 预测过程
def predict(model, test_file, output_file):
    model.load_state_dict(torch.load("best_model.pth"))
    model.eval()
    
    predictions = []
    
    # 读取测试集并包装为 tqdm 进度条
    with open(test_file, "r") as f:
        lines = f.readlines()
        
        for line in tqdm(lines, desc="Predicting", unit="sample"):
            guid = line.strip().split("\t")[0]
            text = open(f"{DATA_DIR}/{guid}.txt", "r").read()
            text_encoding = tokenizer(text, truncation=True, padding="max_length", max_length=MAX_LENGTH, return_tensors="pt")
            
            img = Image.open(f"{DATA_DIR}/images/{guid}.jpg")
            img = transform(img).unsqueeze(0)
            
            output = model(text_encoding.input_ids, text_encoding.attention_mask, img)
            _, predicted = torch.max(output, 1)
            predictions.append(predicted.item())
    
    # 保存预测结果
    with open(output_file, "w") as f:
        for guid, label in zip(open(test_file, 'r').readlines(), predictions):
            f.write(f"{guid.strip()}\t{['positive', 'neutral', 'negative'][label]}\n")

# 主函数
def main():
    train_loader, val_loader = load_data()
    model = initialize_model()
    train(model, train_loader, val_loader)
    predict(model, TEST_FILE, "predictions.txt")
    print("Prediction completed and saved to 'predictions.txt'")

if __name__ == "__main__":
    main()