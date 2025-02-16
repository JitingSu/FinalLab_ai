import os
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import BertModel, BertTokenizer
from torchvision import transforms, models
from sklearn.model_selection import train_test_split
from model import MultimodalDataset, MultimodalModel  
from PIL import Image  
from torchvision.models import ResNet50_Weights
import wandb

# 参数设置
BATCH_SIZE = 32
EPOCHS = 15
LEARNING_RATE = 1e-5
MAX_LENGTH = 128
NUM_CLASSES = 3
PATIENCE = 5  

# 数据路径们
DATA_DIR = "/kaggle/working/dataset"
TRAIN_FILE = os.path.join(DATA_DIR, "train.txt")
TEST_FILE = os.path.join(DATA_DIR, "test_without_label.txt")
IMG_DIR = os.path.join(DATA_DIR, "data")

# wandb初始化
wandb.init(
    project="FinalLab",  
    config={  
        "learning_rate": LEARNING_RATE,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "model": "BERT + ResNet50"
    }
)

# 数据预处理：设置BERT分词器和图像预处理
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", clean_up_tokenization_spaces=True)
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(30),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def load_data():
    """
    加载训练数据并划分为训练集和验证集，同时创建对应的数据集加载器。
    
    返回:
        train_loader (DataLoader): 训练集的数据加载器。
        val_loader (DataLoader): 验证集的数据加载器。
    """
    with open(TRAIN_FILE, 'r') as file:
        lines = file.readlines()[1:]
    
    train_data, val_data = train_test_split(lines, test_size=0.2)
    
    train_dataset = MultimodalDataset(train_data, IMG_DIR, tokenizer, transform, MAX_LENGTH, IMG_DIR)
    val_dataset = MultimodalDataset(val_data, IMG_DIR, tokenizer, transform, MAX_LENGTH, IMG_DIR)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    return train_loader, val_loader

def initialize_model():
    """
    初始化多模态模型，包括文本模型（BERT）和图像模型（ResNet50）。
    
    返回:
        model (MultimodalModel): 初始化后的多模态模型。
    """
    text_model = BertModel.from_pretrained("bert-base-uncased")
    img_model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
    img_model.fc = torch.nn.Identity()  
    model = MultimodalModel(text_model, img_model, NUM_CLASSES)
    return model

def train(model, train_loader, val_loader, device):
    """
    训练多模态模型，并在每个epoch后评估验证集准确率。
    
    参数:
        model (nn.Module): 要训练的多模态模型。
        train_loader (DataLoader): 训练集的数据加载器。
        val_loader (DataLoader): 验证集的数据加载器。
        device (torch.device): 训练设备（CPU或GPU）。
    """
    # 将模型移动到指定设备上进行训练
    model = model.to(device)
    
    # 使用交叉熵损失函数，并添加L2正则化（weight_decay）来防止过拟合
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    
    # 使用学习率调度器，每5个epoch调整一次学习率，学习率衰减因子为0.1
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    
    best_val_accuracy = 0
    train_losses = []
    train_accuracies = []
    val_accuracies = []
    
    # 早停计数器
    patience_counter = 0
    
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for input_ids, attention_mask, img, label in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} - Training", unit="batch"):
            input_ids, attention_mask, img, label = input_ids.to(device), attention_mask.to(device), img.to(device), label.to(device)
            
            optimizer.zero_grad()
            output = model(input_ids, attention_mask, img)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
            _, predicted = torch.max(output, 1)
            total_train += label.size(0)
            correct_train += (predicted == label).sum().item()
        
        avg_train_loss = running_loss / len(train_loader)
        avg_train_accuracy = 100 * correct_train / total_train
        train_losses.append(avg_train_loss)
        train_accuracies.append(avg_train_accuracy)
        
        print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {avg_train_loss}, Accuracy: {avg_train_accuracy}%")
        
        # 记录训练损失和准确率到wandb
        wandb.log({"train/loss": avg_train_loss, "train/accuracy": avg_train_accuracy})
        
        # 验证集评估
        val_accuracy = evaluate(model, val_loader, device)
        val_accuracies.append(val_accuracy)
        print(f"Validation Accuracy: {val_accuracy}%")
        
        # 记录验证准确率到wandb
        wandb.log({"val/accuracy": val_accuracy})

        # 如果当前验证准确率高于最佳验证准确率，则更新最佳验证准确率并保存模型
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), "best_model.pth")
            patience_counter = 0
        else:
            patience_counter += 1
        
        # 如果早停计数器达到阈值，则触发早停
        if patience_counter >= PATIENCE:
            print("Early stopping triggered, stopping training.")
            break
        
        # 更新学习率
        scheduler.step()

    wandb.finish()

def evaluate(model, val_loader, device):
    """
    在验证集上评估模型的性能。
    
    参数:
        model (nn.Module): 要评估的模型。
        val_loader (DataLoader): 验证集的数据加载器。
        device (torch.device): 评估设备（CPU或GPU）。
    
    返回:
        accuracy (float): 模型在验证集上的准确率。
    """
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for input_ids, attention_mask, img, label in tqdm(val_loader, desc="Validation", unit="batch"):
            input_ids, attention_mask, img, label = input_ids.to(device), attention_mask.to(device), img.to(device), label.to(device)
            output = model(input_ids, attention_mask, img)
            _, predicted = torch.max(output, 1)
            total += label.size(0)
            correct += (predicted == label).sum().item()
    
    return 100 * correct / total

def predict(model, test_file, output_file, device):
    """
    使用训练好的模型对测试数据进行预测，并将预测结果保存到文件中。
    
    参数:
        model (nn.Module): 训练好的多模态模型。
        test_file (str): 测试数据文件路径。
        output_file (str): 预测结果输出文件路径。
        device (torch.device): 预测设备（CPU或GPU）。
    """
    # 加载最佳模型的权重
    model.load_state_dict(torch.load("best_model.pth", weights_only=True))
    model.eval()
    model = model.to(device)
    
    predictions = []
    
    with open(test_file, "r") as f:
        lines = f.readlines()[1:] 
        
        for line in tqdm(lines, desc="Predicting", unit="sample"):
            guid = line.strip().split(",")[0]
            try:
                text = open(f"{DATA_DIR}/data/{guid}.txt", "r", encoding="ISO-8859-1").read()  
            except FileNotFoundError:
                print(f"Text file {guid}.txt not found. Skipping this sample.")
                continue
            
            text_encoding = tokenizer(text, truncation=True, padding="max_length", max_length=MAX_LENGTH, return_tensors="pt")
            
            try:
                img = Image.open(f"{DATA_DIR}/data/{guid}.jpg")  
            except FileNotFoundError:
                print(f"Image file {guid}.jpg not found. Skipping this sample.")
                continue

            img = transform(img).unsqueeze(0)
            input_ids = text_encoding.input_ids.to(device)
            attention_mask = text_encoding.attention_mask.to(device)
            img = img.to(device)
            
            output = model(input_ids, attention_mask, img)
            _, predicted = torch.max(output, 1)
            predictions.append(predicted.item())
    
    with open(output_file, "w") as f:
        f.write("guid,tag\n")  
        for guid, label in zip(open(test_file, 'r').readlines()[1:], predictions):
            f.write(f"{guid.strip().split(',')[0]},{['positive', 'neutral', 'negative'][label]}\n")
    
    print(f"Prediction completed and saved to {output_file}")

def main():
    """
    主函数，用于执行整个模型训练和预测流程。
    """
    # 检查是否有可用的CUDA设备，如果有则使用CUDA，否则使用CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader = load_data()
    model = initialize_model()
    model = model.to(device)
    train(model, train_loader, val_loader, device)
    predict(model, TEST_FILE, "predictions.txt", device)
    print("Prediction completed and saved to 'predictions.txt'")

if __name__ == "__main__":
    main()