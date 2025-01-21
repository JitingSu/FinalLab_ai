import os
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import BertModel, BertTokenizer
from torchvision import transforms, models
from sklearn.model_selection import train_test_split
from model import MultimodalDataset, MultimodalModelvs  
from PIL import Image  
from torchvision.models import ResNet50_Weights
import wandb
import matplotlib.pyplot as plt  

# 参数设置
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 1e-5
MAX_LENGTH = 128
NUM_CLASSES = 3

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

# 加载数据
def load_data():
    with open(TRAIN_FILE, 'r') as file:
        lines = file.readlines()[1:]
    
    train_data, val_data = train_test_split(lines, test_size=0.2)
    
    train_dataset = MultimodalDataset(train_data, IMG_DIR, tokenizer, transform, MAX_LENGTH, IMG_DIR)
    val_dataset = MultimodalDataset(val_data, IMG_DIR, tokenizer, transform, MAX_LENGTH, IMG_DIR)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    return train_loader, val_loader

def initialize_model():
    text_model = BertModel.from_pretrained("bert-base-uncased")
    img_model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
    img_model.fc = torch.nn.Identity()  # Remove last fully connected layer
    model = MultimodalModelvs(text_model, img_model, NUM_CLASSES)
    return model

def train(model, train_loader, val_loader, device, use_text=True, use_image=True, model_name="BERT + ResNet50"):
    model = model.to(device)  # 确保模型在正确的设备上

    # 使用L2正则化（weight_decay）来防止过拟合
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    
    # 使用学习率调度器，每10个epoch调整一次学习率
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    
    train_losses = []
    train_accuracies = []
    val_accuracies = []
    
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for input_ids, attention_mask, img, label in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} - Training", unit="batch"):
            # 确保所有输入数据都在相同的设备上
            input_ids, attention_mask, img, label = input_ids.to(device), attention_mask.to(device), img.to(device), label.to(device)
            
            optimizer.zero_grad()
            output = model(input_ids, attention_mask, img, use_text=use_text, use_image=use_image)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
            # 计算训练准确率
            _, predicted = torch.max(output, 1)
            total_train += label.size(0)
            correct_train += (predicted == label).sum().item()
        
        avg_train_loss = running_loss / len(train_loader)
        avg_train_accuracy = 100 * correct_train / total_train
        train_losses.append(avg_train_loss)
        train_accuracies.append(avg_train_accuracy)
        
        print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {avg_train_loss:.4f}, Accuracy: {avg_train_accuracy:.2f}%")
        
        # 验证集评估
        val_accuracy = evaluate(model, val_loader, device, use_text=use_text, use_image=use_image)
        val_accuracies.append(val_accuracy)
        print(f"Validation Accuracy: {val_accuracy:.2f}%")
        
        # 学习率调度器
        scheduler.step()

    # 返回训练损失和验证准确率
    return train_losses, train_accuracies, val_accuracies

def evaluate(model, val_loader, device, use_text=True, use_image=True):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for input_ids, attention_mask, img, label in tqdm(val_loader, desc="Validation", unit="batch"):
            # 确保所有输入数据都在相同的设备上
            input_ids, attention_mask, img, label = input_ids.to(device), attention_mask.to(device), img.to(device), label.to(device)
            output = model(input_ids, attention_mask, img, use_text=use_text, use_image=use_image)
            _, predicted = torch.max(output, 1)
            total += label.size(0)
            correct += (predicted == label).sum().item()
    
    return 100 * correct / total

def predict(model, test_file, output_file, device, use_text=True, use_image=True):
    """
    使用训练好的模型进行预测，并将结果保存到文件
    """
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
            
            output = model(input_ids, attention_mask, img, use_text=use_text, use_image=use_image)
            _, predicted = torch.max(output, 1)
            predictions.append(predicted.item())
    
    with open(output_file, "w") as f:
        f.write("guid,tag\n")  
        for guid, label in zip(open(test_file, 'r').readlines()[1:], predictions):
            f.write(f"{guid.strip().split(',')[0]},{['positive', 'neutral', 'negative'][label]}\n")
    
    print(f"Prediction completed and saved to {output_file}")

def plot_metrics(train_losses_list, val_accuracies_list, model_names, save_dir="plots"):
    """
    绘制训练损失和验证准确率图表，并保存到本地
    """
    os.makedirs(save_dir, exist_ok=True)  # 创建保存图表的目录

    # 绘制训练损失图表
    plt.figure(figsize=(10, 5))
    for i, train_losses in enumerate(train_losses_list):
        plt.plot(train_losses, label=f"{model_names[i]} - Train Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.legend()
    plt.savefig(os.path.join(save_dir, "training_loss.png"))  # 保存图表
    plt.close()

    # 绘制验证准确率图表
    plt.figure(figsize=(10, 5))
    for i, val_accuracies in enumerate(val_accuracies_list):
        plt.plot(val_accuracies, label=f"{model_names[i]} - Val Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Validation Accuracy")
    plt.legend()
    plt.savefig(os.path.join(save_dir, "validation_accuracy.png"))  # 保存图表
    plt.close()

def main():
    # 设置设备为 GPU 或 CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 加载数据
    train_loader, val_loader = load_data()
    
    # 初始化模型并转移到指定设备
    model = initialize_model()
    model = model.to(device)  # 将模型转移到设备
    
    # 训练模型并记录结果
    train_losses_list = []
    val_accuracies_list = []
    model_names = ["Text Only", "Image Only", "Text + Image"]

    # 训练文本模型
    train_losses, train_accuracies, val_accuracies = train(model, train_loader, val_loader, device, use_text=True, use_image=False, model_name=model_names[0])
    train_losses_list.append(train_losses)
    val_accuracies_list.append(val_accuracies)

    # 训练图像模型
    train_losses, train_accuracies, val_accuracies = train(model, train_loader, val_loader, device, use_text=False, use_image=True, model_name=model_names[1])
    train_losses_list.append(train_losses)
    val_accuracies_list.append(val_accuracies)

    # 训练文本 + 图像模型
    train_losses, train_accuracies, val_accuracies = train(model, train_loader, val_loader, device, use_text=True, use_image=True, model_name=model_names[2])
    train_losses_list.append(train_losses)
    val_accuracies_list.append(val_accuracies)

    # 绘制并保存图表
    plot_metrics(train_losses_list, val_accuracies_list, model_names)

    # 预测
    predict(model, TEST_FILE, "predictions.txt", device, use_text=True, use_image=True)
    print("Prediction completed and saved to 'predictions.txt'")

if __name__ == "__main__":
    main()