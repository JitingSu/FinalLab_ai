import os
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import BertModel, BertTokenizer
from torchvision import transforms, models
from sklearn.model_selection import train_test_split
from model import MultimodalDataset, MultimodalModel  
from PIL import Image  
import matplotlib.pyplot as plt  

# 参数设置
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 1e-5
MAX_LENGTH = 128
NUM_CLASSES = 3

# 数据路径
DATA_DIR = "/kaggle/working/dataset"
TRAIN_FILE = os.path.join(DATA_DIR, "train.txt")
TEST_FILE = os.path.join(DATA_DIR, "test_without_label.txt")
IMG_DIR = os.path.join(DATA_DIR, "data")

# 设置BERT分词器和图像预处理
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", clean_up_tokenization_spaces=True)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    # 对图像进行归一化处理，使用的均值和标准差是基于ImageNet数据集的统计值
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 加载数据
def load_data():
    with open(TRAIN_FILE, 'r') as file:
        lines = file.readlines()[1:]
    
    # 划分为训练集和验证集
    train_data, val_data = train_test_split(lines, test_size=0.2)
    
    train_dataset = MultimodalDataset(train_data, IMG_DIR, tokenizer, transform, MAX_LENGTH, IMG_DIR)
    val_dataset = MultimodalDataset(val_data, IMG_DIR, tokenizer, transform, MAX_LENGTH, IMG_DIR)
    
    # train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    # val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    # 设置多个进程并行加载数据
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    return train_loader, val_loader

def initialize_model():
    """
    初始化多模态模型
    """
    # 从预训练的BERT模型中加载基础的、不区分大小写的版本
    text_model = BertModel.from_pretrained("bert-base-uncased")
    
    # 加载预训练的ResNet-50模型
    img_model = models.resnet50(pretrained=True)
    img_model.fc = torch.nn.Identity()  
    
    # 创建一个MultimodalModel实例，将文本模型和图像模型作为参数传递给它
    model = MultimodalModel(text_model, img_model, NUM_CLASSES)
    
    return model

def train(model, train_loader, val_loader):
    """
    训练多模态模型
    参数:
        model (nn.Module): 多模态模型
        train_loader (DataLoader): 训练数据加载器
        val_loader (DataLoader): 验证数据加载器
    """
    # 检查是否有可用的 GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 将模型迁移到 GPU
    model = model.to(device)
    
    # 定义损失函数，使用交叉熵损失
    criterion = torch.nn.CrossEntropyLoss()
    # 定义优化器，使用Adam优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    best_val_accuracy = 0
    train_losses = []
    val_accuracies = []
    
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        
        for input_ids, attention_mask, img, label in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} - Training", unit="batch"):
            # 将数据迁移到 GPU
            input_ids, attention_mask, img, label = input_ids.to(device), attention_mask.to(device), img.to(device), label.to(device)
            
            # 清空梯度
            optimizer.zero_grad()
            # 前向传播
            output = model(input_ids, attention_mask, img)
            # 计算损失
            loss = criterion(output, label)
            # 反向传播
            loss.backward()
            # 更新参数
            optimizer.step()
            # 累加损失
            running_loss += loss.item()
        
        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {avg_train_loss}")
        
        # 验证集评估
        val_accuracy = evaluate(model, val_loader, device)  # 传递device给evaluate函数
        val_accuracies.append(val_accuracy)
        print(f"Validation Accuracy: {val_accuracy}%")
        
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), "best_model.pth")
    
    # 绘制训练损失和验证准确率图表
    plot_metrics(train_losses, val_accuracies)


def evaluate(model, val_loader, device):
    """
    在验证集上评估模型的性能
    参数:
        model (nn.Module): 多模态模型
        val_loader (DataLoader): 验证数据加载器
        device (torch.device): 设备 (CPU 或 GPU)
    返回:
        accuracy (float): 模型在验证集上的准确率
    """
    # 将模型设置为评估模式，关闭dropout等训练时使用的操作
    model.eval()
    correct = 0
    total = 0
    
    # 不计算梯度，减少内存消耗并加速计算
    with torch.no_grad():
        # 包装 val_loader 以显示进度条
        for input_ids, attention_mask, img, label in tqdm(val_loader, desc="Validation", unit="batch"):
            # 将数据迁移到 GPU
            input_ids, attention_mask, img, label = input_ids.to(device), attention_mask.to(device), img.to(device), label.to(device)
            
            # 前向传播，计算模型输出
            output = model(input_ids, attention_mask, img)
            # 获取预测结果，即输出中概率最大的类别
            _, predicted = torch.max(output, 1)
            # 累加总样本数
            total += label.size(0)
            # 累加正确预测的样本数
            correct += (predicted == label).sum().item()
    
    # 计算准确率并返回
    return 100 * correct / total


# 绘制训练损失和验证准确率的图表并保存为文件
def plot_metrics(train_losses, val_accuracies):
    # 绘制训练损失
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Training Loss")
    plt.title("Training Loss vs. Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()

    # 绘制验证准确率
    plt.subplot(1, 2, 2)
    plt.plot(val_accuracies, label="Validation Accuracy", color='orange')
    plt.title("Validation Accuracy vs. Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy (%)")
    plt.legend()

    # 保存图表为图片文件
    plt.tight_layout()
    plt.savefig('training_metrics.png')  # 保存图表为PNG文件
    print("Training metrics plot saved as 'training_metrics.png'")

def predict(model, test_file, output_file, device):
    """
    在测试集上进行预测并保存结果
    参数:
        model (nn.Module): 训练好的多模态模型
        test_file (str): 测试集文件路径
        output_file (str): 预测结果保存路径
        device (torch.device): 设备 (CPU 或 GPU)
    """
    model.load_state_dict(torch.load("best_model.pth", weights_only=True))
    model.eval()
    
    # 将模型迁移到 GPU
    model = model.to(device)
    
    predictions = []
    
    with open(test_file, "r") as f:
        lines = f.readlines()[1:]  # 跳过第一行（包含标题）
        
        for line in tqdm(lines, desc="Predicting", unit="sample"):
            guid = line.strip().split(",")[0]  # 获取guid

            # 读取文本文件，注意加上 encoding 参数
            try:
                text = open(f"{DATA_DIR}/data/{guid}.txt", "r", encoding="ISO-8859-1").read()  
            except FileNotFoundError:
                print(f"Text file {guid}.txt not found. Skipping this sample.")
                continue
            
            # 对文本进行编码
            text_encoding = tokenizer(text, truncation=True, padding="max_length", max_length=MAX_LENGTH, return_tensors="pt")
            
            # 打开图像文件，确保图像路径正确
            try:
                img = Image.open(f"{DATA_DIR}/data/{guid}.jpg")  
            except FileNotFoundError:
                print(f"Image file {guid}.jpg not found. Skipping this sample.")
                continue

            # 对图像进行预处理并增加一个维度以匹配模型输入
            img = transform(img).unsqueeze(0)
            
            # 将数据迁移到 GPU
            input_ids = text_encoding.input_ids.to(device)
            attention_mask = text_encoding.attention_mask.to(device)
            img = img.to(device)
            
            # 模型预测
            output = model(input_ids, attention_mask, img)
            _, predicted = torch.max(output, 1)
            predictions.append(predicted.item())
    
    # 保存预测结果
    with open(output_file, "w") as f:
        f.write("guid,tag\n")  # 写入标题
        for guid, label in zip(open(test_file, 'r').readlines()[1:], predictions):  # 跳过第一行标题
            f.write(f"{guid.strip().split(',')[0]}\t{['positive', 'neutral', 'negative'][label]}\n")

    print(f"Prediction completed and saved to {output_file}")


def main():
    # 检查是否有可用的 GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 加载数据
    train_loader, val_loader = load_data()
    
    # 初始化模型
    model = initialize_model()
    
    # 将模型迁移到 GPU
    model = model.to(device)
    
    # 训练模型
    train(model, train_loader, val_loader, device)
    
    # 预测测试集
    predict(model, TEST_FILE, "predictions.txt", device)
    print("Prediction completed and saved to 'predictions.txt'")


if __name__ == "__main__":
    main()