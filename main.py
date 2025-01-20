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
EPOCHS = 30
LEARNING_RATE = 1e-5
MAX_LENGTH = 128
NUM_CLASSES = 3
PATIENCE = 3  # 早停的耐心值

# 数据路径
DATA_DIR = "/kaggle/working/dataset"
TRAIN_FILE = os.path.join(DATA_DIR, "train.txt")
TEST_FILE = os.path.join(DATA_DIR, "test_without_label.txt")
IMG_DIR = os.path.join(DATA_DIR, "data")

# wandb初始化
wandb.init(
    project="FinalLab",  # 设置为你在wandb网站上创建的项目名称
    config={  # 配置你的超参数
        "learning_rate": LEARNING_RATE,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "model": "BERT + ResNet50"
    }
)

# 设置BERT分词器和图像预处理
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
    img_model.fc = torch.nn.Identity()  
    model = MultimodalModel(text_model, img_model, NUM_CLASSES)
    return model

def train(model, train_loader, val_loader, device):
    model = model.to(device)
    
    # 使用L2正则化（weight_decay）来防止过拟合
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    
    # 使用学习率调度器，每10个epoch调整一次学习率
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    
    best_val_accuracy = 0
    train_losses = []
    train_accuracies = []
    val_accuracies = []
    
    # 早停
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
            
            # 计算训练准确率
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

        # # 早停判断
        # if val_accuracy > best_val_accuracy:
        #     best_val_accuracy = val_accuracy
        #     torch.save(model.state_dict(), "best_model.pth")
        #     patience_counter = 0
        # else:
        #     patience_counter += 1
        
        # if patience_counter >= PATIENCE:
        #     print("Early stopping triggered, stopping training.")
        #     break
        
        # 学习率调度器
        scheduler.step()

    wandb.finish()

def evaluate(model, val_loader, device):
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader = load_data()
    model = initialize_model()
    model = model.to(device)
    train(model, train_loader, val_loader, device)
    predict(model, TEST_FILE, "predictions.txt", device)
    print("Prediction completed and saved to 'predictions.txt'")

if __name__ == "__main__":
    main()