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
import matplotlib.pyplot as plt

# 参数设置
BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 1e-5
MAX_LENGTH = 128
NUM_CLASSES = 3

# 数据路径
# DATA_DIR = "/kaggle/working/dataset"
DATA_DIR = "E:/Learning_material/junior/AI/5_FinalLab/dataset"
TRAIN_FILE = os.path.join(DATA_DIR, "train.txt")
TEST_FILE = os.path.join(DATA_DIR, "test_without_label.txt")
IMG_DIR = os.path.join(DATA_DIR, "data")

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

def initialize_model(model_type="multimodal"):
    """
    初始化不同的模型
    :param model_type: "text", "image", "multimodal"
    """
    text_model = BertModel.from_pretrained("bert-base-uncased")
    img_model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
    img_model.fc = torch.nn.Identity()
    
    if model_type == "text":
        model = MultimodalModel(text_model, img_model, NUM_CLASSES)
        model.img_model = torch.nn.Identity()  # 忽略图像部分
    elif model_type == "image":
        model = MultimodalModel(text_model, img_model, NUM_CLASSES)
        model.text_model = torch.nn.Identity()  # 忽略文本部分
    else:
        model = MultimodalModel(text_model, img_model, NUM_CLASSES)  # 多模态模型
    
    return model

def train(model, train_loader, val_loader, device):
    model = model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    train_losses = []
    train_accuracies = []
    val_accuracies = []
    
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
        
        val_accuracy = evaluate(model, val_loader, device)
        val_accuracies.append(val_accuracy)
        print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {avg_train_loss}, Accuracy: {avg_train_accuracy}%, Validation Accuracy: {val_accuracy}%")
    
    return train_losses, train_accuracies, val_accuracies

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

def plot_comparison_graph(text_losses, text_accuracies, val_accuracies, image_losses, image_accuracies, multimodal_losses, multimodal_accuracies):
    plt.figure(figsize=(15, 5))
    
    # Loss graph
    plt.subplot(1, 2, 1)
    plt.plot(text_losses, label="Text Model Loss")
    plt.plot(image_losses, label="Image Model Loss")
    plt.plot(multimodal_losses, label="Multimodal Model Loss")
    plt.title("Training Loss Comparison")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    
    # Accuracy graph
    plt.subplot(1, 2, 2)
    plt.plot(text_accuracies, label="Text Model Accuracy")
    plt.plot(image_accuracies, label="Image Model Accuracy")
    plt.plot(multimodal_accuracies, label="Multimodal Model Accuracy")
    plt.title("Training Accuracy Comparison")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('model_comparison.png')  # Save the figure as a PNG file
    print("Comparison plot saved as 'model_comparison.png'")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader = load_data()
    
    # Train text model
    text_model = initialize_model(model_type="text")
    text_model = text_model.to(device)
    text_losses, text_accuracies, val_accuracies = train(text_model, train_loader, val_loader, device)
    
    # Train image model
    image_model = initialize_model(model_type="image")
    image_model = image_model.to(device)
    image_losses, image_accuracies, _ = train(image_model, train_loader, val_loader, device)
    
    # Train multimodal model
    multimodal_model = initialize_model(model_type="multimodal")
    multimodal_model = multimodal_model.to(device)
    multimodal_losses, multimodal_accuracies, _ = train(multimodal_model, train_loader, val_loader, device)
    
    # Plot comparison graph
    plot_comparison_graph(text_losses, text_accuracies, val_accuracies, image_losses, image_accuracies, multimodal_losses, multimodal_accuracies)

if __name__ == "__main__":
    main()