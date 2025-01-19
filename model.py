import torch
from torch.utils.data import Dataset
from transformers import BertModel, BertTokenizer
from torchvision import models
from torch import nn
from PIL import Image
from torchvision.transforms import transforms

class MultimodalDataset(Dataset):
    def __init__(self, data, img_dir, tokenizer, transform, max_length, data_dir):
        """
        初始化数据集类
        :param data: 包含文本和标签的列表，格式为 ["guid1,label1", "guid2,label2", ...]
        :param img_dir: 图片存放的文件夹路径
        :param tokenizer: BERT分词器
        :param transform: 图像的预处理变换
        :param max_length: 文本最大长度
        :param data_dir: 文本文件所在的目录
        """
        self.data = data
        self.img_dir = img_dir
        self.tokenizer = tokenizer
        self.transform = transform
        self.max_length = max_length
        self.data_dir = data_dir 

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        line = self.data[idx]
        
        try:
            # 尝试拆分每行数据（用逗号分隔）
            guid, label = line.strip().split(",")
        except ValueError:
            # 如果拆分失败，输出警告并跳过该行
            print(f"Skipping invalid line: {line}")
            return None 

        # 处理文本数据
        try:
            # 使用 ISO-8859-1 编码
            with open(f"{self.data_dir}/{guid}.txt", "r", encoding="ISO-8859-1") as file:
                text = file.read()
        except FileNotFoundError:
            print(f"Text file {guid}.txt not found. Skipping this sample.")
            return None  

        encoding = self.tokenizer(
            text, truncation=True, padding="max_length", max_length=self.max_length, return_tensors="pt"
        )
        input_ids = encoding['input_ids'].squeeze(0)  # 去掉多余的batch维度
        attention_mask = encoding['attention_mask'].squeeze(0)

        # 处理图像数据
        try:
            img = Image.open(f"{self.img_dir}/{guid}.jpg")
            img = self.transform(img)  
        except FileNotFoundError:
            print(f"Image file {guid}.jpg not found. Skipping this sample.")
            return None  

        # 将标签转换为数字
        label_map = {'positive': 0, 'neutral': 1, 'negative': 2}
        if label not in label_map:
            print(f"Invalid label {label} for guid {guid}. Skipping this sample.")
            return None  

        label = label_map[label]

        return input_ids, attention_mask, img, torch.tensor(label)


# 多模态融合模型: MultimodalModel
class MultimodalModel(nn.Module):
    def __init__(self, text_model, img_model, num_classes):
        """
        初始化多模态融合模型
        :param text_model: 文本模型（如BERT）
        :param img_model: 图像模型（如ResNet）
        :param num_classes: 输出类别数（对于三分类任务：positive, neutral, negative）
        """
        super(MultimodalModel, self).__init__()
        self.text_model = text_model
        self.img_model = img_model
        
        # 文本模型部分
        self.text_fc = nn.Linear(768, 256)  # BERT的输出是768维，做一个映射到256维
        
        # 图像模型部分
        self.img_fc = nn.Linear(2048, 256)  # ResNet50的输出是2048维，做一个映射到256维
        
        # 融合后的全连接层
        self.fc = nn.Linear(256 * 2, num_classes)  # 文本和图像特征拼接后是512维，映射到类别数

    def forward(self, input_ids, attention_mask, img):
        # 文本部分
        text_output = self.text_model(input_ids, attention_mask=attention_mask)
        text_features = text_output.pooler_output  # 获取BERT的池化层输出
        text_features = self.text_fc(text_features)  # 通过全连接层
        
        # 图像部分
        img_features = self.img_model(img)  # 通过ResNet50提取图像特征
        img_features = self.img_fc(img_features)  # 通过全连接层

        # 融合文本和图像特征
        combined_features = torch.cat((text_features, img_features), dim=1)

        # 分类部分
        output = self.fc(combined_features)
        return output

# # 预训练模型加载函数
# from torchvision.models import ResNet50_Weights

# def get_model(num_classes):
#     # 加载预训练的BERT模型
#     text_model = BertModel.from_pretrained("bert-base-uncased")
    
#     # 加载预训练的ResNet50模型（使用weights代替pretrained）
#     img_model = models.resnet50(weights=ResNet50_Weights.DEFAULT)  # 或使用 ResNet50_Weights.IMAGENET1K_V1
    
#     img_model.fc = nn.Identity()  # 去掉ResNet最后的分类层
    
#     # 创建多模态模型
#     model = MultimodalModel(text_model, img_model, num_classes)
#     return model

# def get_model(num_classes):
#     # 加载预训练的BERT模型
#     text_model = BertModel.from_pretrained("bert-base-uncased")
    
#     # 加载预训练的ResNet50模型
#     img_model = models.resnet50(pretrained=True)
#     img_model.fc = nn.Identity()  # 去掉ResNet最后的分类层
    
#     # 创建多模态模型
#     model = MultimodalModel(text_model, img_model, num_classes)
#     return model