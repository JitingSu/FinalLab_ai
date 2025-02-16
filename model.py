import torch
from torch.utils.data import Dataset
from torch import nn
from PIL import Image
import torch.nn.functional as F

class MultimodalDataset(Dataset):
    def __init__(self, data, img_dir, tokenizer, transform, max_length, data_dir):
        """
        初始化MultimodalDataset类的实例。

        :param data: 包含数据的列表，每个元素是一个字符串，表示一行数据，格式为 "guid,label"。
        :param img_dir: 图像文件所在的目录路径。
        :param tokenizer: 用于处理文本数据的tokenizer对象。
        :param transform: 用于处理图像数据的transform对象。
        :param max_length: 文本数据的最大长度，用于tokenizer的padding。
        :param data_dir: 文本文件所在的目录路径。
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
            guid, label = line.strip().split(",")
        except ValueError:
            print(f"Skipping invalid line: {line}")
            return None

        # 处理文本数据
        try:
            # 使用 ISO-8859-1 编码
            with open(
                f"{self.data_dir}/{guid}.txt", "r", encoding="ISO-8859-1"
            ) as file:
                text = file.read()
        except FileNotFoundError:
            print(f"Text file {guid}.txt not found. Skipping this sample.")
            return None

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)

        # 处理图像数据
        try:
            img = Image.open(f"{self.img_dir}/{guid}.jpg")
            img = self.transform(img)
        except FileNotFoundError:
            print(f"Image file {guid}.jpg not found. Skipping this sample.")
            return None

        # 将标签转换为数字
        label_map = {"positive": 0, "neutral": 1, "negative": 2}
        if label not in label_map:
            print(f"Invalid label {label} for guid {guid}. Skipping this sample.")
            return None

        label = label_map[label]

        return input_ids, attention_mask, img, torch.tensor(label)


class MultimodalModel(nn.Module):
    def __init__(self, text_model, img_model, num_classes, dropout_rate=0.5):
        """
        初始化多模态融合模型，添加Dropout来防止过拟合
        :param text_model: 文本模型（BERT）
        :param img_model: 图像模型（ResNet）
        :param num_classes: 输出类别数（对于三分类任务：positive, neutral, negative）
        :param dropout_rate: Dropout的丢弃率，默认为0.5
        """
        super(MultimodalModel, self).__init__()
        self.text_model = text_model
        self.img_model = img_model

        # 文本模型部分
        self.text_fc = nn.Linear(768, 256)  # BERT的输出是768维，做一个映射到256维
        self.text_dropout = nn.Dropout(dropout_rate)  # 添加Dropout层

        # 图像模型部分
        self.img_fc = nn.Linear(2048, 256)  # ResNet50的输出是2048维，做一个映射到256维
        self.img_dropout = nn.Dropout(dropout_rate)  # 添加Dropout层

        # 融合后的全连接层
        self.fc = nn.Linear(
            256 * 2, num_classes
        )  # 文本和图像特征拼接后是512维，映射到类别数

    def forward(self, input_ids, attention_mask, img):
        # 文本部分
        text_output = self.text_model(input_ids, attention_mask=attention_mask)
        text_features = text_output.last_hidden_state.mean(dim=1)  # 计算平均值
        text_features = self.text_fc(text_features)  # 通过全连接层
        text_features = self.text_dropout(text_features)  # 应用Dropout

        # 图像部分
        img_features = self.img_model(img)  # 通过ResNet50提取图像特征
        img_features = self.img_fc(img_features)  # 通过全连接层
        img_features = self.img_dropout(img_features)  # 应用Dropout

        # 融合文本和图像特征
        combined_features = torch.cat((text_features, img_features), dim=1)

        # 分类部分
        output = self.fc(combined_features)
        return output


class MultimodalModelvs(nn.Module):
    def __init__(self, text_model, img_model, num_classes):
        """
        初始化多模态融合模型，支持不同的输入组合。

        :param text_model: 文本模型（BERT）
        :param img_model: 图像模型（ResNet）
        :param num_classes: 输出类别数（对于三分类任务：positive, neutral, negative）
        """
        super(MultimodalModelvs, self).__init__()
        self.text_model = text_model  # 文本模型（BERT）
        self.img_model = img_model  # 图像模型（ResNet）

        # 定义多个 fc 层以支持不同的输入组合
        self.fc_text_only = nn.Linear(
            768, num_classes
        )  # 仅文本输入时使用的全连接层，输入维度为768（BERT输出维度），输出维度为类别数
        self.fc_img_only = nn.Linear(
            2048, num_classes
        )  # 仅图像输入时使用的全连接层，输入维度为2048（ResNet输出维度），输出维度为类别数
        self.fc_combined = nn.Linear(
            768 + 2048, num_classes
        )  # 文本和图像同时输入时使用的全连接层，输入维度为768+2048（BERT和ResNet输出维度之和），输出维度为类别数

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        img=None,
        use_text=True,
        use_image=True,
    ):
        text_features = None
        img_features = None

        # 仅使用文本数据
        if use_text and input_ids is not None and attention_mask is not None:
            text_output = self.text_model(
                input_ids=input_ids, attention_mask=attention_mask
            )
            text_features = text_output.last_hidden_state.mean(
                dim=1
            )  # 使用池化的文本特征

        # 仅使用图像数据
        if use_image and img is not None:
            img_features = self.img_model(img)  # 获取图像特征

        # 合并文本特征和图像特征
        if text_features is not None and img_features is not None:
            combined_features = torch.cat((text_features, img_features), dim=1)  # 拼接
            output = self.fc_combined(combined_features)  # 使用拼接后的特征
        elif text_features is not None:
            output = self.fc_text_only(text_features)  # 仅使用文本特征
        elif img_features is not None:
            output = self.fc_img_only(img_features)  # 仅使用图像特征
        else:
            raise ValueError("Both input modalities are None!")

        return output

class MultimodalModelef(nn.Module):
    def __init__(self, text_model, img_model, num_classes):
        """
        初始化多模态融合模型。

        :param text_model: 文本模型（BERT）
        :param img_model: 图像模型（ConvNeXt-Base）
        :param num_classes: 输出类别数（对于三分类任务：positive, neutral, negative）
        """
        super(MultimodalModelef, self).__init__()
        self.text_model = text_model
        self.img_model = img_model

        # 文本模型部分
        self.text_fc = nn.Linear(768, 256)  # BERT 的输出是 768 维，做一个映射到 256 维

        # 图像模型部分
        self.img_fc = nn.Linear(
            1024, 256
        )  # ConvNeXt-Base 的输出是 1024 维，做一个映射到 256 维

        # 融合后的全连接层
        self.fc = nn.Linear(
            256 * 2, num_classes
        )  # 文本和图像特征拼接后是 512 维，映射到类别数

    def forward(self, input_ids, attention_mask, img):
        # 文本部分
        text_output = self.text_model(input_ids, attention_mask=attention_mask)
        text_features = text_output.last_hidden_state.mean(dim=1)  # 平均池化
        text_features = self.text_fc(text_features)

        # 图像部分
        img_features = self.img_model(
            img
        )  # ConvNeXt 的输出形状是 (batch_size, 1024, H, W)
        img_features = F.adaptive_avg_pool2d(img_features, (1, 1))  # 全局平均池化
        img_features = img_features.view(
            img_features.size(0), -1
        )  # 展平为 (batch_size, 1024)
        img_features = self.img_fc(img_features)

        # 融合文本和图像特征
        combined_features = torch.cat((text_features, img_features), dim=1)

        # 分类部分
        output = self.fc(combined_features)
        return output