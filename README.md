# FinalLab_ai

当代人工智能大作业

## Setup

此实验实现基于Python3.11.3。要运行代码，还需要安装以下库：

- torch==2.1.0
- tqdm==4.66.1
- transformers==4.30.0
- torchvision==0.15.1
- scikit-learn==1.3.0
- Pillow==10.0.0
- wandb==0.15.0

也可以运行以下命令进行依赖安装：

```python
pip install -r requirements.txt
```

## Repository structure
本次实验文件夹结构组成如下：

```python
|-- result.txt    		# 预测文件
|-- main.py  		    # 主程序入口
|-- model.py  		    # 模型构建与数据预处理模块
|-- draft.ipynb			# 在kaggle运行代码的notebook过程
|-- requirements.txt
|-- README.md  
|-- best_model.pth		# 训练保存的最佳模型
```

## Run 
1. 确保本地保存有本次实验的数据集，数据集结构如下：

```python
|-- train.txt    		  # 数据的guid和对应的情感标签
|-- test_without_label.py # 数据的guid和空的情感标签
|-- data   			      # 包括所有的训练文本和图片
    |-- 1.jpg    		
    |-- 1.txt
    |-- ...				  # 以此类推

```

2. 在 `main.py` 中的第 22 行，将 `DATA_DIR` 变量中的路径修改为自己电脑中的数据集路径：

```python
DATA_DIR = "/kaggle/working/dataset"
```

3. 在终端运行以下命令进行模型的训练和测试，建议在 GPU 环境下运行，通常需要大约 10 分钟时间，若使用 CPU，则可能需要3小时的时间：

```powershell
python main.py
```

4. 如果想跳过训练步骤，直接使用已经训练好的模型 `best_model.pth` 进行测试，可以将主函数中的这一行代码注释掉：

```python
train(model, train_loader, val_loader, device)

# 以下代码已包含在predict函数中
# model.load_state_dict(torch.load("best_model.pth"))
# model.eval()
```


## Reference

Parts of this code are based on the following repositories:

- [Zhihu](https://zhuanlan.zhihu.com/p/402997033)

- [Paper](https://ieeexplore.ieee.org/abstract/document/9736584)

- [GloGNN](https://github.com/RecklessRonan/GloGNN/blob/master/readme.md)
