## 1、tensor
https://docs.pytorch.org/tutorials/beginner/basics/tensorqs_tutorial.html

## 2、Quickstart
torch.utils.data.Dataset:stores the samples and their corresponding labels  

torch.utils.data.DataLoader:wraps an iterable around the Dataset

full datasets list:  
https://docs.pytorch.org/vision/stable/datasets.html

## 3、Datasets
https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial.html  

构建自己的dataset  

    import os
    import pandas as pd
    from torchvision.io import decode_image

    class CustomImageDataset(Dataset):
        def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
            self.img_labels = pd.read_csv(annotations_file)
            self.img_dir = img_dir
            self.transform = transform
            self.target_transform = target_transform

        def __len__(self):
            return len(self.img_labels)

        def __getitem__(self, idx):
            img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
            image = decode_image(img_path)
            label = self.img_labels.iloc[idx, 1]
            if self.transform:
                image = self.transform(image)
            if self.target_transform:
                label = self.target_transform(label)
            return image, label
其中，
__init__ 函数在实例化 Dataset 对象时运行一次。我们会对包含图像的目录、标注文件以及两种transform（下一节将详细介绍）进行初始化。   

The __init__ function is run once when instantiating the Dataset object. We initialize the directory containing the images, the annotations file, and both transforms (covered in more detail in the next section).  

__len__ 函数返回我们数据集中的样本数量。  

__getitem__ 函数会加载并返回数据集中给定索引 `idx` 处的一个样本。它根据该索引确定image在磁盘上的位置，使用 `decode_image` 将其转换为张量，从 `self.img_labels` 的 csv 数据中获取对应的标签，对它们调用变换函数（如果适用），并以元组的形式返回张量图像和对应的标签。  

The __getitem__ function loads and returns a sample from the dataset at the given index idx. Based on the index, it identifies the image’s location on disk, converts that to a tensor using decode_image, retrieves the corresponding label from the csv data in self.img_labels, calls the transform functions on them (if applicable), and returns the tensor image and corresponding label in a tuple.  

---
We have loaded that dataset into the DataLoader and can iterate through the dataset as needed. Each iteration below returns a batch of train_features and train_labels (containing batch_size=64 features and labels respectively). Because we specified shuffle=True, after we iterate over all batches the data is shuffled (for finer-grained control over the data loading order, take a look at Samplers). 

Datasets可以一次提取一个样本的特征和标签。在实际训练模型时，需要以“小批量”的形式传递样本，在每个 epoch（轮次）重新打乱数据以减少模型过拟合，并使用 Python 的multiprocessing来加快数据提取速度。简而言之：datasets仅支持单条查数，难以支持批量数据及同步的数据处理。

## transform
https://docs.pytorch.org/tutorials/beginner/basics/transforms_tutorial.html  

Data does not always come in its final processed form that is required for training machine learning algorithms. We use transforms to perform some manipulation of the data and make it suitable for training.

数据并不总是以训练机器学习算法所需的最终处理形式存在。我们使用变换对数据进行一些处理，使其适合训练。

All TorchVision datasets have two parameters -transform to modify the features and target_transform to modify the labels - that accept callables containing the transformation logic. The torchvision.transforms module offers several commonly-used transforms out of the box.

所有 TorchVision 数据集都有两个参数——`transform` 用于修改特征，`target_transform` 用于修改标签——这两个参数接收包含变换逻辑的可调用对象。`torchvision.transforms` 模块提供了几个常用的现成变换。

The FashionMNIST features are in PIL Image format, and the labels are integers. For training, we need the features as normalized tensors, and the labels as one-hot encoded tensors. To make these transformations, we use ToTensor and Lambda.

FashionMNIST的特征以PIL图像格式呈现，标签则是整数。在训练时，我们需要将特征转换为归一化的张量，将标签转换为独热编码的张量。为了实现这些转换，我们会使用ToTensor和Lambda。

    import torch
    from torchvision import datasets
    from torchvision.transforms import ToTensor, Lambda

    ds = datasets.FashionMNIST(
        root="data",
        train=True,
        download=True,
        transform=ToTensor(),
        target_transform=Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1))
    )

Lambda Transforms  
Lambda transforms apply any user-defined lambda function. Here, we define a function to turn the integer into a one-hot encoded tensor. It first creates a zero tensor of size 10 (the number of labels in our dataset) and calls scatter_ which assigns a value=1 on the index as given by the label y.

Lambda 变换可应用任何用户定义的 lambda 函数。在这里，我们定义了一个函数，用于将整数转换为独热编码的张量。该函数首先创建一个大小为 10（我们数据集中的标签数量）的零张量，然后调用 scatter_ 方法，在由标签 y 给定的索引位置赋值 1。

