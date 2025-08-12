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

具体有多少transform，后续实践的时候再看。

## Build the Neural Network
https://docs.pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html

### Define the Class

We define our neural network by subclassing nn.Module, and initialize the neural network layers in \_\_init__. Every nn.Module subclass implements the operations on input data in the forward method

使用nn.Module定义一个神经网络，在__init__函数中初始化网络层。每个 nn.Module subclass 在forward方法中实现输入数据的操作


    class NeuralNetwork(nn.Module):
        def __init__(self):
            super().__init__()
            self.flatten = nn.Flatten()
            self.linear_relu_stack = nn.Sequential(
                nn.Linear(28*28, 512),
                nn.ReLU(),
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Linear(512, 10),
            )

    def forward(self, x):
            x = self.flatten(x)
            logits = self.linear_relu_stack(x)
            return logits

We create an instance of NeuralNetwork, and move it to the device, and print its structure.

    model = NeuralNetwork().to(device)
    print(model)

to use the model, we pass it the input data. This executes the model’s forward, along with some background operations. Do not call model.forward() directly!

使用model(),而不是model.forward()函数的用法

Calling the model on the input returns a 2-dimensional tensor with dim=0 corresponding to each output of 10 raw predicted values for each class, and dim=1 corresponding to the individual values of each output. We get the prediction probabilities by passing it through an instance of the nn.Softmax module.

在输入上调用模型会返回一个二维张量，其中第 0 维对应每个输出（即每个类别的 10 个原始预测值），第 1 维对应每个输出的具体数值。
简单来说，假设输入是一批张图片，模型输出的张量形状可能是 (1, 10)—— 其中第一个维度（dim=0）的大小为 1（表示 1 个样本），第二个维度（dim=1）的大小为 10（对应 10 个类别的原始预测分数，未经过 softmax 归一化）。
如果输入是一个包含 32 张图片的批量数据，输出张量的形状会是 (32, 10)——dim=0 的 32 对应 32 个样本，dim=1 的 10 对应每个样本的 10 个类别预测值。

    X = torch.rand(1, 28, 28, device=device)
    logits = model(X)
    pred_probab = nn.Softmax(dim=1)(logits)
    y_pred = pred_probab.argmax(1)
    print(f"Predicted class: {y_pred}")


### model layers  

#### nn.Flatten

We initialize the nn.Flatten layer to convert each 2D 28x28 image into a contiguous array of 784 pixel values ( the minibatch dimension (at dim=0) is maintained).

我们初始化 `nn.Flatten` 层，用于将每张 2D 28x28 图像转换为一个包含 784 个像素值的连续数组（小批量维度（在 dim=0 处）会被保留）。

简单来说，`nn.Flatten` 的作用是"展平"图像数据：
- 输入是形状为 `(批量大小, 1, 28, 28)` 的张量（假设单通道灰度图）
- 经过 `nn.Flatten` 后，输出变为 `(批量大小, 784)` 的张量（28×28=784）

这样处理后，图像的空间结构被展开为一维向量，便于后续全连接层进行处理。而批量维度（记录样本数量的维度）会被保留，确保小批量中的每个样本都能正确对应到其展平后的特征向量。

#### nn.Linear
The linear layer is a module that applies a linear transformation on the input using its stored weights and biases.

线性层是一个模块，它利用自身存储的权重和偏置对输入执行线性变换。

具体来说，线性层会对输入张量进行如下运算：
输出 = 输入 × 权重 + 偏置

其中，权重（weight）和偏置（bias）是线性层的可学习参数，会在训练过程中不断更新。假设输入的特征维度为 in_features，输出的特征维度为 out_features，那么权重的形状为 (out_features, in_features)，偏置的形状为 (out_features,)。

#### nn.ReLU

Non-linear activations are what create the complex mappings between the model’s inputs and outputs. They are applied after linear transformations to introduce nonlinearity, helping neural networks learn a wide variety of phenomena.

In this model, we use nn.ReLU between our linear layers, but there’s other activations to introduce non-linearity in your model.

非线性激活函数用于在模型的输入和输出之间建立复杂的映射关系。它们在 linear 变换之后应用，以引入非线性特性，帮助神经网络学习各种复杂的模式。

在这个模型中，我们在线性层之间使用了 `nn.ReLU` 激活函数，但还有其他激活函数也可以为模型引入非线性，例如 `nn.Sigmoid`、`nn.Tanh`、`nn.LeakyReLU` 等。这些非线性激活函数是神经网络能够拟合复杂数据分布的关键——如果只有线性变换，无论网络有多少层，整体仍然是线性映射，无法捕捉数据中的非线性关系。

#### nn.Sequential

nn.Sequential is an ordered container of modules. The data is passed through all the modules in the same order as defined. You can use sequential containers to put together a quick network like seq_modules.

nn.Sequential 是一个有序的模块容器。数据会按照模块定义的顺序依次通过所有模块。你可以使用这种顺序容器来快速组合出一个网络，例如 seq_modules 这样的网络结构。

简单来说，nn.Sequential 就像一个 "流水线"，它将多个神经网络层按顺序打包：

输入数据会先经过第一个模块处理
处理结果自动传入第二个模块
以此类推，直到通过最后一个模块并输出结果

#### nn.Softmax

The last linear layer of the neural network returns logits - raw values in [-infty, infty] - which are passed to the nn.Softmax module. The logits are scaled to values [0, 1] representing the model’s predicted probabilities for each class. dim parameter indicates the dimension along which the values must sum to 1.

神经网络的最后一个线性层会返回 logits（即取值范围在 [-∞, ∞] 之间的原始值），这些值会被传入 `nn.Softmax` 模块。经过处理后，logits 会被缩放至 [0, 1] 区间内，代表模型对每个类别的预测概率。`dim` 参数指定了沿哪个维度进行归一化，使得该维度上的所有值之和为 1。

例如，对于形状为 `(批量大小, 10)` 的输出（10 个类别），设置 `dim=1` 意味着对每个样本（批量中的每个元素）的 10 个预测值进行归一化，确保每个样本对应的 10 个概率值之和为 1，从而符合概率分布的特性。

### Model Parameters

Many layers inside a neural network are parameterized, i.e. have associated weights and biases that are optimized during training. Subclassing nn.Module automatically tracks all fields defined inside your model object, and makes all parameters accessible using your model’s parameters() or named_parameters() methods.

神经网络中的许多层都是可参数化的，也就是说，它们具有相关的权重和偏置，这些参数会在训练过程中被优化。通过继承 `nn.Module`，可以自动跟踪模型对象内部定义的所有字段，并使所有参数可通过模型的 `parameters()` 或 `named_parameters()` 方法访问。

这一特性非常实用，例如：
- 调用 `model.parameters()` 可以获取模型中所有可学习的参数（权重和偏置），方便传入优化器进行更新
- 使用 `named_parameters()` 可以同时获取参数的名称和对应的张量，便于调试或针对性地调整某些层的参数

这种自动参数跟踪机制简化了神经网络的实现流程，让开发者无需手动管理参数列表，专注于模型结构设计即可。

    print(f"Model structure: {model}\n\n")

    for name, param in model.named_parameters():
        print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")