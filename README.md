# C3D_model  
Implementation of C3D model based on UCF101 dataset  
这个项目是基于C3D模型和UCF101数据集的行为识别项目，15个epoch识别准确度可以达到95%左右  
## 开发环境：  
Jupyter notebook:6.5.6   
python:3.9.18  
pytorch:2.1.2  
CUDA:按自己的电脑配置安装（<font color="red">强烈建议安装CUDA使用GPU训练否则训练时间会很久，本人MacBookAir m1芯片GPU不支持3D卷积，用CPU训练15个epoch花费了30个小时</font>）
### 额外需要的库：  
opencv-python:4.5.4.60  
tensorboardX:2.5.1  
tensorboard:2.7.0  
tqdm:4.62.3
## 项目开发前的准备
### (1).先简单介绍一下项目结构 
<img src="[项目结构图.jpeg](https://github.com/xjx12/C3D_model/blob/3374bf42b4e579f312491a990c4e30a714a94285/%E9%A1%B9%E7%9B%AE%E7%BB%93%E6%9E%84%E5%9B%BE.jpeg)">
<p style="text-indent: 2em;">如项目结构图所示，接下来我简单介绍一下以避免在项目运行时因为路径的原因报错：</p>
<p style="text-indent: 0em;">(1).C3D(根目录):存放项目文件的根目录</p>
<p style="text-indent: 2em;">(2).data(数据集文件夹)：用于存放数据集</p>
<p style="text-indent: 4em;">(3).UCF101(原始数据集)：原始视频数据集，需要自行下载，下载地址：https://www.crcv.ucf.edu/research/data-sets/ucf101/</p>
        <p style="text-indent: 4em;">(3).ucf101(图片数据集文件)：用于存放视频数据集处理后的图片数据集。(<font color="red">该文件夹需要自己先创建</font>)</p>
<p style="text-indent: 6em;">(4).train(训练集文件夹)：用于存放训练数据(运行data_process.ipynb会自动生成,如果生成失败请自己创建)</p>
<p style="text-indent: 6em;">(4).test(测试集文件夹)：用于存放测试数据(运行data_process.ipynb会自动生成,如果生成失败请自己创建)</p>
<p style="text-indent: 6em;">(4).val(验证集文件夹)：用于存放验证数据(运行data_process.ipynb会自动生成,如果生成失败请自己创建)</p>
<p style="text-indent: 4em;">(3).labels.txt(标签文件)：用于存放视频对应的标签（运行data_process.ipynb会自动生成）</p>
<p style="text-indent: 4em;">(3).testvideo.avi(推理视频)：用于模型推理，也可以直接用原始数据集中的视频，需要注意模型推理时的推理视频路径</p>
<p style="text-indent: 2em;">(2).model_resule(日志及训练模型保存文件夹)：用于存放训练日志以及保存训练好的模型以便在模型推测加载训练好的模型(<font color="red">该文件夹需要自己先创建</font>)</p>
<p style="text-indent: 4em;">(3).models(日志及训练模型保存文件夹)：用于存放训练日志以及保存训练好的模型以便在模型推测加载训练好的模型(运行train.ipynb文件自动生成)</p>
<p style="text-indent: 2em;">(2).c3d-pretrained.pth(预训练权重文件)：用于模型训练时预加载的权重，可以让模型更快收敛以及防止过拟合。该文件需要自己下载，注意文件名的一致。</p>
<p style="text-indent: 2em;">(2).C3D_model.py(C3D模型文件)：C3D模型。</p>
<p style="text-indent: 2em;">(2).data_process.ipynb(数据预处理文件)：生成标签文件以及将原始数据集UCF101划分为train(64%)、val(16%)和test(20%)后将视频文件变成一帧一帧的图片文件方便后续模型训练</p>
<p style="text-indent: 2em;">(2).dataset.py(数据集加载文件)：用于训练模型时加载图片数据</p>
<p style="text-indent: 2em;">(2).train.ipynb(模型训练文件)：用于模型的训练</p>
<p style="text-indent: 2em;">(2).inference.ipynb(模型推理文件)：用于模型推理</p>

### (2)、下载需要的文件  
由于数据集和预训练权重文件太大了上传不了，所以这两个文件需要自己下载  
UCF101数据集链接:https://www.crcv.ucf.edu/research/data-sets/ucf101/  
预训练权重文件(c3d-pretrained.pth)  
百度网盘链接: https://pan.baidu.com/s/1ygA_GdfjXaPa0Mg2EHU_8Q?pwd=0238 提取码: 0238  
下载后将这两个文件放到对应位置就可以开始做项目了  

如果不想训练只做推理可以使用下面提供的模型：
链接: https://pan.baidu.com/s/1MWVp8xbcI2bxwfcgkmOXpg?pwd=0238 提取码: 0238  
<font color="red">注意:模型文件.tar不需要解压，下载后放到对应位置即可</font>
### (3)、按照项目结构图创建缺失的文件夹
在该项目中需要创建2个文件夹，一个是ucf101，另一个是model_resule

## 1、模型介绍及模型搭建
### (1)、模型介绍
<p style="text-indent: 2em;">C3D模型是一个用于3D视觉任务的模型，其网络结构类似于常见的2D卷积网络，C3D模型结构和2D卷积的VGG-11模型一模一样，但主要区别在于C3D模型使用了3D卷积操作，而非传统的2D卷积。C3D模型的详细介绍如下：</p>  

• 模型结构：C3D网络包含8个卷积层、5个最大池化层和2个全连接层。这种结构使得C3D模型能够捕获视频中的时空特征，从而在处理基于视频分析的任务时具有优势，详细结构请参考图C3D and VGG-11。  

• 特征提取：C3D模型能够提取出视频中的物体信息、场景信息和动作信息，并将这些信息隐式编码到特征中。这使得C3D模型在处理各种3D视觉任务时都能取得不错的效果，而无需针对特定任务进行微调（finetune）。  

• 输入处理：在训练过程中，所有的视频都被分割成clips，帧被resize为(128,171)，输入维度为(None,nframes,128,171,3)，其中nframes表示每个clip的帧数（通常为16帧）。在训练过程中，输入数据的HW（高度和宽度）会被随机crop成(112,112)，这种操作类似于attention机制，可以增加模型的鲁棒性并提升训练效果。  

• C3D模型的应用广泛，可以处理各种基于视频的3D视觉任务，如动作识别、视频分类等。同时，C3D模型也可以与其他深度学习技术结合使用，以进一步提升性能。  

此外，需要注意的是，C3D模型中的“C3D”一词有时也用于指代其他技术或产品，如俄罗斯三维建模软件KOMPAS-3D的三维几何内核C3D。这款软件广泛应用于多个领域，包括汽车工业、重型机械、航空航天等，与C3D模型在3D视觉任务中的应用有所不同。  
### (2)、模型搭建  
模型的搭建请参考C3D_model.py

## 2、数据集预处理  
### (1)、data_process.ipynb
<p style="text-indent: 2em;">这个ipynb脚本中的主要作用是对视频数据进行预处理，将其分割成帧，并按照一定的比例划分为训练集、验证集和测试集并且自动生成对应的文件夹，同时生成相应的标签文件labels.txt。以下是详细的功能说明：<p>

#### 函数 `process_video`
这个函数负责将单个视频文件处理成帧并保存：
1. **读取视频**：从指定路径读取视频文件。
2. **视频帧采样**：以特定的频率（`EXTRACT_FREQUENCY`）采样视频帧，确保每个视频有足够的帧用于训练。
3. **帧大小调整**：调整帧的大小为128x171，然后裁剪到112x112。
4. **保存帧**：将处理后的帧保存为图像文件。

#### 函数 `preprocess`
这个函数负责将整个数据集按照一定比例划分为训练集、验证集和测试集，并调用 `process_video` 对每个视频进行处理：
1. **创建输出目录**：创建存放训练集、验证集和测试集的目录。
2. **划分数据集**：使用 `train_test_split` 将每个类别的视频文件划分为训练集、验证集和测试集。
3. **处理视频**：对每个视频文件调用 `process_video` 函数，处理视频并保存帧。

#### 函数 `label_text_write`
这个函数负责生成标签文件 `labels.txt`：
1. **遍历类别文件夹**：遍历原始数据路径下的每个类别文件夹，收集视频文件的路径和类别标签。
2. **生成标签索引**：为每个类别分配一个唯一的索引。
3. **写入标签文件**：将类别标签和索引写入 `labels.txt` 文件。

#### 主程序 `if __name__ == "__main__"`
在主程序中，调用上述函数执行预处理步骤：
1. **生成标签文件**：调用 `label_text_write` 生成 `labels.txt` 文件。
2. **划分数据集**：调用 `preprocess` 将原始数据集划分为训练集、验证集和测试集，并生成相应的图片数据集。
    
### (2)、dataset.py
<p style="text-indent: 2em;">这个python脚本的作用是定义一个用于加载和处理视频数据的自定义数据集类 VideoDataset，并创建数据加载器（DataLoader）用于训练、验证和测试。以下是详细的功能说明：<p>

#### 类 `VideoDataset`
`VideoDataset` 类继承自 `torch.utils.data.Dataset`，用于加载和处理视频数据，生成训练、验证和测试集的数据。

##### 初始化方法 `__init__`
```python
def __init__(self, dataset_path, images_path, clip_len):
    ...
```
- **`dataset_path`**: 数据集的根路径。
- **`images_path`**: 数据集的子目录名称，例如 `train`、`val` 或 `test`。
- **`clip_len`**: 每个视频片段的帧数长度。

在初始化方法中，代码执行以下操作：
1. 设置视频帧的尺寸和裁剪后的尺寸。
2. 遍历指定目录下的所有类别文件夹，收集所有视频帧的路径，并生成对应的标签列表。
3. 创建标签到索引的映射，将标签转换为整数索引。

##### 方法 `__len__`
```python
def __len__(self):
    return len(self.fnames)
```
返回数据集中的视频片段数量。

##### 方法 `__getitem__`
```python
def __getitem__(self, index):
    ...
```
获取指定索引的视频片段，进行以下操作：
1. **加载帧**: 调用 `load_frames` 方法加载指定视频片段的帧。
2. **裁剪帧**: 调用 `crop` 方法对帧进行随机裁剪。
3. **归一化**: 调用 `normalize` 方法对帧进行归一化处理。
4. **转换为张量**: 调用 `to_tensor` 方法将帧转换为 PyTorch 张量。
5. 返回视频片段的张量和对应的标签。

##### 方法 `load_frames`
```python
def load_frames(self, file_dir):
    ...
```
从指定目录加载视频帧，将其存储为一个 NumPy 数组。

##### 方法 `crop`
```python
def crop(self, buffer, clip_len, crop_size):
    ...
```
对视频帧进行随机裁剪，生成指定长度和尺寸的片段。

##### 方法 `normalize`
```python
def normalize(self, buffer):
    ...
```
对视频帧进行归一化处理。

##### 方法 `to_tensor`
```python
def to_tensor(self, buffer):
    ...
```
将视频帧的维度转换为 PyTorch 张量所需的格式。

### 数据加载器
在 `if __name__ == "__main__":` 代码块中，创建了三个数据加载器，分别用于训练、验证和测试集。

```python
train_data = VideoDataset(dataset_path='data/ucf101', images_path='train', clip_len=16)
train_loader = DataLoader(train_data, batch_size=64, shuffle=True, num_workers=0)

test_data = VideoDataset(dataset_path='data/ucf101', images_path='test', clip_len=16)
test_loader = DataLoader(test_data, batch_size=64, shuffle=True, num_workers=0)

val_data = VideoDataset(dataset_path='data/ucf101', images_path='val', clip_len=16)
val_loader = DataLoader(val_data, batch_size=64, shuffle=True, num_workers=0)
```
- **`dataset_path`**: 数据集的根路径。
- **`images_path`**: 指定是训练集、验证集还是测试集。
- **`clip_len`**: 每个视频片段的帧数长度。
- **`batch_size`**: 每个批次的数据量。
- **`shuffle`**: 是否打乱数据。
- **`num_workers`**: 用于加载数据的子进程数。

这些数据加载器用于将预处理好的视频数据加载到模型中进行训练、验证和测试。

## 3、模型训练
### train.ipynb
<p style="text-indent: 2em;">这个脚本的作用是定义并实现一个训练、验证和测试 C3D（3D卷积神经网络）模型的函数 train_model。该函数执行以下主要任务：<p>

### 功能说明

#### 3.1. 模型实例化和配置

```python
model = C3D_model.C3D(num_classes, pretrained=True)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
model.to(device)
criterion.to(device)
```
- **实例化 C3D 模型**：创建一个预训练的 C3D 模型实例，若要从零开始训练请将pretrained设置为False。
- **定义损失函数**：使用交叉熵损失函数。
- **配置优化器**：使用 SGD 优化器，设置学习率、动量和权重衰减。
- **设置学习率调度器**：每 5 个 epoch 将学习率减少到原来的 0.1 倍。
- **将模型和损失函数加载到指定的设备（CPU 或 GPU）**。

#### 3.2. 日志记录器配置

```python
log_dir = os.path.join(save_dir, 'models', datetime.now().strftime('%b%d_%H-%M-%S') + '_' + socket.gethostname())
writer = SummaryWriter(log_dir=log_dir)
```
- **日志记录器**：配置 TensorBoard 日志记录器，以记录训练过程中的损失和准确率。

#### 3.3. 数据加载器和数据大小

```python
trainval_loaders = {'train': train_dataloader, 'val': val_dataloader}
trainval_sizes = {x: len(trainval_loaders[x].dataset) for x in ['train', 'val']}
test_size = len(test_dataloader.dataset)
```
- **数据加载器**：定义训练、验证和测试的数据加载器。
- **计算数据集大小**：计算训练集、验证集和测试集的样本数量。

#### 3.4. 训练和验证循环

```python
for epoch in range(num_epochs):
    for phase in ['train', 'val']:
        start_time = timeit.default_timer()
        running_loss = 0.0
        running_corrects = 0.0

        if phase == 'train':
            model.train()
        else:
            model.eval()

        for inputs, labels in tqdm(trainval_loaders[phase]):
            inputs = Variable(inputs, requires_grad=True).to(device)
            labels = Variable(labels).to(device)
            optimizer.zero_grad()

            if phase == 'train':
                outputs = model(inputs)
            else:
                with torch.no_grad():
                    outputs = model(inputs)

            probs = nn.Softmax(dim=1)(outputs)
            preds = torch.max(probs, 1)[1]
            labels = labels.long()
            loss = criterion(outputs, labels)

            if phase == 'train':
                loss.backward()
                optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        scheduler.step()
        epoch_loss = running_loss / trainval_sizes[phase]
        epoch_acc = running_corrects.double() / trainval_sizes[phase]

        if phase == 'train':
            writer.add_scalar('data/train_loss_epoch', epoch_loss, epoch)
            writer.add_scalar('data/train_acc_epoch', epoch_acc, epoch)
        else:
            writer.add_scalar('data/val_loss_epoch', epoch_loss, epoch)
            writer.add_scalar('data/val_acc_epoch', epoch_acc, epoch)
        stop_time = timeit.default_timer()

        print("[{}] Epoch: {}/{} Loss: {} Acc: {}".format(phase, epoch + 1, num_epochs, epoch_loss, epoch_acc))
        print("Execution time: " + str(stop_time - start_time) + "\n")

writer.close()
```
- **训练和验证循环**：
  - 在每个 epoch 中，遍历训练和验证两个阶段。
  - 在训练阶段，模型处于训练模式；在验证阶段，模型处于评估模式。
  - 对输入数据进行前向传播，计算损失，反向传播并更新模型参数（仅在训练阶段）。
  - 记录并累加损失和正确预测的数量。
  - 每个 epoch 结束后，调整学习率并记录损失和准确率到 TensorBoard。  
    <font color="red">查看损失和准确率请在该项目环境的命令行里使用以下命令:tensorboard --logdir 日志文件夹  
        (日志文件夹路径为:C3D/model_resule/models/xxx.lan，请注意是整个文件夹而不是文件夹里的.lan文件)，将命令行给出的网站复制到浏览器打开即可看到对应的损失和准确率。</font>

#### 3.5. 模型保存

```python
torch.save({'epoch': epoch + 1, 'state_dict': model.state_dict(), 'opt_dict': optimizer.state_dict(),},
           os.path.join(save_dir, 'models', 'C3D' + '_epoch-' + str(epoch) + '.pth.tar'))
print("Save model at {}\n".format(os.path.join(save_dir, 'models', 'C3D' + '_epoch-' + str(epoch) + '.pth.tar')))
```
- **保存模型权重**：在训练完成后，将模型的状态字典和优化器的状态字典保存到指定路径，保存路径为：C3D/model_resule/models/C3D_epoch_xx.pth.tar

#### 3.6. 模型测试

```python
model.eval()
running_corrects = 0.0
for inputs, labels in tqdm(test_dataloader):
    inputs = inputs.to(device)
    labels = labels.long()
    labels = labels.to(device)

    with torch.no_grad():
        outputs = model(inputs)

    probs = nn.Softmax(dim=1)(outputs)
    preds = torch.max(probs, 1)[1]

    running_corrects += torch.sum(preds == labels.data)
epoch_acc = running_corrects.double() / test_size
print("test Acc: {}".format(epoch_acc))
```
- **模型测试**：在测试集上评估模型性能，记录并输出测试准确率。

#### 3.7.main主函数：

##### 3.7.1. 定义训练设备

```python
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
```
- **定义训练设备**：检查是否有可用的 GPU。如果有，则使用第一个 GPU（cuda:0）；否则，使用 CPU。

##### 3.7.2. 定义训练参数

```python
num_classes = 101  # 类别数
num_epochs = 15    # 训练轮次
lr = 1e-3          # 学习率
save_dir = 'model_resule'  # 保存模型结果的目录
```
- **设置模型参数**：定义类别数、训练轮次、学习率和模型保存目录。

##### 3.7.3. 数据加载

```python
train_data = VideoDataset(dataset_path='data/ucf101', images_path='train', clip_len=16)
train_dataloader = DataLoader(train_data, batch_size=8, shuffle=True, num_workers=0)

test_data = VideoDataset(dataset_path='data/ucf101', images_path='test', clip_len=16)
test_dataloader = DataLoader(test_data, batch_size=8, num_workers=0)

val_data = VideoDataset(dataset_path='data/ucf101', images_path='val', clip_len=16)
val_dataloader = DataLoader(val_data, batch_size=8, num_workers=0)
```
- **加载数据集**：
  - 使用 `VideoDataset` 类分别加载训练、测试和验证数据集，设置每个视频片段的长度为 16 帧。
  - 使用 `DataLoader` 为每个数据集创建数据加载器，设置批量大小为 8，并在训练集上启用随机打乱。

##### 3.7.4. 训练模型

```python
train_model(num_classes, train_dataloader, val_dataloader, test_dataloader, num_epochs, lr, device, save_dir)
```
- **调用 `train_model` 函数**：开始模型的训练、验证和测试流程。  

<font color="red">当出现</font>**Save model at model_resule/models/C3D_epoch-14.pth.tar**<font color="red">即代表模型训练完成并且已经保存好训练好的模型了，但是到这里脚本还并没有运行完，因为还需要在test数据集上验证测试精度以确保模型没有过拟合</font>

<font color="red">当出现</font>**test Acc: 0.9xxxxx**<font color="red"> 时即代表测试集测试完成</font>

### 至此，模型训练阶段完成

#### 3.8 `train_model` 函数回顾

该函数包括以下主要步骤：
- **模型实例化和配置**：实例化 C3D 模型，并配置损失函数、优化器和学习率调度器。
- **日志记录配置**：使用 TensorBoard 记录训练和验证过程中的损失和准确率。
- **训练和验证循环**：在每个 epoch 中，遍历训练和验证阶段，进行前向传播、反向传播和参数更新，并记录损失和准确率。
- **模型保存**：在训练完成后，保存模型的状态字典和优化器的状态字典。
- **模型测试**：在测试集上评估模型性能，输出测试准确率。

<font color="red" size="5">注意：</font>
- **<font color="red">3.7.3数据加载的路径是视频处理后的ucf101图片数据集，请注意路径</font>**  
- **<font color="red">一个batch_size是(1,3,16,112,112)，所以一个batch_size就有16张图片，请根据自己电脑的显存修改batch_size的大小</font>**
- **<font color="red">如果需要更高的准确率请将num_epochs次数增加，并且适当修改其他超参数，例如学习率lr、批量大小batch_size、动量momentum、权重衰减weight_decay和学习率调度器等</font>**

## 4、模型推理
### inference.ipynb

<p style="text-indent: 2em;">这个ipynb脚本使用预训练的 C3D 模型对视频进行动作识别的推理过程，并将预测结果显示在视频帧上。具体步骤如下：<p>

### 4.1功能说明

#### 4.1.1. 函数定义

**`center_corp(frame)`**
- 功能：对输入的视频帧进行中心裁剪。
- 实现：裁剪出帧的中心部分 `frame[8:120,30:142,:]`，返回裁剪后的图像。

**`center_crop(frame)`**
- 功能：对输入的视频帧进行中心裁剪。
- 实现：计算出裁剪后的高度和宽度（112x112），然后从帧的中心裁剪出这部分图像，返回裁剪后的图像。

#### 4.1.2. 模型推理函数

**`inference()`**
- 功能：加载预训练的 C3D 模型，对视频进行动作识别，并在视频帧上显示预测结果。

### 4.2具体步骤：

(1). **定义设备**
   ```python
   device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
   ```
   - 检查是否有可用的 GPU，如果有则使用 GPU，否则使用 CPU。

(2). **加载数据标签**
   ```python
   with open("./data/labels.txt", 'r') as f:  --->标签文件路径
       class_names = f.readlines()
   ```
   - 读取动作类别标签。

(3). **加载模型和模型参数**
   ```python
   model = C3D_model.C3D(num_classes=101)
   checkpoint = torch.load('./model_resule/models/C3D_epoch-14.pth.tar')  --->保存训练好的模型的路径
   model.load_state_dict(checkpoint['state_dict'])
   ```
   - 实例化 C3D 模型，并加载训练好的模型参数。

(4). **将模型放入设备并设置为验证模式**
   ```python
   model.to(device)
   model.eval()
   ```
   - 将模型移动到指定设备（CPU 或 GPU）并设置为评估模式。

(5). **打开视频文件并进行处理**
   ```python
   video = './data/testvideo_2.avi'  --->推理视频路径
   cap = cv2.VideoCapture(video)
   retaining = True
   ```
   - 打开视频文件，并初始化一个变量 `retaining`，用于判断是否继续读取视频帧。

(6). **处理视频帧**
   ```python
   while retaining:
       retaining, frame = cap.read()
       if not retaining or frame is None:
           continue

       tmp_ = center_crop(cv2.resize(frame, (171, 128)))
       tmp = tmp_ - np.array([[[90.0, 98.0, 102.0]]])
       clip.append(tmp)
   ```
   - 循环读取视频帧，对每个帧进行缩放、中心裁剪和归一化处理，并将处理后的帧添加到 `clip` 列表中。

(7). **进行模型推理**
   ```python
   if len(clip) == 16:
       inputs = np.array(clip).astype(np.float32)
       inputs = np.expand_dims(inputs, axis=0)
       inputs = np.transpose(inputs, (0, 4, 1, 2, 3))
       inputs = torch.from_numpy(inputs)
       inputs = torch.autograd.Variable(inputs, requires_grad=False).to(device)
       
       with torch.no_grad():
           outputs = model.forward(inputs)
       
       probs = torch.nn.Softmax(dim=1)(outputs)
       label = torch.max(probs, 1)[1].detach().cpu().numpy()[0]
   ```
   - 当 `clip` 列表中帧数达到 16 时，转换成适合模型输入的格式，进行模型推理，得到预测结果。

(8). **显示预测结果**
   ```python
   cv2.putText(frame, class_names[label].split(' ')[-1].strip(), (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
   cv2.putText(frame, "prob: %.4f" % probs[0][label], (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
   ```
   - 将预测类别和概率显示在视频帧上。

(9). **显示视频**
   ```python
   cv2.imshow('C3D_model', frame)
   if cv2.waitKey(10) & 0xFF == ord(' '):
       break
   ```
   - 使用 OpenCV 显示视频帧，按下空格键停止播放。

(10). **释放资源**
   ```python
   cap.release()
   cv2.destroyAllWindows()
   ```
   - 释放视频捕捉资源并关闭所有 OpenCV 窗口。
    
### 效果请查看视频
    
### 至此整个项目结束。

<font color="red" size="5">注意</font>
<p style="text-indent: 2em;"><font color="red">由于Jupyter notebook本质上是在浏览器中运行的交互式环境，在调用OpenCV的imshow方法时，会出现窗口不会自动关闭的问题，这主要是因为imshow是一个阻塞函数，它需要一个独立的窗口管理，而Jupyter notebook不完全支持这种模式，这里给出几个常见的解决方案：</font><p>

- 1.确保正确释放窗口
- 2.调整waitKey时间
- 3.使用Jupyter notebook魔法命令
- 4.更换开发环境，例如使用PyCharm（推荐）

# 编写不易，觉得这个仓库有用的还请各位大佬点个Star，如需协助请联系:xjx12@foxmail.com
