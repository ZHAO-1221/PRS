# 项目架构说明

本项目使用 RTMDet 检测视频中的人和狗，使用 RTMPose 逐帧提取关键点，并将连续骨架序列输入 ST-GCN，完成狗行为识别和人狗互动识别。

RTMPose 在本项目中只负责关键点提取，不进行单帧姿态分类。当前场景限定为一个人和一只狗，因此暂不设置多目标追踪模块。

## 目录结构

```text
pet_recognition_system/
├── configs/                         RTMDet、RTMPose和ST-GCN的模型及训练配置文件
│
├── pet_recognition/                 项目的核心Python代码
│   ├── models/                      RTMDet、RTMPose和ST-GCN的模型结构与调用代码
│   ├── datasets/                    数据集读取、标签解析和数据增强代码
│   ├── preprocessing/               关键点过滤、插值、平滑和归一化处理
│   ├── inference/                   视频读取、关键点提取和行为识别推理流程
│   ├── postprocessing/              预测平滑、片段合并和行为时间轴生成
│   ├── evaluation/                  AP、PCK、Accuracy、Macro-F1和混淆矩阵计算
│   └── visualization/               检测框、骨架、预测标签和时间轴可视化
│
├── tools/                           数据准备、模型训练、测试和完整流程运行脚本
│
├── data/                            原始视频、标注、短视频和骨架序列数据集
│
├── checkpoints/                     下载或训练完成后选定的模型权重文件
│
├── work_dirs/                       训练日志、中间权重和实验结果
│
├── outputs/                         检测、关键点、预测、时间轴和截取视频的输出目录
│
├── app/                             移动端原型的前端页面和后端接口
│
├── demo/                            RTMPose、ST-GCN及完整系统的演示程序
│
├── tests/                           各模块及完整流程的自动测试代码
│
├── requirements.txt                 项目所需Python库及版本
│
├── setup.py                         项目的Python包安装配置
│
└── README.md                        项目介绍、安装方法、数据准备和运行说明
```

## 主要目录职责

### `configs/`

存放 RTMDet、RTMPose 和 ST-GCN 的模型、数据集、训练及推理配置。该目录保存参数，不保存模型实现代码或权重。

### `pet_recognition/`

项目的核心 Python 包，包含数据读取、模型调用、骨架处理、推理、后处理、评估和可视化功能。

- `models/`：模型结构以及 RTMDet、RTMPose、ST-GCN 的封装代码。
- `datasets/`：狗关键点、狗行为及人狗互动数据集的读取和转换代码。
- `preprocessing/`：处理低置信度点、缺失点、骨架抖动和坐标归一化。
- `inference/`：连接视频解码、人狗检测、关键点提取和行为识别流程。
- `postprocessing/`：平滑分类结果，合并连续行为片段并生成时间轴。
- `evaluation/`：计算关键点和行为识别的评估指标。
- `visualization/`：绘制检测框、骨架、行为标签和时间轴。

### `tools/`

存放可以直接从命令行运行的脚本，例如数据准备、模型训练、模型测试、批量关键点提取和完整视频处理。

### `data/`

存放原始视频、训练图片、关键点标注、行为短视频、类别标签、骨架序列以及训练集、验证集和测试集划分。

### `checkpoints/`

存放下载或训练后最终选定的 `.pth` 模型权重。模型结构代码位于 `pet_recognition/models/`。

### `work_dirs/`

存放训练过程产生的日志、每轮权重、验证结果和其他实验记录。最终选定的权重再复制到 `checkpoints/`。

### `outputs/`

存放系统运行结果，包括检测框、逐帧关键点、骨架序列、行为预测、时间轴、截取的视频片段和可视化视频。

### `app/`

存放移动端原型所需的前端页面和后端接口，实现视频上传、处理进度、行为时间轴和片段回放。

### `demo/`

存放 RTMPose、ST-GCN 和完整系统的最小演示程序，用于快速验证模型能否运行。

### `tests/`

存放数据处理、关键点提取、ST-GCN 推理和完整流水线的自动测试。

## 根目录文件

- `requirements.txt`：记录项目依赖及版本。
- `setup.py`：将 `pet_recognition` 安装为可导入的 Python 包。
- `README.md`：项目简介、安装方法和主要使用入口。

## 容易混淆的目录

```text
pet_recognition/models/   模型结构和调用代码
checkpoints/              最终选定的模型权重
work_dirs/                训练过程文件和实验记录
outputs/                  系统推理产生的结果
```
