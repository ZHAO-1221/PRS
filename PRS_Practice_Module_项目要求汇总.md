# PRS Practice Module 项目说明

## 项目要求

- 选题：任何能体现模式识别与机器学习技术优势的实际应用
- 用 AI/ML 技术设计并构建一个模式识别系统，解决真实世界问题，并用到 3 门课程模块所教的技能
- 以下 4 个方面至少覆盖 3 个：
  1. Supervised / Unsupervised learning（监督 / 无监督学习）
  2. Machine learning / Deep learning（机器学习 / 深度学习）
  3. Hybrid machine learning / Ensemble（混合学习 / 集成方法）
  4. Intelligent sensing / Sense making（智能感知 / 意义建构）

## 项目示例

题目参考：股票市场预测、语音转文字、人脸情绪识别、自动图像描述生成、手势识别、迷你社交机器人、皮肤癌检测、缺陷检测系统

**Skin Cancer Detection**：离线训练 + 在线部署
- 在 Google Colab 上训练 Keras 神经网络
- Flask + Docker 部署：手机拍照 → 模型 → 输出分类结果

**Private Healthcare Administration**：一套流程服务 3 个用例
- 用例：坏账（write-off）预测、住院收入预测、入院时账单估算
- 流程：数据源 → EDA & 特征提取 → 多模型训练（Google Cloud）→ 按 loss 选最优模型 → 预测 → Streamlit 前端
- 模型：
  - Write-off 预测：Decision Trees、Logistic Regression、Naive Bayes、NN、KNN
  - 收入预测：NN、ARIMA
  - 账单估算：Linear Regression、NN、Random Forest、XGBoost

**Crowdedness Prediction**：同一问题用 4 种方法对比
- Image Regression：直接从图像预测人数（Sequential CNN、ResNet）
- Density-Based：生成密度图（U-Net，试了 4 层 / 3 层 encoder）
- Object Detection：用自标注的人头框训练（YOLOv5s）
- Image Classification：分 Sparse / Medium / Crowded 三类（Sequential CNN、ResNet）

**FoodieVision**：分阶段递进
1. 识别单个食物（Bun、Soup、Egg、Potato Fries、Juice 等）
2. 食物分割（segmentation）
3. 自然语言问答界面

示例里常见的做法：多模型对比选优、多种方法解同一问题、分阶段递进、有可交互的部署前端（Flask / Docker / Streamlit）。

## 交付物（小组打包成 1 个 ZIP）

- 可运行的模式识别系统
- 数据集
- 最终报告：工具 / 技术、系统设计 / 模型、系统性能、发现与讨论
- 代码、模型文件、其他支持文档（如有）
- 视频演示（.mp4/.mov/.wmv 等），10–15 分钟
- 两次演示的 Slides
