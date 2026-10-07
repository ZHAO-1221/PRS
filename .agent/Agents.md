# AGENTS.md — PRS Group 28 项目说明

## 项目一句话

**基于多模态视频与骨架特征的室内狗行为及人狗互动识别**（NUS PRS Practice Module，Group 28）。
让家用宠物摄像头从"看得见狗在哪"走向"看得懂狗在做什么、和谁在互动"。

## 课程要求（摘自 `PRS_Practice_Module_项目要求汇总.md`）

- 4 个方向至少覆盖 3 个：监督/无监督学习、ML/DL、混合/集成、智能感知/意义建构。
- 最终交付（打包 1 个 ZIP）：可运行系统、数据集、最终报告（工具/技术、系统设计、性能、讨论）、代码与模型文件、10–15 分钟视频演示、两次演示的 Slides。

## 系统设计

### 范围

- 只处理**单只宠物**，或**单只宠物 + 单人**的画面；不做多人多宠。
- 只用视觉骨架，**不做音频识别**。

### 两个模型

| | 模型 1 骨架提取 | 模型 2 行为/互动分类 |
|---|---|---|
| 输入 | 整段视频 | 模型 1 输出的连续骨架帧 |
| 输出 | 逐帧标注好骨架的帧（狗骨架，有人时加人骨架） | 姿态类别：狗行为 + 人狗互动 |
| 方法 | RTMDet 检测人/狗 → RTMPose 估计关键点（人 256×192，狗 256×256；CSPNeXt-M + RTMCCHead + SimCC；两分支不共享权重，各 17 点） | ST-GCN（9 block，64/128/256 通道），基线 LSTM；画面有人时拼接人骨架 + 关系特征 |

- ST-GCN 邻接分区采用论文中的第三种（spatial configuration，按重心分 A0/A1/A2）。
- 损失：加权交叉熵（类别不平衡），互动可试 Focal Loss。

### 类别（一个片段可同时有狗行为和互动两个标签）

- **狗行为（6 类）**：站、坐、躺或睡、走或跑、进食或喝水、独自玩耍。
- **人狗互动（5 类）**：none（无互动）、feeding（喂食）、petting（抚摸，小幅度姿态）、gaming（玩耍，大幅度姿态）、静态共处。

### 关系特征

- 人宠距离：由人、宠骨架中心点计算。
- 人手部/肘部关键点与宠物骨架的重合时间。
- 人宠骨架中心点相邻帧的相对位移（pending，待定）。

### 在线推理流程

1. 输入室内视频
2. 模型 1：检测人和狗 + 估计关键点，输出带骨架的帧
3. 串联成骨架序列，插值补缺失点 + 平滑 + 归一化，计算关系特征
4. 模型 2：输出狗行为 / 人狗互动类别
5. 输出类别、置信度、时间线

### 狗姿态模型三阶段

- **P0**：官方 RTMPose-M AP-10K 预训练，不再训练，直接测室内数据（基线）
- **P1**：P0 + APT-36K 狗类数据训练
- **P2**：P1 + 自建室内数据微调（小学习率 + 增强：翻转换左右点、尺度/旋转、遮挡）
- 不用 AP-10K 再训练：预训练已用过，再训无新信息且测试不公平。
- 注意：`diagrams/Dog_Pose_Three_Stage_Adaptation_*` 图中 P1 仍写 "AP-10K Train"，实际应为 APT-36K。

## 数据

- **APT-36K**：只用狗类，沿用官方划分，不重标。帧均为 1920×1080，每帧一个 JSON 标注。
- **自建室内数据**：授权或公开视频，人工标注 300–800 张图（狗框 + 17 点，格式同 APT 的逐帧 JSON），另加片段级行为/互动标签。
- **测试参考**：夜间猫狗视频（见 `viedo_link.md`）。
- **铁律**：先按狗、按视频划分 train/val/test，再抽帧切片段，防止数据泄漏。验证集只用于选模型，测试集只在最后用一次。

其它数据集/模型链接见 `viedo_link.md`（AP-10K、Animal-Pose、DETRPose、mmpose RTMPose 配置）。

## 评估

- 姿态：COCO AP、AP@0.5、AP@0.75、PCK，分头部/前腿/后腿报告；比较 P0/P1/P2。
- 狗行为：Accuracy、Macro-F1、Balanced Acc、各类 P/R/F1、混淆矩阵、片段级 F1；LSTM vs ST-GCN；消融置信度、缺失点处理、骨架增强。
- 互动：同上指标；有无人骨架、有无关系特征。
- 鲁棒性：未见过的狗/视频；遮挡、关键点缺失、相似类别混淆（如 petting vs 静态共处）。

## 仓库结构

```
PRS_Practice_Module_项目要求汇总.md   课程要求与往届示例
Reolink_Background.md                  选题背景：Reolink 公司、痛点、与课程映射
Reolink_CCTV_产品分类与数据集调研.md   CCTV 产品分类与可用数据集调研
viedo_link.md                          数据集 / 模型链接
proposal/                              Proposal（中/英 docx + 提交版 PDF）
diagrams/                              架构图，HTML 源 + PNG，_CN / _EN 成对
presentation/
  index.html                           网页版 Proposal 演示
  Proposal_Group28_CN.pptx             中文 PPT（英文版的源）
  build_en_pptx.py                     由中文 PPT 生成英文 PPT + 双语讲稿（python-pptx）
  Proposal_Group28_EN_Script*.txt      12 页双语讲稿（_new 为最新）
  Group 28.pptx                        最终演示用 PPT
  images/                              幻灯片配图（由 diagrams 导出）
frontend/
  playful-a-sticker-story.html         手机端原型（每日报告：关键片段 + 休息/活动/陪伴时长）
```


## Agent 文件

- `.agent/prime.md`：开发进度与里程碑，新会话先读，完成一步就更新。
- `.agent/lesson.md`：踩坑与技巧，遇到就记。
