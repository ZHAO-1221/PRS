# PRIME — 开发进度

新会话先读这里，再看 `Agents.md` 了解项目全貌。每完成一步就更新本文件。

## 当前状态（2026-10-06）

Proposal 阶段完成，尚未开始写训练/推理代码。

## 里程碑

- [x] 选题与背景调研（Reolink 背景、CCTV 数据集调研）— 2026-09-24
- [x] Proposal 中英文版 + PDF 提交 — 2026-09-29
- [x] 架构图中英文版 — 2026-10-04
- [x] Proposal 演示 PPT（中/英）+ 双语讲稿 — 2026-10-06
- [x] 手机端前端原型（静态 HTML）— 2026-10-05
- [ ] 数据：APT-36K 狗类整理、自建室内数据采集与标注（框 + 17 点）
- [ ] 数据划分：按狗 / 按视频切 train/val/test
- [ ] 狗姿态 P0 基线测试 → P1（APT-36K）→ P2（室内微调）
- [ ] 检测跟踪：RTMDet + ByteTrack，导出骨架序列（JSON/NPY）
- [ ] Task A：LSTM 基线 + ST-GCN
- [ ] Task B：双流 ST-GCN + 关系特征 + 音频融合
- [ ] 评估与消融实验
- [ ] 可运行 Demo（上传视频 → 行为/互动/时间线）
- [ ] 最终报告、10–15 分钟视频、第二次演示 Slides、ZIP 打包

## 待办 / 已知问题

- `diagrams/Dog_Pose_Three_Stage_Adaptation_*` 中 P1 仍写 "AP-10K Train"，应改为 APT-36K。

## 进度日志

<!-- 新的写在最上面：日期 · 谁 · 做了什么 · 下一步 -->

- 2026-10-06 · 建立 `.agent/`：Agents.md（项目说明）、prime.md（进度）、lesson.md（经验）。
