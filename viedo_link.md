训练集：
https://github.com/AlexTheBad/AP-10K 日间宠物骨架
https://huggingface.co/datasets/ohicarip/interpet4d 人宠互动音频（英文）
https://zenodo.org/records/18972388 人宠互动音频
https://sites.google.com/view/animal-pose/ 该数据集提供了五个类别的动物姿态注释：狗、猫、牛、马、绵羊，总计在4000+张图片中有6000+个实例。此外，数据集还包含其他7个动物类别的边界框注释。20个点
AP36k 数据来自约 2,400 段视频片段，约 36,000 帧，覆盖 30 个物种，17点骨架，和AP-10k一致，基本都是室外场景。先去掉灵长类，训练后再用犬科微调。


测试集：
https://www.youtube.com/watch?v=L7VKt2QSaZw  夜间猫狗
https://petcube.com/blog/pets-caught-on-night-vision-pet-camera/ 夜间宠

模型：
https://github.com/SebastianJanampa/DETRPose 多目标姿态检测
https://github.com/open-mmlab/mmpose/tree/main/configs/body_2d_keypoint/rtmpose 动物的
