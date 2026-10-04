"""Build Proposal_Group28_EN.pptx from the Chinese deck: English slide text,
English diagrams, bilingual (EN/中文) speaker notes.  Run: python build_en_pptx.py"""
from pathlib import Path
from pptx import Presentation
from pptx.util import Pt, Inches

HERE = Path(__file__).parent
SRC, OUT = HERE / "Proposal_Group28_CN.pptx", HERE / "Proposal_Group28_EN.pptx"

# shape name -> paragraphs; a paragraph is a str or a list of run texts (keeps per-run styling)
TEXT = {
    1: {
        "t1": "DOG BEHAVIOR & HUMAN–DOG INTERACTION · PROPOSAL",
        "t4": ["Indoor Dog Behavior &", ["Human–Dog ", "Interaction", ""]],
        "t6": "Recognition via dual-species RTMPose skeletons + ST-GCN temporal modeling",
    },
    2: {
        "t8": "01 · BACKGROUND",
        "Text 0": "Pet Cameras See, but Don't Understand",
        "t11": "MARKET · 2024",
        "t12": "Pet cameras go mainstream",
        "t13": "Owners treat dogs as family and want to know they are safe and well while away.",
        "t14": "$198M",
        "t15": "Global market size, 2024",
        "t17": "GROWTH · 2024–2030",
        "t18": "Demand keeps growing",
        "t19": "Expected to reach ~$454M by 2030, driven by owners treating pets as family.",
        "t21": "Compound annual growth rate",
        "t23": "GAP",
        "t24": "Today's products only \"see\"",
        "t25": "Detect, track, push alerts. Did the dog eat? Did anyone play with it? Owners still scrub the footage.",
        "t26": "Understand",
        "t27": "The step this project adds",
        "t28": "Source: Grand View Research, Pet Monitoring Camera Market Report",
    },
    3: {
        "t29": "02 · OBJECTIVES",
        "t30": "GOAL · A WORKING VIDEO-ANALYSIS PROTOTYPE",
        "Text 0": "Make the Camera Understand the Dog",
        "t31": "Upload an indoor video → get dog behavior, interaction type, confidence and a timeline.",
        "t35": "Dog Behavior Recognition",
        "t36": "Dog only: infer what it is doing from how its skeleton moves over time.",
        "t38": ["Input: dog skeleton sequence T × 17 × 3",
                "Model: ST-GCN (LSTM baseline)",
                "6 classes: stand/walk · sit · lie/sleep · eat/drink · play · other"],
        "t42": "Human–Dog Interaction",
        "t43": "Human and dog together: motion + human–dog relation + sound.",
        "t45": ["Input: human + dog skeletons + relation features (optional audio)",
                "Model: dual-stream ST-GCN, late fusion",
                "4 classes: none · feeding · petting · playing together"],
    },
    4: {
        "t46": "03 · SYSTEM & DATA",
        "Text 0": "System Architecture",
        "t49": ["Offline: dog pose AP-10K → APT-36K dogs → indoor data; human pose COCO-17",
                "Online in 6 steps: video → detect/track → pose → skeleton graph → recognition → output",
                "Tracking: RTMDet / YOLO + ByteTrack / BoT-SORT"],
        "t50": "Dual-species skeletons are the shared representation: pose estimation sees the joints, ST-GCN understands the motion.",
        "t52": "INFERENCE", "t54": "Video in, JSON + timeline out",
        "t56": "TASKS", "t58": "Task A behavior · Task B interaction",
        "t60": "METRICS", "t62": "AP / PCK · Acc / Macro-F1",
    },
    5: {
        "t63": "03 · SYSTEM & DATA",
        "Text 0": "Where the Data Comes From",
        "t67": ["① APT-36K: official split, dog samples only, no re-labeling",
                "② Own videos: Group ID per dog + full video, split first, then sample frames and clips",
                "Split first, sample later: one dog or video never spans train and test"],
        "t68": "AP-10K is RTMPose's pretraining set, so we don't train on it again; public dog pose data comes from APT-36K.",
        "t70": "SOURCES", "t72": "APT-36K dogs · indoor videos",
        "t74": "FRAME LABELS", "t76": "indoor images: dog box + 17 keypoints",
        "t78": "OUTPUTS", "t80": "Keypoints · clip labels · derived features",
    },
    6: {
        "t81": "03 · SYSTEM & DATA",
        "Text 0": "Training Dog Pose",
        "t84": ["P0: official AP-10K RTMPose-M, tested as-is (baseline)",
                "P1: fine-tune on the APT-36K dog subset",
                "P2: keep fine-tuning on indoor data with a small learning rate"],
        "t85": "One shared test set. If P1 beats P0, dog specialization helps; if P2 beats P1, indoor adaptation helps.",
        "t87": "VERSIONS", "t89": "P0 → P1 → P2, stage by stage",
        "t91": "TEST SET", "t93": "One held-out test set for all three",
        "t95": "POSE METRICS",
    },
    7: {
        "t98": "04 · MODEL DESIGN",
        "Text 0": "Dual-Species Skeleton Extraction",
        "t102": ["Human and dog branches share one design: affine crop → CSPNeXt-M → RTMCCHead → SimCC",
                 "Only input size differs (human 256×192, dog 256×256); no shared weights",
                 "Aligned across frames by track ID, then masked, interpolated and smoothed"],
        "t103": "RTMCCHead = 7×7 conv + FC + GAU; SimCC turns coordinate regression into classification.",
        "t105": "BACKBONE", "t107": "CSPNeXt-M output channels",
        "t109": "KEYPOINTS", "t111": "Human COCO · dog AP-10K",
        "t113": "OUTPUT", "t115": "2 streams · x, y, confidence",
    },
    8: {
        "t116": "04 · MODEL DESIGN",
        "Text 0": "How Graph Convolution Works",
        "t119": ["Example: human right elbow and dog left elbow with their 1-hop neighbors (D = 1)",
                 "Uni-labeling: 1 group; distance: root / 1-hop, 2 groups",
                 "Spatial configuration: root, centripetal, centrifugal by distance to the center of gravity, 3 groups"],
        "t120": "Both networks use spatial configuration partitioning: one adjacency matrix per group, A₀, A₁, A₂.",
        "t122": "STRATEGIES", "t124": "Uni-label · distance · spatial",
        "t126": "NEIGHBORHOOD", "t128": "Root node and 1-hop neighbors",
        "t130": "ADJACENCY", "t132": "Spatial config., shared by Task A / B",
    },
    9: {
        "t133": "04 · MODEL DESIGN",
        "Text 0": ["Task A", "Dog Behavior"],
        "t137": ["Follows the original ST-GCN (Yan et al., AAAI 2018)",
                 "9 residual units in 3 stages: channels 64 → 128 → 256, time T → T/2 → T/4",
                 "Global average pooling → Linear + Softmax; baseline: LSTM"],
        "t138": "Each unit: spatial graph conv (A₀–A₂), then 9×1 temporal conv, then residual add.",
        "t140": "RESIDUAL UNITS", "t142": "3 stages × 3 units",
        "t144": "FEATURE", "t146": "After global average pooling",
        "t148": "CLASSES", "t150": "Stand · sit · lie · eat · play · other",
    },
    10: {
        "t151": "04 · MODEL DESIGN",
        "Text 0": ["Task B", "Interaction"],
        "t154": ["Human and dog skeletons each go through an ST-GCN → f_h and f_d",
                 "Plus relation features f_r (distance, velocity, orientation) and optional audio (YAMNet / MERT)",
                 "Concat + MLP at the end: late fusion"],
        "t155": "Late fusion keeps audio optional: with no or poor sound, the skeletons alone still give a prediction.",
        "t157": "STREAMS", "t159": "Human + dog, one ST-GCN each",
        "t161": "RELATION", "t163": "Distance · velocity · orientation",
        "t165": "CLASSES", "t167": "None · feeding · petting · play",
    },
    11: {
        "t168": "05 · EVALUATION",
        "Text 0": "How We Prove Each Step Works",
        "t172": "Pose",
        "t174": "P0 / P1 / P2 on the same test set, reported for head, forelimbs and hindlimbs.",
        "t177": "Dog Behavior",
        "t178": "Accuracy · Macro-F1 · confusion matrix",
        "t179": "LSTM vs ST-GCN; ablate confidence, missing-joint handling, skeleton augmentation.",
        "t182": "Interaction",
        "t183": "Same metrics + ablations",
        "t184": ["Single vs dual stream", "± relation features", "± audio"],
        "t187": "Robustness",
        "t188": "Unseen dogs and scenes",
        "t189": ["Split by dog + full video", "Occlusion, missing joints", "Multi-subject, no audio"],
        "t192": "Val set for model selection only; test set used once at the end, the same one for every comparison.",
    },
    12: {
        "t197": 'From "where is my dog?" to "what is it doing, and with whom?"',
        "t203": "Skeletons link everything",
        "t204": "Pose estimation sees joints; ST-GCN understands motion over time.",
        "t207": "Every step is testable",
        "t208": 'P0 → P1 → P2 and ablations answer "does it help?" one by one.',
        "t211": "Leak-free data split",
        "t212": "Grouped by dog and full video: we test on unseen dogs and scenes.",
        "t214": "→ QUESTIONS WELCOME · 5 MIN Q&A",
    },
}

# per-shape font-size overrides (pt) where English is wider than the Chinese it replaces
SIZE = {(3, "t35"): 30, (3, "t42"): 30, (2, "t26"): 48}

IMAGES = {
    "diagram-system-overview": "04-overall-en.png",
    "diagram-data-preparation": "05-data-en.png",
    "diagram-dog-pose-stages": "06-dog-pose-stages-en.png",
    "diagram-dual-species-pose": "07-rtmpose-en.png",
    "diagram-stgcn-partition": "08-partition-en.png",
    "diagram-task-a-stgcn": "09-task-a-stgcn-en.png",
    "diagram-task-b-dual-stream": "10-task-b-dual-stream-en.png",
}

W, Z = "Sun Wenjing", "Zhao Jiayi"
# slide -> (speaker, time window, English paragraph, 中文段落, optional tip)
NOTES = {
    1: (W, "0:00–0:40 (~40 s)",
        "Good morning, everyone. We are Group 28. I am Sun Wenjing, and this is my teammate Zhao Jiayi. Our project is about indoor dog behavior and human–dog interaction recognition. We use video and skeleton features. Our idea is simple. A home camera should not only see the dog. It should also understand what the dog is doing, and how people interact with it. First, I will talk about the background, our goals, the system and the data. Then Jiayi will talk about the models and how we will test them.",
        "各位老师、同学好，我们是第 28 组。我是 Sun Wenjing，这位是我的组员 Zhao Jiayi。我们的项目是室内狗行为和人宠互动识别，用到视频和骨架特征。我们的想法很简单：家里的摄像头不应该只是拍到狗，还应该看懂狗在做什么、人和狗在怎样互动。我先介绍背景、目标、系统和数据，然后由 Jiayi 介绍模型和评估方法。",
        None),
    2: (W, "0:40–2:10 (~1.5 min)",
        "Why do we do this project? Today, many families have pets. Many owners see their dog as part of the family. When they are not at home, they want to know if the dog is safe and healthy. The market shows this clearly. In 2024, the global pet camera market was about 198 million US dollars. By 2030, it may reach about 454 million. That is a growth of about 12.6 percent every year. But most pet cameras can only see. They find the dog in the picture. They follow it. They send an alert when there is noise or movement. Owners want to know more. Did my dog eat? Did it sleep well? Did someone feed it or play with it? Today, owners still need to watch the videos by themselves. Also, one picture is not enough. Eating and sniffing look very similar. Sleeping and lying down look similar. Petting and just standing close also look similar. The difference is in how the body moves over time, and how the person and the dog move together. So our project adds one step: from seeing to understanding.",
        "为什么做这个项目？现在很多家庭都养宠物，很多主人把狗当成家人。不在家的时候，他们想知道狗是否安全、健康。市场数据也说明了这一点：2024 年全球宠物摄像头市场约 1.98 亿美元，到 2030 年可能达到约 4.54 亿美元，每年增长约 12.6%。但是大多数宠物摄像头只能“看见”：找到画面里的狗、跟着拍、有声音或动静就发提醒。主人想知道的更多：狗吃饭了吗？睡得好吗？有没有人喂它、陪它玩？现在主人还是要自己翻录像。而且只看一张图是不够的：进食和闻嗅看起来很像，睡觉和躺着很像，抚摸和只是站在旁边也很像。区别在于身体随时间怎么动，以及人和狗怎么一起动。所以我们的项目要补上一步：从“看得见”到“看得懂”。",
        None),
    3: (W, "2:10–3:25 (~1.25 min)",
        "Our goal is a working prototype. The user uploads an indoor video. The system then tells us three things: what the dog is doing, how the person and the dog interact, and when it happens. It also gives a confidence score. We have two tasks. Task A is on the left. It is dog behavior recognition. Here we only look at the dog. The input is the dog skeleton over time. We use a model called ST-GCN, and we compare it with an LSTM. There are six classes: stand or walk, sit, lie or sleep, eat or drink, play, and other. Task B is on the right. It is human–dog interaction recognition. Here we look at both the person and the dog. The input is the human skeleton, the dog skeleton, and some relation features. Audio is optional. We use a two-stream ST-GCN. There are four classes: no interaction, feeding, petting, and playing together. One clip can have both labels. For example, the dog is eating, and the person is feeding it.",
        "我们的目标是做一个能运行的原型。用户上传一段室内视频，系统告诉我们三件事：狗在做什么、人和狗怎样互动、发生在什么时间，同时给出置信度。我们有两个任务。左边是 Task A，犬行为识别，只看狗。输入是狗骨架随时间的变化，用 ST-GCN 模型，并和 LSTM 对比。一共 6 类：站或走、坐、躺或睡、吃或喝、玩耍、其他。右边是 Task B，人宠互动识别，同时看人和狗。输入是人骨架、狗骨架和一些关系特征，音频可选。我们用双流 ST-GCN，一共 4 类：无互动、喂食、抚摸、一起玩。一个片段可以同时有两个标签，比如狗在吃东西，同时人在喂它。",
        None),
    4: (W, "3:25–4:55 (~1.5 min)",
        "This is the full system. Every later slide shows one part of this picture. The top part is offline training. For dog pose, we start from a model trained on AP-10K. Then we train it on dog images from APT-36K. Then we fine-tune it on our own indoor data. For human pose, we use a ready model trained on COCO. For the recognition models, we use the validation set to choose the best model. We use the test set only once, at the end. The bottom part is online inference. It has six steps. Step one: we take an indoor video. Step two: we find the people and dogs, and track them. We use RTMDet or YOLO to find them, and ByteTrack or BoT-SORT to track them. So each person and each dog keeps the same ID. Step three: we find the body joints of the person and the dog. Step four: we turn the joints into a skeleton graph. Step five: we recognize the behavior and the interaction. Step six: we output the class, the confidence and the time. The main idea is simple. The skeleton connects everything. Pose estimation finds the joints. ST-GCN understands the movement.",
        "这是整个系统，后面每一页讲的都是这张图的一部分。上半部分是离线训练。狗姿态模型先用 AP-10K 训练好的模型，再用 APT-36K 的狗图片训练，最后用我们自己的室内数据微调。人体姿态直接用 COCO 上训练好的模型。识别模型用验证集选出最好的模型，测试集只在最后用一次。下半部分是在线推理，一共六步。第一步，输入室内视频。第二步，找到画面里的人和狗并跟踪：用 RTMDet 或 YOLO 检测，用 ByteTrack 或 BoT-SORT 跟踪，让每个人和每只狗保持同一个 ID。第三步，找出人和狗的身体关节点。第四步，把关节点变成骨架图。第五步，识别行为和互动。第六步，输出类别、置信度和时间。核心思想很简单：骨架把一切连接起来。姿态估计负责找关节，ST-GCN 负责理解动作。",
        "The training box in the diagram still says \"AP-10K … dog subset\". Say it is APT-36K. / 图中训练数据处仍写 AP-10K，口头说明是 APT-36K。"),
    5: (W, "4:55–6:15 (~1.3 min)",
        "Now, where does our data come from? We have two sources. The first source is a public dataset called APT-36K. We do not train on AP-10K. The reason is simple. Our pose model was already trained on AP-10K. Training on it again teaches the model nothing new, and the test would not be fair. From APT-36K, we only take the dog images. We keep the official split, and we do not label them again. The second source is indoor videos. Some we record ourselves. Some are used with permission. Some are public. Here we follow one important rule. We split the data first, by dog and by video. After that, we cut frames and clips. So the same dog or the same video never appears in both training and testing. This stops data leakage. In the end, we get three kinds of data. First, 300 to 800 indoor images, each with a dog box and 17 joints. Second, labels for each clip, for behavior and interaction. Third, features we get automatically, like human joints, bowls, toys and sound.",
        "我们的数据从哪里来？有两个来源。第一个是公开数据集 APT-36K。我们不用 AP-10K 训练，原因很简单：我们的姿态模型已经在 AP-10K 上训练过了，再训练一次学不到新东西，测试也不公平。APT-36K 我们只取狗的图片，沿用官方划分，不重新标注。第二个来源是室内视频，有的是我们自己拍的，有的是获得授权的，有的是公开视频。这里我们遵守一条重要规则：先按狗和视频划分数据，再去抽帧和切片段。这样同一只狗、同一段视频不会同时出现在训练和测试里，避免数据泄漏。最后我们得到三类数据：第一，300 到 800 张室内图片，每张标出狗框和 17 个关节点；第二，每个片段的行为和互动标签；第三，自动得到的特征，比如人体关节、食盆、玩具和声音。",
        None),
    6: (W, "6:15–7:30 (~1.25 min) → hand over",
        "Next, we train the dog pose model in three stages. P0 is the baseline. We use the official AP-10K model with no extra training, and we test it on our indoor data. We want to see how it works when the dog lies on its side, is partly hidden, is in low light, or moves fast. P1 adds dog training. We start from P0 and train on the dog images from APT-36K. This should help the model find the head and legs better. P2 adds indoor training. We start from P1 and train on our own indoor images with a small learning rate. We also flip, scale, rotate and partly cover the images to get more training data. We test all three on the same test set. We use four scores: COCO AP, AP at 0.5, AP at 0.75, and PCK. If P1 is better than P0, dog training helps. If P2 is better than P1, indoor training helps. That is all for the data part. Now Jiayi will explain the models.",
        "接下来，狗姿态模型分三个阶段训练。P0 是基线：直接用官方 AP-10K 模型，不做额外训练，在我们的室内数据上测试，看它在狗侧躺、被遮挡、光线暗、动作快的时候表现如何。P1 加入狗的训练：从 P0 出发，用 APT-36K 的狗图片训练，让模型更准地找到头和腿。P2 加入室内训练：从 P1 出发，用我们自己的室内图片、以较小的学习率训练，还会对图片做翻转、缩放、旋转和部分遮挡来增加数据。三个版本在同一个测试集上比较，用四个指标：COCO AP、AP@0.5、AP@0.75 和 PCK。如果 P1 比 P0 好，说明狗的训练有用；如果 P2 比 P1 好，说明室内训练有用。数据部分就到这里，下面请 Jiayi 介绍模型。",
        "The P1 box in the diagram still says \"AP-10K Train\". Say it is APT-36K. / 图中 P1 训练数据仍写 AP-10K，口头说明是 APT-36K。"),
    7: (Z, "7:30–8:50 (~1.3 min)",
        "Thank you, Wenjing. Now let's look at the online part. This slide is step three: finding the joints of the person and the dog. We have two branches, one for the person and one for the dog. Both use the same design, called RTMPose. First, we cut out the person or the dog from the frame. Next, a backbone called CSPNeXt-M turns the image into features. It outputs 768 channels. Then a head called RTMCCHead processes these features. Finally, SimCC finds the x and y position of each joint. It treats this as a classification problem. The two branches have one difference: the input size. The person image is 256 by 192. The dog image is 256 by 256. They also do not share weights, because people and dogs have very different bodies. Each branch finds 17 joints in every frame. Each joint has an x value, a y value and a confidence score. Then we link the joints across frames by track ID. We also fill in missing joints and smooth the result. The final output has two streams, T frames, 17 joints and 3 values.",
        "谢谢 Wenjing。下面看在线部分。这一页是第三步：找出人和狗的关节点。我们有两个分支，一个给人，一个给狗，结构相同，都是 RTMPose。首先，从画面里把人或狗裁剪出来。然后，主干网络 CSPNeXt-M 把图片变成特征，输出 768 个通道。接着，RTMCCHead 处理这些特征。最后，SimCC 找出每个关节的 x 和 y 位置，它把这个问题当作分类来做。两个分支只有一个区别：输入尺寸。人的图片是 256×192，狗的图片是 256×256。两个分支也不共享权重，因为人和狗的身体差别很大。每个分支在每一帧找出 17 个关节点，每个点有 x、y 和置信度。然后按跟踪 ID 把各帧的关节连起来，补上缺失的点，再做平滑。最终输出是 2 条流、T 帧、17 个关节、3 个值。",
        None),
    8: (Z, "8:50–10:05 (~1.25 min)",
        "Now we have skeletons. Next, we use graph convolution to recognize actions. This slide shows how graph convolution works on a skeleton. This picture is an example, not a pipeline. In the top row, the center joint is the right elbow of a person. In the bottom row, it is the left elbow of a dog. The dashed circle shows its direct neighbors. The ST-GCN paper gives three ways to group these neighbors. The first way puts all of them in one group. The second way uses two groups: the joint itself, and its neighbors. The third way uses three groups, based on the center of the body. One group is the joint itself. One group is closer to the center. One group is farther from the center. We use the third way in both of our models. Each group has its own matrix. We call them A0, A1 and A2. You will see them on the next slide.",
        "现在有了骨架，下一步用图卷积来识别动作。这一页说明图卷积在骨架上是怎么做的。这张图是一个例子，不是流程图。上面一行，中心关节是人的右肘；下面一行，中心关节是狗的左肘。虚线圈出的是它的直接邻居。ST-GCN 论文给了三种分组方法。第一种，把所有邻居放在一组。第二种，分两组：关节自己一组，邻居一组。第三种，按身体重心分三组：关节自己一组，离重心更近的一组，离重心更远的一组。我们两个模型都用第三种。每一组有自己的矩阵，叫 A0、A1 和 A2，下一页会看到。",
        None),
    9: (Z, "10:05–11:25 (~1.3 min)",
        "This is Task A, the dog behavior model. We follow the original ST-GCN paper by Yan and others, from AAAI 2018. The input is the dog skeleton. It has T frames, 17 joints and 3 values. First, we encode the graph and normalize the data. Then we have nine ST-GCN blocks in three stages. Stage one has 64 channels. Stage two has 128 channels, and the time length becomes half. Stage three has 256 channels, and the time length becomes a quarter. Each block does two things. First, it looks at nearby joints in the same frame, using A0 to A2. Then, it looks at the same joint across 9 frames. A shortcut adds the input back. After the blocks, we average all the features into one vector of 256 numbers. A linear layer and softmax then give the probability of each of the six behaviors. Dogs lie down much more than they play, so our data is not balanced. To fix this, we use a weighted cross-entropy loss. We compare this model with an LSTM.",
        "这是 Task A，犬行为识别模型，沿用 Yan 等人 AAAI 2018 的 ST-GCN 原论文。输入是狗的骨架：T 帧、17 个关节、3 个值。先做图编码和归一化，然后是九个 ST-GCN 模块，分三个阶段。第一阶段 64 通道；第二阶段 128 通道，时间长度减半；第三阶段 256 通道，时间长度变成四分之一。每个模块做两件事：先看同一帧里相邻的关节，用 A0 到 A2；再看同一个关节在 9 帧里的变化。最后用残差连接把输入加回来。所有模块之后，把特征平均成一个 256 维的向量，再经过线性层和 softmax，得到六种行为的概率。狗躺着的时间比玩的时间多很多，所以数据不平衡，我们用加权交叉熵来解决。对比模型是 LSTM。",
        None),
    10: (Z, "11:25–12:55 (~1.5 min)",
        "Task A only looks at the dog. Task B also looks at the person. It recognizes how they interact. We use two streams. The human skeleton goes into one ST-GCN. The dog skeleton goes into another ST-GCN. Each stream has the same three stages: 64, 128 and 256 channels. We get a human feature, f_h, and a dog feature, f_d. But the two skeletons are not enough. Feeding, petting and playing are different mainly because of how the person and the dog move together. So we add relation features, f_r. They include the distance between them, their speed toward each other, and which way they face. We can also add sound features from YAMNet or MERT. This is optional. At the end, we join all the features together. Then a small MLP gives the result. This is called late fusion. There are four classes: no interaction, feeding, petting and playing together. Late fusion has one big benefit: sound is optional. If a video has no sound, or bad sound, the model still works with skeletons only. This is important for real home cameras.",
        "Task A 只看狗，Task B 还要看人，识别人和狗怎样互动。我们用两条流：人骨架进一个 ST-GCN，狗骨架进另一个 ST-GCN。每条流都有同样的三个阶段：64、128、256 通道，分别得到人的特征 f_h 和狗的特征 f_d。但只有两个骨架还不够，因为喂食、抚摸和玩耍的区别，主要在于人和狗怎么一起动。所以我们加入关系特征 f_r，包括两者的距离、相互靠近的速度和朝向。我们还可以加入声音特征，用 YAMNet 或 MERT 提取，这是可选的。最后把所有特征拼在一起，用一个小的 MLP 给出结果，这叫后期融合。一共 4 类：无互动、喂食、抚摸、一起玩。后期融合有一个很大的好处：声音是可选的。如果视频没有声音或声音很差，模型只用骨架也能工作。这对真实的家用摄像头很重要。",
        None),
    11: (Z, "12:55–14:20 (~1.4 min)",
        "Finally, how do we show that each step really helps? We test four things. First, pose estimation. We compare P0, P1 and P2 with COCO AP, AP at 0.5, AP at 0.75 and PCK. We also report the head, front legs and back legs separately. Second, dog behavior. We report accuracy, macro-F1 and the confusion matrix. We compare LSTM with ST-GCN. We also remove some parts one at a time, to see if they help. These parts are the confidence score, the way we fill in missing joints, and data augmentation. Third, interaction. We use the same scores. We run three tests: one stream or two streams, with or without relation features, and with or without sound. Fourth, robustness. Our test set uses new dogs and new videos. So it shows how the model works on dogs it has never seen. We also check hard cases: hidden body parts, missing joints, many people or dogs, and no sound. We follow one rule the whole time. The validation set is only for choosing models. The test set is used only once, at the end.",
        "最后，怎么证明每一步真的有用？我们测四个方面。第一，姿态估计：用 COCO AP、AP@0.5、AP@0.75 和 PCK 比较 P0、P1、P2，并分别报告头部、前腿和后腿。第二，犬行为：报告准确率、macro-F1 和混淆矩阵，比较 LSTM 和 ST-GCN；还会一次去掉一个部分，看它有没有用，包括置信度、补缺失关节的方法和数据增强。第三，人宠互动：用同样的指标，做三组对比：单流还是双流、有没有关系特征、有没有声音。第四，鲁棒性：测试集用的是新的狗和新的视频，所以能看出模型在没见过的狗身上表现如何；我们还会检查难的情况：身体被遮挡、关节缺失、多人多狗、没有声音。整个过程遵守一条规则：验证集只用来选模型，测试集只在最后用一次。",
        None),
    12: (Z, "14:20–15:00 (~40 s) → Q&A 15:00–20:00",
        "Let me sum up with three points. First, the skeleton connects the whole system. Second, we test every step with a clear experiment. Third, we split the data by dog and by video, so there is no data leakage. Our goal is simple. We want pet cameras to move from \"where is my dog?\" to \"what is my dog doing, and with whom?\" That is the end of our presentation. Thank you. We are happy to take your questions.",
        "最后总结三点。第一，骨架把整个系统连接起来。第二，每一步都有清楚的实验来验证。第三，数据按狗和视频划分，没有数据泄漏。我们的目标很简单：让宠物摄像头从“我的狗在哪？”走向“我的狗在做什么、和谁在一起？”。我们的介绍到此结束，谢谢大家，欢迎提问。",
        "Q&A 5 min: background, data and dog pose → Sun Wenjing; models and evaluation → Zhao Jiayi. / 背景、数据、狗姿态的问题由 Sun Wenjing 回答；模型和评估的问题由 Zhao Jiayi 回答。"),
}


def set_lang(run):
    rpr = run._r.get_or_add_rPr()
    rpr.set("lang", "en-US")


def fill(shape, paras):
    paras = [paras] if isinstance(paras, str) else paras
    tf_paras = shape.text_frame.paragraphs
    assert len(paras) <= len(tf_paras), (shape.name, len(paras), len(tf_paras))
    for p, new in zip(tf_paras, paras):
        runs = p.runs
        texts = [new] if isinstance(new, str) else new
        for r, t in zip(runs, texts):
            r.text = t
            set_lang(r)
        for r in runs[len(texts):]:
            r._r.getparent().remove(r._r)
    for p in tf_paras[len(paras):]:
        p._p.getparent().remove(p._p)


def notes_text(i, speaker, window, en, zh, tip):
    parts = [f"【Slide {i} / 12 · 第 {i} 页】 Speaker 主讲: {speaker} · {window}", f"[EN] {en}", f"[中文] {zh}"]
    if tip:
        parts.append(f"[Tip 提示] {tip}")
    return "\n\n".join(parts)

def to_light(slide, light_layout):
    """Restyle a DIAGRAM_DARK slide like the DIAGRAM_LIGHT ones (white background)."""
    for rel in slide.part.rels.values():
        if rel.reltype.endswith("/slideLayout"):
            rel._target = light_layout.part
    for sh in list(slide.shapes):
        if sh.name in ("r64", "r99", "r134"):  # white panel behind the left text
            sh._element.getparent().remove(sh._element)
            continue
        caption = sh.height is not None and abs(sh.top - Inches(6.62)) < Inches(0.05)
        for clr in sh._element.iter("{http://schemas.openxmlformats.org/drawingml/2006/main}schemeClr"):
            v = clr.get("val")
            clr.set("val", {"bg1": "tx1", "bg2": "accent3", "accent2": "accent1",
                            "accent4": "tx2" if caption else "accent3"}.get(v, v))


def main():
    prs = Presentation(SRC)
    light = next(l for l in prs.slide_layouts if l.name == "DIAGRAM_LIGHT")
    for i, slide in enumerate(prs.slides, 1):
        if slide.slide_layout.name == "DIAGRAM_DARK":
            to_light(slide, light)
        shapes = {s.name: s for s in slide.shapes}
        for name, paras in TEXT.get(i, {}).items():
            fill(shapes[name], paras)
            if (i, name) in SIZE:
                for p in shapes[name].text_frame.paragraphs:
                    for r in p.runs:
                        r.font.size = Pt(SIZE[(i, name)])
        if i == 2:  # "Understand" is wider than 看懂: widen into the right margin
            shapes["t26"].width = Inches(4.2)
        for name, sh in shapes.items():
            if name in IMAGES:
                blip = sh._element.blipFill.blip
                old = blip.rEmbed
                _, new = slide.part.get_or_add_image_part(str(HERE / "images" / IMAGES[name]))
                blip.rEmbed = new
                slide.part.drop_rel(old)
                sh._element.nvPicPr.cNvPr.set("descr", IMAGES[name].removesuffix(".png"))
        slide.notes_slide.notes_text_frame.text = notes_text(i, *NOTES[i])
    prs.core_properties.title = "Indoor Dog Behavior & Human–Dog Interaction Recognition — Group 28 Proposal"
    prs.save(OUT)

    # self-check: no Chinese left on slides; every slide has bilingual notes
    import re
    cjk = re.compile(r"[一-鿿]")
    for i, slide in enumerate(Presentation(OUT).slides, 1):
        for s in slide.shapes:
            if s.has_text_frame:
                assert not cjk.search(s.text_frame.text), (i, s.name, s.text_frame.text)
        n = slide.notes_slide.notes_text_frame.text
        assert "[EN]" in n and "[中文]" in n, i
    print("wrote", OUT)


if __name__ == "__main__":
    main()
