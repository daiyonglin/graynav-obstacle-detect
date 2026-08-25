# GrayNav Unified Perception Method

## 1. Problem formulation

GrayNav 将单帧灰度图像记为 \(I\in\mathbb{R}^{1\times384\times384}\)，联合学习目标检测 \(D\)、场景语义 \(S\)、离散相对深度 \(Z\) 与台阶边缘 \(E\)：

\[
(D,S,Z,E)=f_{\theta}(I).
\]

设计目标不是将多个独立网络并排运行，而是在共享特征上提供互补安全证据：目标框负责可命名障碍，场景语义补充墙面、地面和台阶，深度提供近远顺序，边缘分支强化落差结构。

## 2. True-mono detector initialization

检测骨干由官方 COCO YOLOv8n 初始化。对首层 RGB 卷积权重 \(W\in\mathbb{R}^{C\times3\times k\times k}\)，灰度输入在三个通道取相同值时有：

\[
W_R*I+W_G*I+W_B*I=(W_R+W_G+W_B)*I.
\]

因此单通道首层使用：

\[
W_{gray}=W_R+W_G+W_B,
\]

既保持预训练响应，又避免运行时复制三通道。Indoor8 分类行从 COCO 对应类别直接复制，DFL 回归头完整继承。

## 3. Shared multi-task architecture

Mono-YOLOv8n Backbone 和 PAN/FPN 生成 P3、P4、P5。三尺度检测头直接输出 raw classification logits 与 16-bin DFL regression logits，解码不进入 NPU 图。

场景分支将高分辨率 P3 细节和上采样后的 P4 语义投影到相同通道数，经 Add、ReLU 与 depthwise-separable 3×3 卷积融合。最终单个 1×1 卷积生成 21 通道：4 类语义、16 级深度与 1 个台阶边缘响应。该打包输出减少了转换器多输出绑定风险，同时保持 CPU 后处理的独立语义。

SurfaceDepth 初始化来自 E3 epoch49 checkpoint；无法直接对应的轻量兼容层采用 Dirac/单位映射和恒等 BN 初始化，不以无约束随机特征破坏已有表征。

## 4. Supervision

检测监督来自 VOC2007 Indoor8 映射和 COCO128 稀有类重放。对 person 样本自动生成完整人体、无面部上半身、下半身、左右截断和遮挡视图，提高局部人体召回。

场景数据采用：

- ADE20K：`ground_candidate / blocked_surface / step_or_drop / unknown_other`；
- StairNetV3：台阶语义与边缘监督，并将台阶外区域作为 unknown 负监督；
- NYU Depth V2：官方 795/654 划分的稠密深度监督。

总损失为：

\[
\mathcal{L}=\mathcal{L}_{det}+\mathcal{L}_{seg}+0.4\mathcal{L}_{depth}+1.2\mathcal{L}_{edge}.
\]

分割损失由加权交叉熵和任务类 Dice 组成；深度损失组合离散等级、log-depth、SILog、深度梯度、灰度边缘感知平滑与 Near/Mid/Far 分组监督；边缘使用 focal BCE 与 Dice。

## 5. Static deployment contract

模型采用静态 batch 1、NCHW、384×384 输入。ONNX/NPU 图只依赖固定尺寸 Conv、BN、ReLU、Add、Concat、Pool 和 nearest Resize。Softmax、ArgMax、DFL、NMS、动态规划与时序状态均在 CPU 侧执行。

该分割使神经网络图保持量化友好，也允许安全决策以可测试的 C++ 代码实现，而不把不可解释的控制逻辑固化进模型。
