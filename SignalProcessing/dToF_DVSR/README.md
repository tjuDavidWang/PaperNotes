# 论文-DVSR

> **Citation：**
*Sun, Z., Ye, W., Xiong, J., Choe, G., Wang, J., Su, S., & Ranjan, R. (2023). Consistent Direct Time-of-Flight Video Depth Super-Resolution. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (pp. 5075-5085).*

**论文链接**：[https://arxiv.org/pdf/2211.08658.pdf](https://arxiv.org/pdf/2211.08658.pdf)

**代码仓库**：https://github.com/facebookresearch/DVSR/

## 综述部分

前人经验：
**[Single Frame]**
assume high resolution spatial information (e.g. high resolution sampling positions) or simplified image formation models (e.g. bilinear downsampling)
=>geometric distortions and flying pixels,temporal jittering

**[Depth enhancement algorithms]**
convert a degraded depth map into a high-quality one

1. depth completion
可靠像素识别->信息传播(插值算法,结合其他信息)->深度图回复恢复
2. depth super-resolution
pixel-to-pixel mapping and learn it at test-time(low-resolution depth map is generated with a weightedaverage sampler (i.e., bilinear downsampling))
选择新的分辨率->计算新像素的位置->双线性插值(考虑上下左右四个像素)
对深度图像进行下采样(减少噪声和不必要的细节)->通过一些算法（如双线性插值、深度学习等）进行上采样（恢复出高分辨率的深度图像）
<= physical image formation model & dToF histogram information 
the histogram contains a distribution of depth values within each low-resolution pixel (not a single depth value)

**[Depth Video Processing]**
video provides two more information:
multi-view stereo & temporal correlation between neighboring frames

**[exctract the muti-view geometry]**

1. from a monocular RGB video
2. self-supervised depth estimation
=> epipolar constraint does not hold in dynamic environments
在动态环境中，难以通过对极线约束来找到一个相机视图中的点在另一个相机视图中的对应点，此时需要光流估计或者物体检测，来识别和处理这些动态对象。
=> dynamic objects need to be filtered out in the estimation pipeline

**[align multiple frames]**

1. use a ConvLSTM structure to fuse concatenated frames without alignment.
2. explicitly align multiple frames with a pre-trained scene flow estimator in a stereo video
=>The performance of these algorithms is largely limited by the inefficient or inaccurate multi-frame alignment module
<= a dToF video super-resolution framework(make DVSR agnostic to static or dynamic environments)

**[dToF processing modes]**

1. peakfitting detection mode
Only the peak depth value with strongest signal is sent to the postprocessing network.
2. histogram mode
more information contained in the histogram is utilized.
=> the lateral spatial information is only known to a low resolution (e.g. 16× lower than desirable).
(dToF会记录空间中所有的反射光，会有深度的某种混合或平均，带来patial ambiguity，因此无法准确的检测出物体的移动，相比之下，iToF传感器通常具有更高的空间分辨率，因为它们使用的是相位测量技术。)

## 论文思想

1. information aggregation between multiple frames in an RGB-dToF video
2. dToF histogram information
3. DyDToF: the first synthetic dataset with physics-based dToF sensor simulations and diverse dynamic objects

**【DVSR】**
[input]  a sequence of T frames(RGB & dToF data after downsampling)
[output] a sequence of H × W high-resolution depth maps.
[Network] recurrent manner (use time info) where multi-frame information is propagated either forward-only(只使用之前的帧) or bidirectionally(同时使用之前和之后的帧进行处理).
[two-stage processing for each frame]

1. Initial Stage: 
dToF sensor data + RGB guidance =an initial high-resolution depth prediction and a confidence map.
2. Refinement Stage: 
first stage processing results + the dToF sensor data = the second depth prediction and confidence map
3. Fusion: initial depth + second depth =<confidence maps>= final prediction
   
    feature extractor: 从输入数据中提取有用的特征
    decoder: 将特征提取器提取出的特征转化为最终的深度预测
    a multi-frame propagation module: 在时间上传播信息，利用前一帧（或多帧）的信息来帮助处理当前帧
    a fusion backbone: 整合多帧传播模块和特征提取器提取出的特征，以生成最终的深度预测
    
- monocular depth video processing algorithms pose “hard” epipolar constraints to extract multi-view geometry.
  在处理图像或视频数据时，严格应用对极几何的约束条件。在实际应用中，由于各种因素（如摄像头的校准误差、场景的动态变化等），对极约束可能无法完全满足，导致错误匹配。
    1. a pre-trained optical flow estimator: not posting supervision on the estimated flow（*this approach suffers from inaccurate flow estimations and the fundamental problem of foreground-background mixing*）
      **光流估计**是一种计算图像序列中像素或特征点运动的技术。它基于的主要假设是图像的亮度在时间上保持恒定。换句话说，如果一个像素点在一帧图像中移动到了下一帧的另一个位置，那么这两个位置的像素值应该是相同或相似的。(计算图像梯度 -> 计算光流向量)
    2. a deformable convolution module: after the optical flow-based warping to pick multiple candidates for feature aggregation
      **可变形卷积**是一种特殊的卷积操作，它可以在空间上进行动态的调整，以适应图像中的变形和运动。这是通过在卷积操作中引入可学习的偏移来实现的。卷积核的滑动可以根据图像的内容进行调整。这意味着，如果图像中有一些变形或运动，可变形卷积可以自动地调整自己，以更好地适应这些变化。
      => easily generalize to both static and dynamic environments
      => the correspondence detection between frames does not need to be accurate

**【HVSR】**

1. compression operations on the temporal dimension of the histogram
<= 1. threshold the histogram to remove the signal below the noise floor
<= 2. uniformly divided into M sections, and within each section, the peak is detected
<= 3. rebin the histogram into 2M time bins defined by the section boundaries and peaks
2. utilize the compressed histogram
<= 1. the M detected peaks are concatenated as input to the network in both stages in DVSR
<= 2. compute a histogram matching error to facilitate the confidence predictions

**【Metrics】**

We evaluate the depth super-resolution results on three metrics: **per-frame absolute error** (AE) (lower better), **per-frame *δτ* metric** (higher better), and **temporal end-point error (TEPE)** (lower better).

可以考虑通过x-t切片是否clean，来看temporal stabilities

## 训练过程

We train the proposed dToF depth and histogram video super-resolution networks on **TarTanAir**, a large scale RGB-D video dataset. We use 14 out of 18 scenes for training. We simulate the dToF raw data from the ground truth depth map following the image formation mode

185k frames