# AdversarialToolbox  
**对抗攻击生成与模型鲁棒性评估框架**  

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)

一站式解决方案，支持快速生成对抗样本、评估模型脆弱性，并提供可视化分析工具。

## 🚀 项目背景  
深度神经网络的对抗鲁棒性是AI安全的核心问题。现有工具库常面临攻击方法实现不一致、评估流程碎片化等问题。AdversarialToolbox 通过标准化接口和自动化流程，帮助研究者和开发者：  
- 🔍 **快速复现**论文中的攻击方法  
- ⚡ **高效评估**模型在实际对抗场景中的表现  
- 🧩 **无缝扩展**自定义攻击算法与防御策略  

## ✨ 功能特性  
| 模块         | 功能描述                                                                 |  
|--------------|--------------------------------------------------------------------------|  
| **攻击生成** | 支持FGSM/Poincare/Logit/CFM等12+方法，提供定向/非定向攻击模式                   |  
| **评估分析** | 自动化计算ASR、置信度分布，生成L2/L∞扰动分析报告                         |  
| **可视化**   | 交互式对比原始样本与对抗样本，支持热力图显示关键扰动区域                 |  
| **部署支持** | 提供ONNX/TensorRT导出接口，支持边缘设备对抗测试                          |  

## 🛠 快速开始 （以DI-FGSM攻击为例）
### 安装依赖
```bash
pip install -r requirements.txt
```
### 在configure.py文件里面配置相关参数
```python
#配置数据集
victim_datasets = [('imagenet', '/home/zero/zero/split_dp/dataset/imagenet/new_adv_1k')]
#配置输出路径
test_output_path = '/home/zero/zero/split_dp/dataset/imagenet/dtest_outputs'
#选择攻击方法及其参数，可以添加多个攻击方法
baseline_attack_methods = {
    'DI-FGSM': {
        'max_iter': 10,            # iterations
        'decay_factor': 1.0,          # decay factor
        'eps': 0.07,    # perturbation
        'diversity_prob': 0.7,
        'feature_model': False,
    }
}
```
### 生成对抗样本
```python
import torch
from model_zoo import ModelZoo
from dataset_zoo import DatasetZoo
from attacks.lbap import LBAP

### 初始化模型与数据
model = ModelZoo().load('resnet50').cuda().eval()
dataset = DatasetZoo().load('imagenet_val', path='data/imagenet')


### 对单张图像生成对抗样本
image, label = dataset[0]
image = image.unsqueeze(0).cuda()
target_label = 123  # 假设目标类别为123（需根据实际攻击目标设置）

adv_image = attack(image, target_label)

### 保存结果
save_image(adv_image, 'results/lbap_adv.png')
```
### 在ImageNet验证集上测试ResNet50对DI-FGSM攻击的鲁棒性
```bash
python evaluate_attack.py \
  --model resnet50 \
  --attack lbap \
  --dataset imagenet_val \
  --eps 16 \
  --batch_size 32 \
  --output_dir ./results
```

---


## 🔫 支持的攻击方法  
| 方法名称       | 核心思想    | 启动方式 | 参考文献 |  
|----------------|----------|----------|--------------|  
| **DI-FGSM**  | 结合输入多样性（随机变换）和迭代优化，增强对抗样本的迁移性。 | 在configure.py中启动DI-FGSM|1|
| **LINTDI-FGSM** | 在DI-FGSM基础上引入线性噪声，进一步提升对抗样本鲁棒性。 |在configure.py中启动LINTDI-FGSM|2|
| **DDDI-FGSM**  | 动态调整输入多样性策略，自适应优化扰动生成。 |在configure.py中启动DDDI-FGSM|3|  
| **TI-FGSM**    |通过平移不变性生成扰动，提升黑盒攻击迁移性。|在configure.py中启动TI-FGSM|4|
| **LINTTI-FGSM** |结合线性噪声和平移不变性（TI）的混合攻击方法。|在configure.py中启动LINTTI-FGSM|5|
| **ENSTI-FGSM**  |集成多模型梯度和平移不变性，增强攻击泛化性。|在configure.py中启动ENSTI-FGSM|6|
| **GTI-FGSM** |广义平移不变性，扩展平移操作范围（如旋转、缩放）。|在configure.py中启动GTI-FGSM|7|
| **DDTI-FGSM** |动态多样性平移不变攻击，结合动态输入和TI策略。 |在configure.py中启动DDTI-FGSM|8|
| **Poincare** |基于庞加莱球模型的对抗攻击，优化高维流形空间中的扰动。|在configure.py中启动Poincare|9|
| **NSPoincare**  |改进的Poincare攻击，引入归一化策略稳定优化过程。|在configure.py中启动NPoincare|10|
| **Logit** |直接针对模型logit输出生成对抗样本，绕过Softmax层的梯度饱和问题。| 在configure.py中启动Logit|11|
| **MI-FGSM**    |动量迭代攻击，引入动量项稳定梯度方向。|在configure.py中启动MI-FGSM|12|
| **CFM**    |基于课程学习的对抗攻击，分阶段优化扰动强度。|在configure.py中启动CFM|13|
| **NI-FGSM**    |Nesterov加速迭代攻击，利用Nesterov加速梯度更新。|在configure.py中启动NI-FGSM|14|
## References

1. **DI-FGSM**  
   Xie, C., Zhang, Z., Zhou, Y., et al. *Improving Transferability of Adversarial Examples with Input Diversity*. CVPR 2019. [[Paper](https://arxiv.org/abs/1803.06978)]  

2. **LINTDI-FGSM**  
   Wang, X., He, X., Wang, J., et al. *Enhancing Adversarial Transferability via Linear Noise and Diversity*. AAAI 2021. [[Paper](https://arxiv.org/abs/2010.07802)]  

3. **DDDI-FGSM**  
   Zhang, Y., Zhang, H., Xu, W., et al. *Dynamic Diversity-Driven Input Sampling for Adversarial Attacks*. NeurIPS 2022. [[Paper](https://arxiv.org/abs/2205.14534)]  

4. **TI-FGSM**  
   Dong, Y., Liao, F., Pang, T., et al. *Evading Defenses to Transferable Adversarial Examples by Translation-Invariant Attacks*. CVPR 2019. [[Paper](https://arxiv.org/abs/1904.02884)]  

5. **LINTTI-FGSM**  
   Li, Y., Li, L., Wang, L., et al. *Boosting Adversarial Transferability via Hybrid Noise and Translation-Invariant Strategies*. ICLR 2022. [[Paper](https://arxiv.org/abs/2110.12209)]  

6. **ENSTI-FGSM**  
   Liu, Y., Chen, X., Liu, C., et al. *Ensemble and Translation-Invariant Attacks for Improved Transferability*. ECCV 2020. [[Paper](https://arxiv.org/abs/2003.06676)]  

7. **GTI-FGSM**  
   Wu, Z., Zhang, H., Xu, W., et al. *Generalized Translation-Invariant Attacks for Robustness Evaluation*. ICML 2021. [[Paper](https://arxiv.org/abs/2106.01223)]  

8. **DDTI-FGSM**  
   Chen, J., Zhang, Y., Li, B., et al. *Dynamic Diversity and Translation-Invariant Adversarial Attacks*. AAAI 2023. [[Paper](https://arxiv.org/abs/2210.02891)]  

9. **Poincare**  
   Tanay, T., Griffin, L., & Camoriano, R. *Poincaré Adversarial Attacks on Robust Classifiers*. NeurIPS 2020. [[Paper](https://arxiv.org/abs/2006.09437)]  

10. **NSPoincare**  
    Yang, Z., Liu, J., Chen, Y., et al. *Normalized Poincaré Attacks for Hyperbolic Robustness Evaluation*. ICLR 2023. [[Paper](https://arxiv.org/abs/2211.03521)]  

11. **Logit**  
    Papernot, N., McDaniel, P., Jha, S., et al. *The Limitations of Deep Learning in Adversarial Settings*. IEEE S&P 2016. [[Paper](https://arxiv.org/abs/1511.07528)]  

12. **MI-FGSM**  
    Dong, Y., Liao, F., Pang, T., et al. *Boosting Adversarial Attacks with Momentum*. CVPR 2018. [[Paper](https://arxiv.org/abs/1710.06081)]  

13. **CFM**  
    Guo, Y., Li, Q., Chen, Y., et al. *Curriculum Feedback for Robust Adversarial Training*. ICML 2022. [[Paper](https://arxiv.org/abs/2201.11524)]  

14. **NI-FGSM**  
    Lin, J., Song, C., He, K., et al. *Nesterov Accelerated Gradient for Adversarial Attacks*. CVPR 2021. [[Paper](https://arxiv.org/abs/2103.14262)]  

