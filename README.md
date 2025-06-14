# 🌀 Deep Learning Training Stability with Custom ResNet and Focal Loss

This project explores effective training practices in Deep Learning using PyTorch, with a focus on **stability & robustness**.

- A **custom ResNet-like CNN architecture** with residual blocks  
- Use of **Focal Loss** instead of standard cross-entropy. This is particularly useful when dealing with **imbalanced or noisy data**, as it dynamically down-weights easy examples and focuses training on harder ones.
- **He initialization** is applied to convolutional layers to ensure stable and efficient training, especially when using ReLU activations. It helps avoid problems like vanishing or exploding gradients early in training.
- A noisy CIFAR-10 dataset with synthetic label noise to simulate real-world imperfections, helping assess model resilience under suboptimal conditions.

---

## 🔷 Running this

### Requirements
- Python 3.10+
- PyTorch 2.x
- torchvision
- scikit-learn
- matplotlib
- tqdm
- pyyaml


### Clone and Run

```bash
git clone https://github.com/d1akon/dl_training_stability_project.git
cd dl_training_stability_project
```

Then run training:
```bash
python train.py
```

This will:
- Download CIFAR-10
- Add label noise
- Train a ResNet-like model with Focal Loss
- Save the best model in `checkpoints/`
- Save metrics in `training_metrics.csv`

To visualize metrics:

```bash
python plot_metrics.py
```

---

### Custom Config

Edit `config.yaml` to adjust:

```yaml
batch_size: 128
learning_rate: 0.001
epochs: 30
noise_ratio: 0.2
gamma: 2.0              #--- Focal Loss parameter
use_augmentation: true
clip_grad_norm: 5.0
```
## Evaluation

Metrics saved per epoch:
- Loss
- Accuracy
- F1 Score (macro)
- Precision & Recall

![image](https://github.com/user-attachments/assets/b4dc071b-003f-431b-abb2-7adb8c297bcf)

## Ⓜ Model Architecture

```
Input: [B, 3, 32, 32]
|
BasicBlock1 -> [B, 32, 32, 32]
|
BasicBlock2 (downsample) -> [B, 64, 16, 16]
|
BasicBlock3 (downsample) -> [B, 128, 8, 8]
|
GlobalAvgPool -> [B, 128, 1, 1]
|
Flatten + Dropout -> [B, 128]
|
Linear -> [B, 10]
```
![image](https://github.com/user-attachments/assets/8d6f3c3d-279d-4312-a389-024748559f25)

And also each BasicBlock internal architecture:

![image](https://github.com/user-attachments/assets/c72757ea-d0f2-4216-ac6d-6b0a6cde5cc2)

## References
- Focal Loss: https://arxiv.org/abs/1708.02002  
- ResNet: https://arxiv.org/abs/1512.03385  


