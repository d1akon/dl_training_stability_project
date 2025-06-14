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
git clone https://github.com/your_username/dl-training-stability-project.git
cd dl-training-stability-project
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


