# 🌀 Deep Learning Training Stability with Custom ResNet and Focal Loss

This project explores effective training practices in Deep Learning using PyTorch, with a focus on **stability & robustness**.

- A **custom ResNet-like CNN architecture** with residual blocks  
- Use of **Focal Loss** instead of standard cross-entropy. This is particularly useful when dealing with **imbalanced or noisy data**, as it dynamically down-weights easy examples and focuses training on harder ones.
- **He initialization** is applied to convolutional layers to ensure stable and efficient training, especially when using ReLU activations. It helps avoid problems like vanishing or exploding gradients early in training.
- A noisy CIFAR-10 dataset with synthetic label noise to simulate real-world imperfections, helping assess model resilience under suboptimal conditions.

---
