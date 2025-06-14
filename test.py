import os
import pandas as pd
import torch
from data.cifar10_noise_loader import get_dataloaders
from models.custom_resnet import CustomResNet

def check_gpu_available():
    print("Checking GPU availability...")
    if torch.cuda.is_available():
        print(f"✅ GPU available: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠️ No GPU detected. Training will proceed on CPU.")

def check_data_loaders():
    print("🔍 Verifying DataLoaders...")
    train_loader, val_loader = get_dataloaders(batch_size=16, noise_ratio=0.1)
    assert len(train_loader.dataset) == 50000, "Train dataset should contain 50k images"
    assert len(val_loader.dataset) == 10000, "Validation dataset should contain 10k images"
    print("✅ DataLoaders are correctly set up.")

def check_model_forward():
    print("🔍 Testing model with a forward pass...")
    model = CustomResNet()
    dummy_input = torch.randn(2, 3, 32, 32)
    output = model(dummy_input)
    assert output.shape == (2, 10), "Model output should have shape [batch, num_classes]"
    print("✅ Forward pass successful.")

def check_previous_training_metrics():
    if os.path.exists("training_metrics.csv"):
        df = pd.read_csv("training_metrics.csv")
        print(f"📈 Previous training metrics found: {df.shape[0]} epochs recorded.")
    else:
        print("ℹ️ No previous metrics file found.")

if __name__ == "__main__":
    print("🚦 Starting pre-training checks...\n")
    check_gpu_available()
    check_data_loaders()
    check_model_forward()
    check_previous_training_metrics()
    print("\n✅ Ready to start training.")
