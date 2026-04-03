from cpe487587hw import ImageNetCNN, CNNTrainer
from datasets import load_dataset
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import Dataset, concatenate_datasets
import glob
import subprocess
from datasets import load_from_disk

# first we need to load the dataset, we will use the imagenet dataset from huggingface datasets library
def load_imagenet_dataset():
    # dataset = load_dataset(
    # "ILSVRC/imagenet-1k",
    # token=True,
    # cache_dir="/data/CPE_487-587/imagenet-1k"
    # )
    # dataset = load_dataset(
    # "ILSVRC/imagenet-1k",
    # cache_dir=os.path.expanduser("~/.cache/huggingface/imagenet"),
    # token=True
    # )
    dataset = load_from_disk("/data/CPE_487-587/imagenet-1k-arrow")

    # base_dir = "/data/CPE_487-587/imagenet-1k/ILSVRC___imagenet-1k/default/0.0.0/49e2ee26f3810fb5a7536bbf732a7b07389a47b5"

    # train_files = sorted(glob.glob(os.path.join(base_dir, "imagenet-1k-train-*.arrow")))
    # val_files = sorted(glob.glob(os.path.join(base_dir, "imagenet-1k-validation-*.arrow")))
    # train_dataset = concatenate_datasets([Dataset.from_file(f) for f in train_files])
    # val_dataset = concatenate_datasets([Dataset.from_file(f) for f in val_files])
    
    train_dataset = dataset['train']
    val_dataset = dataset['validation']

    # this is for the hw03 q6
    print("Original dataset size:")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    num_classes = len(train_dataset.features['label'].names)
    print(f"Number of classes: {num_classes}")

    train_size = int(len(train_dataset) * 0.005) # 10 percent selection
    val_size = int(len(val_dataset) * 0.002) # 5 percent selection
    train_dataset = train_dataset.select(range(train_size))
    val_dataset = val_dataset.select(range(val_size))

    print("Selected dataset size:")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    first_example = train_dataset[0]
    image = first_example['image']
    label_id = first_example['label']
    class_names = train_dataset.features['label'].names
    full_label = class_names[label_id]
    primary_name = full_label.split(',')[0].strip()

    plt.figure()
    plt.imshow(image)
    plt.title(f"ID {label_id}: {primary_name}\n({full_label})", fontsize=10)
    plt.axis('off')
    sample_path = os.path.join("results", "sample_image.pdf")
    os.makedirs("results", exist_ok=True)
    plt.savefig(sample_path, bbox_inches="tight")
    plt.close()

    return train_dataset, val_dataset, num_classes

# apply transforms to the dataset
def apply_train_transforms(image):
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    return train_transforms(image)

def apply_val_transforms(image):
    val_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    return val_transforms(image)

# process the dataset and apply transforms
def preprocess_train(example):
    example["pixel_values"] = [
        apply_train_transforms(img.convert("RGB"))
        for img in example["image"]
    ]
    return example

def preprocess_val(example):
    example["pixel_values"] = [
        apply_val_transforms(img.convert("RGB"))
        for img in example["image"]
    ]
    return example

def collate_fn(batch):
    pixel_values = torch.stack([item["pixel_values"] for item in batch])
    labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)
    return pixel_values, labels

def select_device(device):
    if device == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def plot_training_curves(trainer, outdir, keyword):
    epochs = range(1, len(trainer.train_loss) + 1)

    # loss curve
    plt.figure()
    plt.plot(epochs, trainer.train_loss, label="Train Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    loss_path = os.path.join(outdir, f"{keyword}_loss.pdf")
    plt.savefig(loss_path, bbox_inches="tight")
    plt.close()

    # Accuracy curve
    plt.figure()
    plt.plot(epochs, trainer.train_accuracy, label="Train Accuracy")
    plt.plot(epochs, trainer.val_accuracy, label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    acc_path = os.path.join(outdir, f"{keyword}_accuracy.pdf")
    plt.savefig(acc_path, bbox_inches="tight")
    plt.close()

def parse_args():
    p = argparse.ArgumentParser(description="HW03 Q6: ImageNet classification")
    p.add_argument("--eta", type=float, default=0.01)
    p.add_argument("--epoch", type=int, default=10000)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--input_channels", type=int, default=3)
    p.add_argument("--optimizer", type=str, default="SGD")
    p.add_argument("--device", type=str, default=None, choices=["cpu", "cuda"])
    p.add_argument("--keyword", type=str, default="hw03")
    p.add_argument("--outdir", type=str, default="results")
    p.add_argument("--save_onnx", action="store_true")
    p.add_argument("--onnx_name", type=str, default="imagenet_model.onnx")
    return p.parse_args()

def get_best_gpu(strategy="utilization"):
    """
    Select best GPU by 'utilization' or 'memory'.
    """
    if strategy == "memory":
        # Use PyTorch directly for free memory
        free_mem = []
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.mem_get_info(i) # (free, total)
            free_mem.append(props[0])
        return free_mem.index(max(free_mem))

    elif strategy == "utilization":
        result = subprocess.run(
        ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
        capture_output=True, text=True
        )
        utilizations = [int(x.strip()) for x in result.stdout.strip().split("\n")]
        return utilizations.index(min(utilizations))




def main():
    print("********HW03 06: ImageNet Classification********")
    print("Begin our training!!!!!!!!!!!!!!")
    print("cuda available:", torch.cuda.is_available())
    args = parse_args()
    # device = select_device(args.device)

    # Pick strategy: "utilization" or "memory"
    device_id = get_best_gpu(strategy="utilization")
    device = torch.device(f"cuda:7" if torch.cuda.is_available() else "cpu")
    # print(f"Selected GPU: {device_id}")

    os.makedirs(args.outdir, exist_ok=True)
    print(f"Using device: {device}")

    print("Loading dataset!!!!!!!!")
    train_dataset, val_dataset, num_classes = load_imagenet_dataset()
    
    print("Applying dataset transforms!!!!!!!!")
    train_dataset = train_dataset.with_transform(preprocess_train)
    val_dataset = val_dataset.with_transform(preprocess_val)

    print("Creating data loaders!!!!!!!!")
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True, # Important for faster GPU transfer
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True, # Important for faster GPU transfer
        collate_fn=collate_fn
    )
    print("Initializing ImageNetCNN!!!!!!!!")
    model = ImageNetCNN(args.input_channels, num_classes).to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {num_params}")
    loss_function = nn.CrossEntropyLoss()

    print("Setting optimizer!!!!!!!!")
    if args.optimizer.upper() == "SGD":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=args.eta,
            momentum=0.9,
            weight_decay=1e-4
        )
    elif args.optimizer.upper() == "ADAM":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.eta,
            weight_decay=1e-4
        )
    scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer,
    step_size=30,
    gamma=0.1
    )


    print("Initializing trainer!!!!!!!!")
    trainer = CNNTrainer(
        model=model,
        device=device,
        loss_function=loss_function,
        optimizer=optimizer,
        scheduler=scheduler
    )

    print("Start training!!!!!!!!")
    trainer.fit(train_loader, val_loader, args.epoch)

    print("Plot training curves!!!!!!!!")
    plot_training_curves(trainer, args.outdir, args.keyword)

    print("Save ONNX model!!!!!!!!")
    if args.save_onnx:
        onnx_path = os.path.join(args.outdir, args.onnx_name)
        trainer.save(onnx_path)
        print(f"ONNX model exported to {onnx_path}")

if __name__ == "__main__":
    main()