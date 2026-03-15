import os
import json
import argparse
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from cpe487587hw import ACCCruiseDataset, ACCNet, ACCTrainer, DiceLoss


def parse_args():
    p = argparse.ArgumentParser(description="HW03 Q7: ACC classifier")
    p.add_argument("--data_dir", type=str, default="/data/CPE_487-587/ACCDataset")
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--hidden_dim", type=int, default=64)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--split_ratio", type=float, default=0.8)
    p.add_argument("--random_seed", type=int, default=42)
    p.add_argument("--eta", type=float, default=1e-3)
    p.add_argument("--epoch", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--optimizer", type=str, default="ADAM")
    p.add_argument("--loss", type=str, default="dice", choices=["dice"])
    p.add_argument("--device", type=str, default=None, choices=["cpu", "cuda"])
    p.add_argument("--outdir", type=str, default="results_acc")
    p.add_argument("--keyword", type=str, default="hw03q7")
    p.add_argument("--save_onnx", action="store_true")
    p.add_argument("--onnx_name", type=str, default="acc_model.onnx")
    return p.parse_args()


def select_device(device):
    if device == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif device == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def plot_curves(trainer, outdir, keyword):
    epochs = range(1, len(trainer.train_losses) + 1)

    plt.figure()
    plt.plot(epochs, trainer.train_losses, label="Train Loss")
    plt.plot(epochs, trainer.val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(outdir, f"{keyword}_loss.pdf"), bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.plot(epochs, trainer.train_accs, label="Train Acc")
    plt.plot(epochs, trainer.val_accs, label="Val Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(outdir, f"{keyword}_acc.pdf"), bbox_inches="tight")
    plt.close()


def save_norm_stats(stats, outdir, keyword):
    path = os.path.join(outdir, f"{keyword}_norm.json")
    with open(path, "w") as f:
        json.dump(
            {
                "mean": stats.mean.tolist(),
                "std": stats.std.tolist(),
            },
            f,
            indent=2,
        )
    print(f"Saved normalization stats to {path}")


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    device = select_device(args.device)
    print(f"Using device: {device}")

    train_dataset = ACCCruiseDataset(
        root_dir=args.data_dir,
        k=args.k,
        split="train",
        split_ratio=args.split_ratio,
        normalize=True,
        random_seed=args.random_seed,
    )
    val_dataset = ACCCruiseDataset(
        root_dir=args.data_dir,
        k=args.k,
        split="val",
        split_ratio=args.split_ratio,
        normalize=True,
        random_seed=args.random_seed,
        stats=train_dataset.stats,  # use train stats for val normalization
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print(f"Input dim: {args.k + 1}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=(device.type == "cuda"),
    )

    model = ACCNet(input_dim=args.k + 1, hidden_dim=args.hidden_dim, dropout=args.dropout).to(device)

    if args.loss == "dice":
        loss_fn = DiceLoss(1)

    # here is the same optimizer and scheduler setup as in the imagenet_impl
    if args.optimizer.upper() == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=args.eta, momentum=0.9, weight_decay=1e-4)
    elif args.optimizer.upper() == "ADAM":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.eta, weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    trainer = ACCTrainer(
        model=model,
        device=device,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
    )

    trainer.fit(train_loader, val_loader, args.epoch)

    pt_path = os.path.join(args.outdir, f"{args.keyword}_acc.pt")
    torch.save(model.state_dict(), pt_path)
    print(f"Saved PyTorch weights to {pt_path}")

    save_norm_stats(train_dataset.stats, args.outdir, args.keyword)
    plot_curves(trainer, args.outdir, args.keyword)

    if args.save_onnx:
        onnx_path = os.path.join(args.outdir, args.onnx_name)
        trainer.export_onnx(file_name=onnx_path, input_dim=args.k + 1)
        print(f"Saved ONNX model to {onnx_path}")


if __name__ == "__main__":
    main()