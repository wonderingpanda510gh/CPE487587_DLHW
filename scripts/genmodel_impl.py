import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import zipfile
import io
import os
import argparse
import pandas as pd
from tqdm import tqdm
import torchvision.utils as vutils
from cpe487587hw import GenModelTrainer, metrics
import matplotlib.pyplot as plt

# use hw04 1.2 to read the dataset
class CelebAZipDataset(Dataset):
    def __init__(self, zip_path, transform=None):
        self.zip_path = zip_path
        self.transform = transform
        with zipfile.ZipFile(zip_path, 'r') as zf:
            self.image_names = sorted([
                name for name in zf.namelist()
                if name.lower().endswith(('.jpg', '.jpeg', '.png'))
            ])

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        with zipfile.ZipFile(self.zip_path, 'r') as zf:
            with zf.open(self.image_names[idx]) as f:
                img = Image.open(io.BytesIO(f.read())).convert('RGB') [cite: 193]
        
        if self.transform:
            img = self.transform(img)
        return img

def save_onnx_model(trainer, model_type, path, device):
    if model_type == "GAN":
        model = trainer.netG
        dummy_input = torch.randn(1, 100, 1, 1, device=device)
    elif model_type == "VAE":
        model = trainer.model.decoder
        dummy_input = torch.randn(1, 128, device=device)
    else: # Diffusion 示例
        model = trainer.model
        dummy_input = (torch.randn(1, 3, 64, 64, device=device), torch.tensor([0], device=device))
    
    torch.onnx.export(model, dummy_input, path, opset_version=11)

# here is the sample images, used for evaluation
def sample_images(trainer, model_type, device):
    trainer.set_eval()
    with torch.no_grad():
        if model_type == "GAN":
            noise = torch.randn(25, 100, 1, 1, device=device)
            return trainer.netG(noise)
        elif model_type == "VAE":
            z = torch.randn(25, 128, device=device)
            return trainer.model.decoder(trainer.model.decoder_input(z).view(-1, 128, 8, 8))
        elif model_type == "Diffusion":
            x = torch.randn(25, 3, 64, 64, device=device)
            for t in reversed(range(1000)):
                x = x - 0.01 * trainer.model(x, torch.full((25,), t, device=device))
            return torch.tanh(x)

# plot the bar chart for the metrics
def plot_metrics(df, model_name, save_dir):
    means = df.mean()
    metrics_names = means.index.tolist()
    values = means.values.tolist()

    plt.figure(figsize=(10, 6))
    # we have five metrics, so we can use different colors for each bar, and we can also add the value on top of each bar
    bars = plt.bar(metrics_names, values, color=['#1072BD', '#77AE43', '#EDB021', '#D7592C', '#7F31BD'])
    
    # annotate the value on the top of each bar
    for bar in bars:
        top = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, top, round(top, 4), va='bottom', ha='center')

    plt.title(f"average metrics value for {model_name}")
    plt.ylabel("value")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    save_path = os.path.join(save_dir, f"{model_name}_metrics_report.pdf")
    plt.savefig(save_path)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="HW04: Generative Modeling")
    parser.add_argument('--model_type', type=str, required=True, choices=['VAE', 'GAN', 'Diffusion'], help='choose the model type to train')
    parser.add_argument('--zip_path', type=str, default='/data/CPE_487-587/img_align_celeba.zip', help='dataset zip file path from hw04 1.2')
    parser.add_argument('--epochs', type=int, default=20, help='number of training epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--onnx_interval', type=int, default=5, help='save ONNX model every x epochs')
    parser.add_argument('--train_ratio', type=float, default=0.8, help='training set ratio')
    parser.add_argument('--save_dir', type=str, default='./results_genmodel', help='directory to save results')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='learning rate')
    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    transform = transforms.Compose([
        transforms.Resize(64), 
        transforms.CenterCrop(64), 
        transforms.ToTensor(), 
        transforms.Normalize([0.5]*3, [0.5]*3) # normalize to [-1, 1] for GAN and Diffusion
    ])

    full_dataset = CelebAZipDataset(zip_path=args.zip_path, transform=transform) [cite: 222]
    
    # split the dataset into training and validation sets, according to the hw04, we need to manually pass the training ratio, so I use the arg.train_ratio
    train_size = int(args.train_ratio * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, _ = torch.utils.data.random_split(full_dataset, [train_size, val_size])

    dataloader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=4, 
        pin_memory=True
    )

    # begin to train the model
    print("********HW04: Generative Modeling********")
    print("Begin our training!!!!!!!!!!!!!!")
    print("cuda available:", torch.cuda.is_available())
    trainer = GenModelTrainer(model_type=args.model_type, learning_rate=args.learning_rate, device=args.device)
    
    print(f"Begin training {args.model_type}, device: {args.device}")
    for epoch in range(args.epochs):
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{args.epochs}")
        for step, images in enumerate(pbar):
            stats = trainer.train_step(images, epoch, step)
            pbar.set_postfix(stats)

        # save ONNX model at specified intervals 
        if (epoch + 1) % args.onnx_interval == 0:
            onnx_path = os.path.join(args.save_dir, f"{args.model_type}_epoch_{epoch+1}.onnx")
            save_onnx_model(trainer, args.model_type, onnx_path, args.device)

    # 25 images for evaluation
    print("End of training, begin sampling and evaluation!!!!!!!!!!!!!!")
    gen_imgs = sample_images(trainer, args.model_type, args.device)
    
    # save the generated images for visualization
    vutils.save_image(gen_imgs, os.path.join(args.save_dir, f"{args.model_type}_samples.png"), nrow=5, normalize=True)

    # compute the metrics
    metrics_results = []
    for i in range(25):
        m = metrics(gen_imgs[i])
        metrics_results.append(m)
    
    # save the metrics results to a csv file
    df = pd.DataFrame(metrics_results)
    print(f"Model name: {args.model_type}, Average metrics: \n", df.mean())
    plot_metrics(df, args.model_type, args.save_dir)
    df.mean().to_csv(os.path.join(args.save_dir, f"{args.model_type}_metrics.csv"))

    print("All done! Results saved in:", args.save_dir)

if __name__ == "__main__":
    main()