import onnxruntime as ort
import torch
import numpy as np
import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import torchvision.utils as vutils
from gen_model import metrics

def plot_onnx_metrics(avg_series, model_name, save_dir):
    plt.figure()
    names = avg_series.index.tolist()
    values = avg_series.values.tolist()
    
    # same color scheme as the training metrics plot
    bars = plt.bar(metrics_names, values, color=['#1072BD', '#77AE43', '#EDB021', '#D7592C', '#7F31BD'])
    plt.title(f"ONNX inference metrics: {model_name}")
    plt.ylabel("value")
    
    for bar in bars:
        top = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, top, round(top, 4), va='bottom', ha='center')
    
    plt.savefig(os.path.join(save_dir, f"{model_name}_onnx_plot.pdf"))
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="ONNX Inference and Evaluation")
    parser.add_argument('--model_type', type=str, required=True, choices=['VAE', 'GAN', 'Diffusion'])
    parser.add_argument('--onnx_path', type=str, required=True, help='ONNX file path')
    parser.add_argument('--save_dir', type=str, default='./inference_results')
    parser.add_argument('--latent_dim', type=int, default=100)
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    providers = ['CPUExecutionProvider']
    if torch.cuda.is_available():
        providers = ['CUDAExecutionProvider']
    
    session = ort.InferenceSession(args.onnx_path, providers=providers)
    input_name = session.get_inputs()[0].name

    print(f"Open the ONNX file successfully: {args.onnx_path}")

    # here we generate 25 image used for evaluation
    gen_imgs_np = []
    
    for i in range(25):
        if args.model_type == "GAN":
            z = np.random.randn(1, args.latent_dim, 1, 1).astype(np.float32)
        elif args.model_type == "VAE":
            z = np.random.randn(1, 128).astype(np.float32)
        else:
            z = np.random.randn(1, 3, 64, 64).astype(np.float32)
        
        # run onnx inference
        outputs = session.run(None, {input_name: z})
        gen_imgs_np.append(outputs[0])

    # convert the numpy array to tensor
    gen_imgs = torch.from_numpy(np.concatenate(gen_imgs_np, axis=0))

    metrics_list = []
    for i in range(25):
        m = metrics(gen_imgs[i])
        metrics_list.append(m)

    df = pd.DataFrame(metrics_list)
    avg_metrics = df.mean()
    
    vutils.save_image(gen_imgs, os.path.join(args.save_dir, f"{args.model_type}_onnx_samples.png"), nrow=5, normalize=True)
    
    csv_path = os.path.join(args.save_dir, f"{args.model_type}_onnx_metrics.csv")
    avg_metrics.to_csv(csv_path)

    plot_onnx_metrics(avg_metrics, args.model_type, args.save_dir)
    
    print("Finish the ONNX inference and evaluation")
    print(avg_metrics)


if __name__ == "__main__":
    main()