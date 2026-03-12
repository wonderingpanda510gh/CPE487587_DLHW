import argparse
import os
import glob
import numpy as np
from PIL import Image
import onnxruntime as ort
from datasets import Dataset, concatenate_datasets
import torchvision.transforms as transforms


def parse_args():
    p = argparse.ArgumentParser(description="ImageNet ONNX inference")
    p.add_argument("--model_path", type=str, default="results/imagenet_model.onnx")
    p.add_argument("--image_path", type=str, help="path of the input image")
    return p.parse_args()


def load_class_names():
    # load the image class
    # we can directly load the class from the server, as we did in the training part
    base_dir = "/data/CPE_487-587/imagenet-1k/ILSVRC___imagenet-1k/default/0.0.0/49e2ee26f3810fb5a7536bbf732a7b07389a47b5"

    train_files = sorted(glob.glob(os.path.join(base_dir, "imagenet-1k-train-*.arrow")))
    if not train_files:
        return None

    # only need one image to test the imagenet
    ds = Dataset.from_file(train_files[0])

    if "label" in ds.features and hasattr(ds.features["label"], "names"):
        return ds.features["label"].names

    return None


def train_transform():
    # do the transforms for the imagenet data, as we did in the training part
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        )
    ])


def preprocess_image(image_path):
    # load the image and apply the same transforms as in the training part
    image = Image.open(image_path).convert("RGB")
    transform = train_transform()
    x = transform(image)              
    x = x.unsqueeze(0).numpy()        
    x = x.astype(np.float32)
    return x


def main():
    args = parse_args()

    # first we need to load the onnx model
    model = ort.InferenceSession(args.model_path, providers=["CUDAExecutionProvider"])

    # get the input and output names of the model
    input_name = model.get_inputs()[0].name
    output_name = model.get_outputs()[0].name

    # preprocess the input image and convert it by using the same transforms as in the training part
    x = preprocess_image(args.image_path)

    # run the model
    outputs = model.run([output_name], {input_name: x})
    output_predictions = outputs[0]                    
    pred_id = int(np.argmax(output_predictions, axis=1)[0])

    # load the class names and print the predicted class name
    class_names = load_class_names()

    print(f"The path of input testing image: {args.image_path}")
    print(f"The output of model prediction id: {pred_id}")

    if class_names is not None and pred_id < len(class_names):
        full_label = class_names[pred_id]
        primary_name = full_label.split(",")[0].strip()
        print(f"The predicted class name is: {primary_name}")
        print(f"The full label is: {full_label}")


if __name__ == "__main__":
    main()