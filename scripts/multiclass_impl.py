import argparse
import os
from datetime import datetime

import numpy as np
import polars as pl
import torch

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from cpe487587hw import SimpleNN, ClassTrainer


col_drop = [
    "Flow ID",
    "Source IP", 
    "Source Port",
    "Destination IP", 
    "Destination Port",
    "Protocol",
    "Timestamp",
]

# define possible label column names, to make sure we can find the right label's name
labels = ["Label", "label", "Class", "class", "Category", "category"]

# define all the arguments
def parse_args():
    p = argparse.ArgumentParser(description="HW02 Q8: Multiclass malware classification")
    p.add_argument("--data", type=str, default="data/Android_Malware.csv")
    p.add_argument("--eta", type=float, default=0.01)
    p.add_argument("--epoch", type=int, default=5000)
    p.add_argument("--test_size", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--optimizer", type=str, default="adam")
    p.add_argument("--device", type=str, default=None, choices=["cpu", "cuda"])
    p.add_argument("--keyword", type=str, default="hw02")
    p.add_argument("--outdir", type=str, default="results")
    p.add_argument("--save_onnx", default= True, action="store_true")
    p.add_argument("--onnx_name", type=str, default="multiclass_model.onnx")
    return p.parse_args()


# select device
def select_device(device_name):
    if device_name == "cpu":
        return torch.device("cpu")
    if device_name == "cuda":
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

# find the label column in the dataset
def find_label_col(columns):
    cols = set(columns)
    for l in labels:
        if l in cols:
            return l

# convet the label to number 
def encode_labels(lables):
    lables_new = np.array([str(v).strip() for v in lables], dtype=object)
    classes = sorted(set(lables_new.tolist()))
    lables_id = {c: i for i, c in enumerate(classes)} # mapping from label to id 
    y = np.array([lables_id[v] for v in lables_new], dtype=np.int64)
    return y, classes, lables_id


def main():
    print("********HW02 08: Multiclass Classification********")
    print("\n Begin!!!!!!!!!!!!!!")
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    device = select_device(args.device)

    # load data 
    df = pl.read_csv(args.data, infer_schema_length=10000)

    # cleaning and processing the data
    present_drop = [c for c in col_drop if c in df.columns]
    if present_drop:
        df = df.drop(present_drop)
    label_col = find_label_col(df.columns)

    # get the features and labels, we map the labels from text to number
    lables = df.select(label_col).to_numpy().reshape(-1)
    X = df.drop(label_col).to_numpy().astype(np.float32)
    y, classes, _ = encode_labels(lables)
    num_classes = len(classes)
    in_features = X.shape[1]

    # 80:20 train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=y,
    )

    # normalize the data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_test = scaler.transform(X_test).astype(np.float32)

    # training
    model = SimpleNN(in_features=in_features, num_classes=num_classes)
    loss_function = torch.nn.BCEWithLogitsLoss()

    trainer = ClassTrainer(
        X_train=X_train,
        Y_train=y_train,
        eta=args.eta,
        epoch=args.epoch,
        loss=loss_function,
        optimizer=args.optimizer,
        model=model,
        device=device,
    )

    trainer.train()
    test_loss, test_acc = trainer.test(X_test, y_test)

    # all the metrics and results saving
    stamp = datetime.now().strftime("%Y%m%d%H%M%S")
    out_csv_dir = os.path.join(args.outdir, f"{args.keyword}_{stamp}")

    metrics = trainer.evaluate()

    # record the onnx model
    onnx_path = ""
    if args.save_onnx:
        onnx_path = os.path.join(args.outdir, f"{args.keyword}_{stamp}_{args.onnx_name}")
        trainer.save(file_name=onnx_path)

    # save the metrics as a csv file
    out_csv = f"{out_csv_dir}_metrics.csv"
    row = {
        "keyword": [args.keyword],
        "timestamp": [stamp],
        "in_features": [in_features],
        "num_classes": [num_classes],
        "eta": [args.eta],
        "epoch": [args.epoch],
        "optimizer": [args.optimizer],
        "device": [str(device)],
        "train_accuracy": [metrics.get("train_accuracy", np.nan)],
        "train_precision": [metrics.get("train_precision", np.nan)],
        "train_recall": [metrics.get("train_recall", np.nan)],
        "train_f1": [metrics.get("train_f1", np.nan)],
        "test_accuracy": [metrics.get("test_accuracy", np.nan)],
        "test_precision": [metrics.get("test_precision", np.nan)],
        "test_recall": [metrics.get("test_recall", np.nan)],
        "test_f1": [metrics.get("test_f1", np.nan)],
        "test_loss": [float(test_loss)],
        "onnx_path": [onnx_path],
    }

    pl.DataFrame(row).write_csv(out_csv)

    print("\n********End Training********")
    print("Saved metrics CSV:", out_csv)
    if onnx_path:
        print("Saved ONNX:", onnx_path)


if __name__ == "__main__":
    main()
