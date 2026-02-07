from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
import torch.onnx
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_recall_fscore_support, accuracy_score


class SimpleNN(nn.Module):
    def __init__(self, in_features, num_classes):
        super().__init__()
        self.in_features = in_features
        self.num_classes = num_classes

        self.fc1 = nn.Linear(in_features, 3)
        self.fc2 = nn.Linear(3, 4)
        self.fc3 = nn.Linear(4, 5)
        self.fc4 = nn.Linear(5, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x


class ClassTrainer():
    def __init__(self, X_train, Y_train, eta, epoch, loss, optimizer, model, device, outdir="results", keyword="hw02", stamp=None):
        self.device = device
        self.X_train = torch.as_tensor(X_train, dtype=torch.float32, device=self.device)
        self.Y_train = torch.as_tensor(Y_train, dtype=torch.long, device=self.device).view(-1)
        self.eta = eta
        self.epoch = epoch
        self.loss = nn.CrossEntropyLoss()
        self.outdir = outdir
        self.keyword = keyword
        self.stamp = stamp
        self.file_prefix = str(Path(self.outdir) / f"{self.keyword}_{self.stamp}")
        
        self.model = model.to(self.device)
        if optimizer == "adam":
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.eta)
        else:
            self.optimizer = optim.SGD(self.model.parameters(), lr=self.eta)

        # initializer as empty torch Tensors with length ofepochs
        self.loss_vector = torch.empty(self.epoch, device="cpu")
        self.accuracy_vector = torch.empty(self.epoch, device="cpu")

        self.last_test_pred = None
        self.last_train_pred = None
        self.last_test_true = None

    def train(self):
        self.model.train()
        for ep in range(self.epoch):
            neural_output = self.model(self.X_train)
            loss = self.loss(neural_output, self.Y_train)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # training accuracy
            with torch.no_grad():
                pred = torch.argmax(neural_output, dim=1)
                acc = (pred == self.Y_train).float().mean()

            if ep % 100 == 0:
                print(f"Epoch {ep+1}/{self.epoch}, Loss: {loss.item():.4f}, Accuracy: {acc.item():.4f}")

            # record loss and accuracy
            self.loss_vector[ep] = float(loss.detach().cpu())
            self.accuracy_vector[ep] = float(acc.detach().cpu())

        with torch.no_grad():
            final_train_output = self.model(self.X_train)
            self.last_train_pred = final_train_output.argmax(dim=1).detach().cpu()
  
        return self.loss_vector, self.accuracy_vector
    
    def test(self, X_test, Y_test):
        X_test = torch.as_tensor(X_test, dtype=torch.float32, device=self.device)
        Y_test = torch.as_tensor(Y_test, dtype=torch.long, device=self.device)
        self.model.eval()
        with torch.no_grad():
            neural_output = self.model(X_test)
            loss = self.loss(neural_output, Y_test)

            pred = torch.argmax(neural_output, dim=1)
            acc = (pred == Y_test).float().mean()

        print(f"Test Loss: {loss.item():.4f}, Test Accuracy: {acc.item():.4f}")

        # record the last test prediction and true labels
        self.last_test_pred = pred.detach().cpu()
        self.last_test_true = Y_test.detach().cpu()
        
        return loss.detach().cpu(), acc.detach().cpu()
    
    def predict(self, X):
        self.model.eval()
        with torch.no_grad():
            neural_output = self.model(X)
            pred = torch.argmax(neural_output, dim=1)
        return pred.detach().cpu()

    def save(self, file_name = "cpe487587_simpleNN.onnx"):
        dummy_input = torch.randn(1, self.model.in_features, device=self.device)
        torch.onnx.export(self.model, dummy_input, file_name, input_names=["input"], output_names=["logits"],export_params=True)

    def evaluate(self):
        # first plot the training loss and training accurancy
        epochs = range(len(self.loss_vector))

        fig, ax1 = plt.subplots()

        # left y-axis: loss
        ax1.plot(epochs, self.loss_vector.numpy(), color="blue", label="Loss")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss", color="blue")
        ax1.tick_params(axis="y", labelcolor="blue")

        # right y-axis: accuracy
        ax2 = ax1.twinx()
        ax2.plot(epochs, self.accuracy_vector.numpy(), color="orange", label="Accuracy")
        ax2.set_ylabel("Accuracy", color="orange")
        ax2.tick_params(axis="y", labelcolor="orange")
        fig.suptitle("Training Loss and Accuracy")

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

        plt.tight_layout()
        plt.savefig(f"{self.file_prefix}_training_loss_acc.pdf", dpi=200, bbox_inches="tight")
        plt.close()

        metrics = {}
        # confusion matrix for training set
        cm = confusion_matrix(self.Y_train.detach().cpu().numpy(), self.last_train_pred.numpy())
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(values_format="d", xticks_rotation=45)
        plt.title("Training Confusion Matrix")
        plt.tight_layout()
        plt.savefig(f"{self.file_prefix}_cpe487587hw02_cm_train.pdf", dpi=200, bbox_inches="tight")
        plt.close()

        # compute precision, recall, f1-score, accuracy for training set
        precision, recall, f1_score, _ = precision_recall_fscore_support(self.Y_train.detach().cpu().numpy(), self.last_train_pred.numpy(), average='weighted')
        accuracy = accuracy_score(self.Y_train.detach().cpu().numpy(), self.last_train_pred.numpy())
        metrics["train_precision"] = precision
        metrics["train_recall"] = recall
        metrics["train_f1"] = f1_score
        metrics["train_accuracy"] = accuracy

        print(f"Training Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1_score:.4f}, Accuracy: {accuracy:.4f}")

        # compute confusion matrix for test set
        cm_t = confusion_matrix(self.last_test_true.numpy(), self.last_test_pred.numpy())
        disp_t = ConfusionMatrixDisplay(confusion_matrix=cm_t)
        disp_t.plot(values_format="d", xticks_rotation=45)
        plt.title("Test Confusion Matrix")
        plt.tight_layout()
        plt.savefig(f"{self.file_prefix}_cpe487587hw02_cm_test.pdf", dpi=200, bbox_inches="tight")
        plt.close()

        # compute precision, recall, f1-score, accuracy for test set
        precision_t, recall_t, f1_score_t, _ = precision_recall_fscore_support(self.last_test_true.numpy(), self.last_test_pred.numpy(), average='weighted')
        accuracy_t = accuracy_score(self.last_test_true.numpy(), self.last_test_pred.numpy())
        metrics["test_precision"] = precision_t
        metrics["test_recall"] = recall_t
        metrics["test_f1"] = f1_score_t
        metrics["test_accuracy"] = accuracy_t

        print(f"Test Precision: {precision_t:.4f}, Recall: {recall_t:.4f}, F1-Score: {f1_score_t:.4f}, Accuracy: {accuracy_t:.4f}")

        return metrics