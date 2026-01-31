from cpe487587hw import deepl
import matplotlib.pyplot as plt
import time
import torch
import os
def main(d, n, learning_rate=0.001, num_epochs=10000):
    """
    this is the main function to run the binary classification, it is used to demonstrate the usage of the binary classification function. Here, learning rate and number of epochs are set to default values 0.001 and 10000.
    
    :param d: number of features
    :param n: number of samples
    """
    # create output directory for pdf and pt files
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "outputs")
    os.makedirs(output_dir, exist_ok=True)

    W1, W2, W3, W4, losses = deepl.binary_classification(d, n, learning_rate, num_epochs)

    # first save the weights
    pt_filepath = os.path.join(output_dir, "trained_weights.pt")
    torch.save(
    {
        "W1": W1,
        "W2": W2,
        "W3": W3,
        "W4": W4,
    },
    pt_filepath,
    )

    # second plot the loss
    plt.figure()
    plt.plot(losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)

    # save the graph
    timestamp = time.strftime("%Y%m%d%H%M%S")
    pdf_filename = f"crossentropyloss_{timestamp}.pdf"
    pdf_filepath = os.path.join(output_dir, pdf_filename)
    plt.savefig(pdf_filepath)
    plt.close()

if __name__ == "__main__":
    d = 32  # number of features
    n = 1024  # number of samples
    main(d, n)