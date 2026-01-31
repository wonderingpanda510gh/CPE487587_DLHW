import torch
from cpe487587hw import binary_classification, animate_large_heatmap, animate_weight_heatmap


def main():
    # default parameters provieded in the homework 02 instruction
    d = 200
    n = 40000
    epochs = 5000
    eta = 0.01

    record_weight_hist = binary_classification(d=d, n=n, learning_rate=eta, num_epochs=epochs, record_weights=True)

    W1_hist = record_weight_hist["W1_hist"]  
    W2_hist = record_weight_hist["W2_hist"]  
    W3_hist = record_weight_hist["W3_hist"]  
    W4_hist = record_weight_hist["W4_hist"]  


    # wight heatmap
    animate_weight_heatmap(W1_hist, dt=0.04, file_name="W1_evolution", title_str="W1 Evolution")
    animate_weight_heatmap(W2_hist, dt=0.04, file_name="W2_evolution", title_str="W2 Evolution")
    animate_weight_heatmap(W3_hist, dt=0.04, file_name="W3_evolution", title_str="W3 Evolution")
    animate_weight_heatmap(W4_hist, dt=0.04, file_name="W4_evolution", title_str="W4 Evolution")

    # large weight heatmap
    animate_large_heatmap(W1_hist, dt=0.04, file_name="W1_evolution", title_str="W1 Evolution")
    animate_large_heatmap(W2_hist, dt=0.04, file_name="W2_evolution", title_str="W2 Evolution")
    animate_large_heatmap(W3_hist, dt=0.04, file_name="W3_evolution", title_str="W3 Evolution")
    animate_large_heatmap(W4_hist, dt=0.04, file_name="W4_evolution", title_str="W4 Evolution")


if __name__ == "__main__":
    main()
