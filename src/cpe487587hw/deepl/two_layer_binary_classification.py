import torch

def binary_classification(d, n, learning_rate=0.001, num_epochs=10000, record_weights = True):
    """
    Deep learning homework 01: Gradient descent and optimization, the learning rate and number of epochs are set to default values 0.001 and 10000.

    Deep learning homework 02: plot the weight history if record_weights is set to True.
    
    :param d: number of features
    :param n: number of samples
    :param learning_rate: learning rate for gradient descent
    :param num_epochs: number of epochs for training
    :param record_weights: whether to record weights for animation
    """

    # check device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # generate random date
    X = torch.rand(n, d, dtype = torch.float32, device = device)
    # generate labels for the random matrix X
    y = (X.sum(dim = 1, keepdim = True) > 2).float()

    # initialize weights
    W1 = torch.empty(d, 48, device=device, requires_grad=True)
    torch.nn.init.normal_(W1, mean=0.0, std=(2.0 / d) ** 0.5)
    W2 = torch.empty(48, 16, device=device, requires_grad=True)
    torch.nn.init.normal_(W2, mean=0.0, std=(2.0 / 48) ** 0.5)
    W3 = torch.empty(16, 32, device=device, requires_grad=True)
    torch.nn.init.normal_(W3, mean=0.0, std=(2.0 / 16) ** 0.5)
    W4 = torch.empty(32, 1, device=device, requires_grad=True)
    torch.nn.init.normal_(W4, mean=0.0, std=(2.0 / 32) ** 0.5)

    # define the loss function
    loss_function = torch.nn.BCEWithLogitsLoss()

    # record the losses
    losses = []
    
    # initialize the wight history list
    if record_weights:
        W1_hist = torch.empty((num_epochs, *W1.shape), dtype=torch.float32, device="cpu")
        W2_hist = torch.empty((num_epochs, *W2.shape), dtype=torch.float32, device="cpu")
        W3_hist = torch.empty((num_epochs, *W3.shape), dtype=torch.float32, device="cpu")
        W4_hist = torch.empty((num_epochs, *W4.shape), dtype=torch.float32, device="cpu")
    else:
        W1_hist, W2_hist, W3_hist, W4_hist = None, None, None, None

    # training loop
    for epoch in range(num_epochs):
        # forward training step
        A = X @ W1
        A = torch.relu(A)
        A = A @ W2
        A = torch.relu(A)
        A = A @ W3
        A = torch.relu(A)
        A = A @ W4

        # compute the loss
        loss = loss_function(A, y)
        losses.append(loss.item())
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item()}")


        # backward update step
        loss.backward()

        with torch.no_grad():
            W1 -= learning_rate * W1.grad
            W2 -= learning_rate * W2.grad
            W3 -= learning_rate * W3.grad
            W4 -= learning_rate * W4.grad

            # zero the gradients after updating
            W1.grad.zero_()
            W2.grad.zero_()
            W3.grad.zero_()
            W4.grad.zero_()
        
        if record_weights:
            W1_hist[epoch] = W1.detach().cpu().clone()   # the reson for using clone() is to avoid the weight history being changed druing training, because if we don't use clone(), the weight history may be changed when the weights are updated in the next epoch, they are still in the computation graph, this will lead to every matrix in the weight history is the same one, and we can't see the weight change during training.
            W2_hist[epoch] = W2.detach().cpu().clone()
            W3_hist[epoch] = W3.detach().cpu().clone()
            W4_hist[epoch] = W4.detach().cpu().clone()

    return {
        "W1": W1.detach().cpu(), # the final weights
        "W2": W2.detach().cpu(),
        "W3": W3.detach().cpu(),
        "W4": W4.detach().cpu(),
        "losses": losses, # the loss history
        "W1_hist": W1_hist, # the wight history
        "W2_hist": W2_hist,
        "W3_hist": W3_hist,
        "W4_hist": W4_hist
    }
