def compute_single_prediction(x_sample, weights):
    """Manual prediction for a single sample"""
    prediction = 0
    for x, w in zip(x_sample, weights):
        prediction += x * w
    return prediction

def shuffle_data(X, Y):
    """Manual data shuffling"""
    import random
    indices = list(range(len(X)))
    random.shuffle(indices)
    
    X_shuffled = [X[i] for i in indices]
    Y_shuffled = [Y[i] for i in indices]
    return X_shuffled, Y_shuffled

def random_weights(size):
    """Manual random weight initialization with smaller values"""
    import random
    return [random.uniform(-0.1, 0.1) for _ in range(size)]

def clip_value(value, min_val=-1e10, max_val=1e10):
    """Clip values to prevent overflow"""
    return min(max(value, min_val), max_val)

def stochastic_gradient_descent(X, Y, lr=0.01, epochs=1000):
    """
    Pure Python implementation of Stochastic Gradient Descent without using any library optimizers.
    Added protection against overflow.
    """
    n_samples = len(X)
    n_features = len(X[0])
    
    # Initialize weights with smaller values
    weights = random_weights(n_features)
    weights_history = [weights[:]]
    loss_history = []
    
    for epoch in range(epochs):
        epoch_loss = 0
        
        # Shuffle data
        X_shuffled, Y_shuffled = shuffle_data(X, Y)
        
        # Process one sample at a time
        for i in range(n_samples):
            x_sample = X_shuffled[i]
            y_true = Y_shuffled[i]
            
            # Forward pass
            y_pred = compute_single_prediction(x_sample, weights)
            
            # Compute sample loss with clipping
            error = clip_value(y_true - y_pred)
            sample_loss = error * error
            epoch_loss += sample_loss
            
            # Compute gradients for single sample
            gradients = []
            for x in x_sample:
                # Clip gradient to prevent explosion
                gradient = clip_value(-2 * error * x)
                gradients.append(gradient)
            
            # Update weights with smaller learning rate if needed
            for j in range(n_features):
                update = lr * gradients[j]
                # Clip weight update
                update = clip_value(update, -1.0, 1.0)
                weights[j] = clip_value(weights[j] - update)
        
        # Store weights and average loss
        weights_history.append(weights[:])
        avg_epoch_loss = epoch_loss / n_samples
        loss_history.append(avg_epoch_loss)
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Average Loss: {avg_epoch_loss:.6f}")
            print(f"Current weights: {weights}")
    
    return weights, weights_history, loss_history

def build_polynomial_features(X, degree):
    """Manual construction of polynomial features with scaling"""
    features = []
    for x in X:
        row = []
        for d in range(degree, -1, -1):
            # Scale down higher degree terms
            value = x ** d
            # Clip to prevent overflow
            value = clip_value(value, -1e10, 1e10)
            row.append(value)
        features.append(row)
    return features