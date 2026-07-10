def compute_prediction(X, weights):
    """Manual prediction with overflow protection"""
    predictions = []
    for row in X:
        pred = 0
        for i, val in enumerate(row):
            pred += clip_value(val * weights[i])
        predictions.append(clip_value(pred))
    return predictions

def compute_mean(values):
    """Manual mean calculation"""
    return sum(values) / len(values)

def compute_loss(y_true, y_pred):
    """Manual MSE calculation with overflow protection"""
    squared_errors = []
    for i in range(len(y_true)):
        error = clip_value(y_true[i] - y_pred[i])
        squared_errors.append(error * error)
    return compute_mean(squared_errors)

def random_weights(size):
    """Manual random weight initialization with smaller values"""
    import random
    return [random.uniform(-0.1, 0.1) for _ in range(size)]

def clip_value(value, min_val=-1e10, max_val=1e10):
    """Clip values to prevent overflow"""
    return min(max(value, min_val), max_val)

def gradient_descent(X, Y, lr=0.01, epochs=1000):
    """
    Pure Python implementation of Gradient Descent with overflow protection
    """
    n_samples = len(X)
    n_features = len(X[0])
    
    # Initialize weights with smaller values
    weights = random_weights(n_features)
    weights_history = [weights[:]]  # Make a copy
    loss_history = []
    
    for epoch in range(epochs):
        # Forward pass with overflow protection
        predictions = compute_prediction(X, weights)
        
        # Compute loss
        current_loss = compute_loss(Y, predictions)
        loss_history.append(current_loss)
        
        # Compute gradients manually with overflow protection
        gradients = [0] * n_features
        for j in range(n_features):
            error_sum = 0
            for i in range(n_samples):
                error = clip_value(Y[i] - predictions[i])
                error_sum = clip_value(error_sum + error * X[i][j])
            gradients[j] = clip_value(-2 * error_sum / n_samples)
        
        # Update weights with overflow protection
        for j in range(n_features):
            update = clip_value(lr * gradients[j], -1.0, 1.0)
            weights[j] = clip_value(weights[j] - update)
            
        weights_history.append(weights[:])  # Make a copy
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {current_loss:.6f}")
            print(f"Current weights: {weights}")
    
    return weights, weights_history, loss_history

def build_polynomial_features(X, degree):
    """Manual construction of polynomial features with scaling"""
    features = []
    for x in X:
        row = []
        for d in range(degree, -1, -1):
            # Scale down higher degree terms
            value = clip_value(x ** d)
            row.append(value)
        features.append(row)
    return features