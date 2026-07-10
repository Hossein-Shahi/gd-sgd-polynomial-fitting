import matplotlib.pyplot as plt
from gradient_descent import gradient_descent, build_polynomial_features as gd_build_features
from stochastic_gradient_descent import stochastic_gradient_descent, build_polynomial_features as sgd_build_features

def load_data(filename):
    """Load data manually from file"""
    X, Y = [], []
    with open(filename, 'r') as f:
        next(f)  # Skip header
        for line in f:
            if line.strip():  # Skip empty lines
                x, y = line.strip().split()
                X.append(float(x))
                Y.append(float(y))
    return X, Y

def compute_mean(values):
    """Manual mean calculation"""
    return sum(values) / len(values)

def compute_std(values):
    """Manual standard deviation calculation"""
    mean = compute_mean(values)
    squared_diff_sum = sum((x - mean) ** 2 for x in values)
    return (squared_diff_sum / len(values)) ** 0.5

def normalize_data(data):
    """Normalize data manually"""
    mean = compute_mean(data)
    std = compute_std(data)
    return [((x - mean) / std) for x in data], mean, std

def sort_data(X, Y):
    """Manual sorting for plotting"""
    pairs = list(zip(X, Y))
    pairs.sort(key=lambda x: x[0])
    return [p[0] for p in pairs], [p[1] for p in pairs]

# Load and prepare data
print("Loading data...")
X, Y = load_data('Part1_x_y_Values.txt')

# Normalize data
print("Normalizing data...")
X_normalized, X_mean, X_std = normalize_data(X)
Y_normalized, Y_mean, Y_std = normalize_data(Y)

# Parameters
degrees = [2, 3, 4]  # Different polynomial degrees
learning_rates = [0.001, 0.01, 0.1]  # Different learning rates
epochs = 1000

# Run experiments for each combination
for degree in degrees:
    print(f"\nProcessing polynomial degree {degree}")
    for lr in learning_rates:
        print(f"\nLearning rate {lr}")
        
        # Build features
        X_features = gd_build_features(X_normalized, degree)
        
        # Run Gradient Descent
        print("Running Gradient Descent...")
        gd_weights, gd_weights_history, gd_loss_history = gradient_descent(
            X_features, Y_normalized, lr=lr, epochs=epochs
        )
        
        # Run Stochastic Gradient Descent
        print("Running Stochastic Gradient Descent...")
        sgd_weights, sgd_weights_history, sgd_loss_history = stochastic_gradient_descent(
            X_features, Y_normalized, lr=lr, epochs=epochs
        )
        
        # Create plots
        plt.figure(figsize=(15, 5))
        
        # Plot 1: Data and fitted curves
        plt.subplot(1, 3, 1)
        plt.scatter(X, Y, color='blue', label='Data Points', alpha=0.5, s=20)
        
        # Sort X for smooth curve plotting
        X_sorted, _ = sort_data(X, Y)
        X_sorted_norm = [(x - X_mean) / X_std for x in X_sorted]
        
        # Generate predictions for plotting
        X_plot_features = gd_build_features(X_sorted_norm, degree)
        
        # GD predictions
        gd_pred = []
        for x_feat in X_plot_features:
            pred = sum(x_feat[j] * gd_weights[j] for j in range(degree + 1))
            gd_pred.append(pred * Y_std + Y_mean)
            
        # SGD predictions
        sgd_pred = []
        for x_feat in X_plot_features:
            pred = sum(x_feat[j] * sgd_weights[j] for j in range(degree + 1))
            sgd_pred.append(pred * Y_std + Y_mean)
        
        plt.plot(X_sorted, gd_pred, 'r-', label='GD Fit')
        plt.plot(X_sorted, sgd_pred, 'g-', label='SGD Fit')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title(f'Fitting Curves (Degree={degree}, LR={lr})')
        plt.legend()
        
        # Plot 2: Loss history
        plt.subplot(1, 3, 2)
        plt.plot(gd_loss_history, 'r-', label='GD Loss')
        plt.plot(sgd_loss_history, 'g-', label='SGD Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss History')
        plt.legend()
        
        # Plot 3: Weight history
        plt.subplot(1, 3, 3)
        for i in range(degree + 1):
            plt.plot([w[i] for w in gd_weights_history], 'r-', alpha=0.5,
                    label=f'GD Weight {i}' if i == 0 else None)
            plt.plot([w[i] for w in sgd_weights_history], 'g-', alpha=0.5,
                    label=f'SGD Weight {i}' if i == 0 else None)
        
        plt.xlabel('Epoch')
        plt.ylabel('Weight Value')
        plt.title('Weight History')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(f'fitting_degree_{degree}_lr_{lr}.png')
        plt.close()

        # Print final results
        print("\nFinal Results:")
        print(f"{'='*50}")
        print(f"Polynomial Degree: {degree}")
        print(f"Learning Rate: {lr}")
        print("GD Final Weights:", gd_weights)
        print("GD Final Loss:", gd_loss_history[-1])
        print("SGD Final Weights:", sgd_weights)
        print("SGD Final Loss:", sgd_loss_history[-1])