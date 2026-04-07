#!/usr/bin/env python3
"""
Training script for neural move predictor.

This script generates training data from solved cubes and trains a neural
network to predict good moves for the Kociemba algorithm.

Usage:
    python train_neural_heuristic.py --samples 100000 --epochs 100
    python train_neural_heuristic.py --samples 500000 --epochs 200 --gpu
"""

import argparse
import sys
import numpy as np
import time
from pathlib import Path

# Add kociemba to path
sys.path.insert(0, str(Path(__file__).parent / 'kociemba'))


def apply_move(cube, move_index, moveCube):
    """
    Apply a move (0-17) to a cube.

    move_index = axis * 3 + power
    where axis is 0-5 (U,R,F,D,L,B) and power is 0-2 (90°, 180°, 270°)
    """
    axis = move_index // 3
    power = move_index % 3  # 0=90°, 1=180°, 2=270°

    # Apply the basic move (power+1) times
    for _ in range(power + 1):
        cube.cornerMultiply(moveCube[axis])
        cube.edgeMultiply(moveCube[axis])


def cube_to_string(cube):
    """Convert CubieCube to facelet string for solver."""
    return cube.toFaceCube().to_String()


def generate_training_data_with_solver(n_samples=10000, max_scramble_length=20, timeout_ms=500):
    """
    Generate training data using actual solver feedback.

    Strategy: For each scrambled state, try all 18 moves and solve the
    resulting states. Label moves that lead to shorter solutions as "good".

    This teaches the network which moves actually reduce solution length.
    """
    from kociemba.pykociemba.coordcube import CoordCube
    from kociemba.pykociemba.cubiecube import CubieCube, moveCube
    from kociemba import _solve_python

    N_MOVES = 18

    print(f"Generating {n_samples} training samples with solver feedback...")
    print(f"This is slower but produces better labels.")

    X = []
    y_soft = []  # Soft labels (probability distribution)

    samples_per_depth = max(1, n_samples // max_scramble_length)

    for depth in range(1, max_scramble_length + 1):
        if depth % 2 == 0:
            print(f"  Depth {depth}/{max_scramble_length} ({len(X)} samples so far)")

        for sample_idx in range(samples_per_depth):
            # Start from solved
            cube = CubieCube()

            # Apply random moves to scramble
            last_axis = -1
            for _ in range(depth):
                while True:
                    mv = np.random.randint(N_MOVES)
                    axis = mv // 3
                    if axis != last_axis:
                        break
                last_axis = axis
                apply_move(cube, mv, moveCube)

            # Get cube string for this state
            try:
                base_string = cube_to_string(cube)
            except:
                continue

            # Try each of 18 moves and measure solution length
            solution_lengths = []
            for mv in range(N_MOVES):
                # Copy cube and apply move
                test_cube = CubieCube(
                    cp=cube.cp[:], co=cube.co[:],
                    ep=cube.ep[:], eo=cube.eo[:]
                )
                apply_move(test_cube, mv, moveCube)

                try:
                    test_string = cube_to_string(test_cube)
                    # Solve with short timeout
                    solution = _solve_python(test_string, None, 24, 1, timeout_ms)
                    sol_len = len(solution.strip().split())
                    solution_lengths.append(sol_len)
                except:
                    # If solve fails, use large penalty
                    solution_lengths.append(30)

            # Convert to soft labels: lower solution length = higher probability
            solution_lengths = np.array(solution_lengths, dtype=np.float32)
            min_len = solution_lengths.min()

            # Moves that achieve minimum length get highest probability
            # Use temperature-scaled softmax on negative lengths
            temperature = 1.0
            scores = -(solution_lengths - min_len) / temperature
            probs = np.exp(scores) / np.exp(scores).sum()

            # Extract features
            coord = CoordCube(cube)
            N_TWIST = 2187
            N_FLIP = 2048
            N_SLICE = 495
            N_PARITY = 2
            N_URFDLF = 20160
            N_FRTOBR = 11880

            features = np.array([
                coord.twist / N_TWIST,
                coord.flip / N_FLIP,
                (coord.FRtoBR // 24) / N_SLICE,
                coord.parity / N_PARITY,
                coord.URFtoDLF / N_URFDLF,
                coord.FRtoBR / N_FRTOBR,
                (coord.twist % 729) / 729,
                (coord.flip % 256) / 256,
                ((coord.FRtoBR // 24) % 99) / 99,
            ], dtype=np.float32)

            X.append(features)
            y_soft.append(probs)

    X = np.array(X, dtype=np.float32)
    y_soft = np.array(y_soft, dtype=np.float32)

    print(f"Generated {len(X)} samples with solver feedback")
    return X, y_soft


def generate_training_data(n_samples=100000, max_scramble_length=25):
    """
    Generate training data from random scrambles (fast but less accurate).

    For better results, use generate_training_data_with_solver() instead.
    """
    from kociemba.pykociemba.coordcube import CoordCube
    from kociemba.pykociemba.cubiecube import CubieCube, moveCube

    N_MOVES = 18

    print(f"Generating {n_samples} training samples (fast mode)...")

    X = []
    y = []

    samples_per_depth = max(1, n_samples // max_scramble_length)

    for depth in range(1, max_scramble_length + 1):
        if depth % 5 == 0:
            print(f"  Depth {depth}/{max_scramble_length} ({len(X)} samples so far)")

        for _ in range(samples_per_depth):
            # Start from solved
            cube = CubieCube()

            # Apply random moves
            moves = []
            last_axis = -1

            for _ in range(depth):
                # Pick random move, avoiding redundant moves
                while True:
                    mv = np.random.randint(N_MOVES)
                    axis = mv // 3
                    if axis != last_axis:
                        break
                moves.append(mv)
                last_axis = axis

                # Apply move
                apply_move(cube, mv, moveCube)

            # Extract coordinates
            coord = CoordCube(cube)

            # The "good" move is the inverse of the last applied move
            last_move = moves[-1]
            axis = last_move // 3
            power = last_move % 3

            # Inverse power: 0->2, 1->1, 2->0
            inv_power = (4 - power - 1) % 3
            good_move = axis * 3 + inv_power

            # Extract features (normalized coordinates)
            N_TWIST = 2187
            N_FLIP = 2048
            N_SLICE = 495
            N_PARITY = 2
            N_URFDLF = 20160
            N_FRTOBR = 11880

            features = np.array([
                coord.twist / N_TWIST,
                coord.flip / N_FLIP,
                (coord.FRtoBR // 24) / N_SLICE,
                coord.parity / N_PARITY,
                coord.URFtoDLF / N_URFDLF,
                coord.FRtoBR / N_FRTOBR,
                (coord.twist % 729) / 729,
                (coord.flip % 256) / 256,
                ((coord.FRtoBR // 24) % 99) / 99,
            ], dtype=np.float32)

            X.append(features)
            y.append(good_move)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)

    print(f"Generated {len(X)} samples")
    return X, y


def train_numpy_soft(X, y_soft, epochs=100, batch_size=256, learning_rate=0.001,
                     hidden_sizes=[256, 128, 64]):
    """Train using soft labels (probability distributions) with NumPy."""
    n_samples, n_features = X.shape
    n_classes = y_soft.shape[1]  # 18 moves
    n_batches = (n_samples + batch_size - 1) // batch_size

    # Initialize weights (He initialization)
    np.random.seed(42)
    weights = []
    biases = []

    sizes = [n_features] + hidden_sizes + [n_classes]
    for i in range(len(sizes) - 1):
        w = np.random.randn(sizes[i], sizes[i+1]) * np.sqrt(2.0 / sizes[i])
        b = np.zeros(sizes[i+1])
        weights.append(w.astype(np.float32))
        biases.append(b.astype(np.float32))

    def relu(x):
        return np.maximum(0, x)

    def softmax(x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / exp_x.sum(axis=1, keepdims=True)

    history = []

    print(f"\nTraining for {epochs} epochs with soft labels...")
    start_time = time.time()

    for epoch in range(epochs):
        # Shuffle
        indices = np.random.permutation(n_samples)
        X_shuffled = X[indices]
        y_shuffled = y_soft[indices]

        epoch_loss = 0

        for batch in range(n_batches):
            start = batch * batch_size
            end = min(start + batch_size, n_samples)

            X_batch = X_shuffled[start:end]
            y_batch = y_shuffled[start:end]  # Soft labels

            # Forward pass
            activations = [X_batch]
            for w, b in zip(weights[:-1], biases[:-1]):
                z = np.dot(activations[-1], w) + b
                activations.append(relu(z))

            logits = np.dot(activations[-1], weights[-1]) + biases[-1]
            probs = softmax(logits)

            # Cross-entropy loss with soft labels
            batch_loss = -np.mean(np.sum(y_batch * np.log(probs + 1e-8), axis=1))
            epoch_loss += batch_loss

            # Backward pass (gradient of cross-entropy with soft labels)
            grad_logits = (probs - y_batch) / len(y_batch)

            grad_w = [np.dot(activations[-1].T, grad_logits)]
            grad_b = [grad_logits.sum(axis=0)]

            grad_a = np.dot(grad_logits, weights[-1].T)

            for i in range(len(weights) - 2, -1, -1):
                grad_z = grad_a * (activations[i + 1] > 0)
                grad_w.insert(0, np.dot(activations[i].T, grad_z))
                grad_b.insert(0, grad_z.sum(axis=0))
                if i > 0:
                    grad_a = np.dot(grad_z, weights[i].T)

            # Update
            for i in range(len(weights)):
                weights[i] -= learning_rate * grad_w[i]
                biases[i] -= learning_rate * grad_b[i]

        epoch_loss /= n_batches
        history.append(epoch_loss)

        if (epoch + 1) % 10 == 0:
            # Compute accuracy (predict argmax, compare to argmax of soft labels)
            x = X[:1000]
            for w, b in zip(weights[:-1], biases[:-1]):
                x = relu(np.dot(x, w) + b)
            logits = np.dot(x, weights[-1]) + biases[-1]
            pred = np.argmax(logits, axis=1)
            true = np.argmax(y_soft[:1000], axis=1)
            acc = np.mean(pred == true)

            elapsed = time.time() - start_time
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}, "
                  f"Acc: {acc:.3f}, Time: {elapsed:.1f}s")

    return weights, biases, history


def train_numpy(X, y, epochs=100, batch_size=256, learning_rate=0.001,
                hidden_sizes=[256, 128, 64]):
    """Train using pure NumPy (CPU)."""
    n_samples, n_features = X.shape
    n_classes = 18
    n_batches = (n_samples + batch_size - 1) // batch_size

    # Initialize weights (He initialization)
    np.random.seed(42)
    weights = []
    biases = []

    sizes = [n_features] + hidden_sizes + [n_classes]
    for i in range(len(sizes) - 1):
        w = np.random.randn(sizes[i], sizes[i+1]) * np.sqrt(2.0 / sizes[i])
        b = np.zeros(sizes[i+1])
        weights.append(w.astype(np.float32))
        biases.append(b.astype(np.float32))

    def relu(x):
        return np.maximum(0, x)

    def softmax(x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / exp_x.sum(axis=1, keepdims=True)

    history = []

    print(f"\nTraining for {epochs} epochs...")
    start_time = time.time()

    for epoch in range(epochs):
        # Shuffle
        indices = np.random.permutation(n_samples)
        X_shuffled = X[indices]
        y_shuffled = y[indices]

        epoch_loss = 0

        for batch in range(n_batches):
            start = batch * batch_size
            end = min(start + batch_size, n_samples)

            X_batch = X_shuffled[start:end]
            y_batch = y_shuffled[start:end]

            # Forward pass
            activations = [X_batch]
            for w, b in zip(weights[:-1], biases[:-1]):
                z = np.dot(activations[-1], w) + b
                activations.append(relu(z))

            logits = np.dot(activations[-1], weights[-1]) + biases[-1]
            probs = softmax(logits)

            # Loss
            batch_loss = -np.mean(np.log(probs[np.arange(len(y_batch)), y_batch] + 1e-8))
            epoch_loss += batch_loss

            # Backward pass
            grad_logits = probs.copy()
            grad_logits[np.arange(len(y_batch)), y_batch] -= 1
            grad_logits /= len(y_batch)

            grad_w = [np.dot(activations[-1].T, grad_logits)]
            grad_b = [grad_logits.sum(axis=0)]

            grad_a = np.dot(grad_logits, weights[-1].T)

            for i in range(len(weights) - 2, -1, -1):
                grad_z = grad_a * (activations[i + 1] > 0)
                grad_w.insert(0, np.dot(activations[i].T, grad_z))
                grad_b.insert(0, grad_z.sum(axis=0))
                if i > 0:
                    grad_a = np.dot(grad_z, weights[i].T)

            # Update
            for i in range(len(weights)):
                weights[i] -= learning_rate * grad_w[i]
                biases[i] -= learning_rate * grad_b[i]

        epoch_loss /= n_batches
        history.append(epoch_loss)

        if (epoch + 1) % 10 == 0:
            # Compute accuracy
            x = X[:1000]
            for w, b in zip(weights[:-1], biases[:-1]):
                x = relu(np.dot(x, w) + b)
            logits = np.dot(x, weights[-1]) + biases[-1]
            pred = np.argmax(logits, axis=1)
            acc = np.mean(pred == y[:1000])

            elapsed = time.time() - start_time
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}, "
                  f"Acc: {acc:.3f}, Time: {elapsed:.1f}s")

    return weights, biases, history


def train_pytorch(X, y, epochs=100, batch_size=256, learning_rate=0.001,
                  hidden_sizes=[256, 128, 64], device='cuda'):
    """Train using PyTorch (GPU if available)."""
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torch.utils.data import TensorDataset, DataLoader
    except ImportError:
        print("PyTorch not available, falling back to NumPy")
        return train_numpy(X, y, epochs, batch_size, learning_rate, hidden_sizes)

    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = 'cpu'

    print(f"Using device: {device}")

    # Create model
    n_features = X.shape[1]
    n_classes = 18

    layers = []
    sizes = [n_features] + hidden_sizes
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i+1]))
        layers.append(nn.ReLU())
    layers.append(nn.Linear(sizes[-1], n_classes))

    model = nn.Sequential(*layers).to(device)

    # Data loaders
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)
    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Training
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    history = []
    start_time = time.time()

    print(f"\nTraining for {epochs} epochs...")

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        n_batches = 0

        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        epoch_loss /= n_batches
        history.append(epoch_loss)

        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                X_test = torch.tensor(X[:1000], dtype=torch.float32).to(device)
                outputs = model(X_test)
                pred = outputs.argmax(dim=1).cpu().numpy()
                acc = np.mean(pred == y[:1000])

            elapsed = time.time() - start_time
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}, "
                  f"Acc: {acc:.3f}, Time: {elapsed:.1f}s")

    # Extract weights
    weights = []
    biases = []
    for layer in model:
        if isinstance(layer, nn.Linear):
            weights.append(layer.weight.detach().cpu().numpy().T)
            biases.append(layer.bias.detach().cpu().numpy())

    return weights, biases, history


def save_model(weights, biases, path, hidden_sizes=[256, 128, 64]):
    """Save model to pickle file."""
    import pickle
    data = {
        'weights': weights,
        'biases': biases,
        'hidden_sizes': hidden_sizes,
    }
    with open(path, 'wb') as f:
        pickle.dump(data, f)
    print(f"Model saved to {path}")


def main():
    parser = argparse.ArgumentParser(description='Train neural move predictor')
    parser.add_argument('--samples', type=int, default=100000,
                        help='Number of training samples')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Training epochs')
    parser.add_argument('--batch-size', type=int, default=256,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--output', type=str,
                        default='kociemba/kociemba/neural_model.pkl',
                        help='Output model path')
    parser.add_argument('--gpu', action='store_true',
                        help='Use GPU (requires PyTorch with CUDA)')
    parser.add_argument('--hidden', type=str, default='256,128,64',
                        help='Hidden layer sizes (comma-separated)')
    parser.add_argument('--solver-feedback', action='store_true',
                        help='Use solver feedback for training (slower but better)')
    parser.add_argument('--timeout', type=int, default=500,
                        help='Solver timeout in ms (for solver-feedback mode)')

    args = parser.parse_args()

    hidden_sizes = [int(x) for x in args.hidden.split(',')]

    print("=" * 60)
    print("Neural Move Predictor Training")
    print("=" * 60)
    print(f"Samples:       {args.samples}")
    print(f"Epochs:        {args.epochs}")
    print(f"Batch size:    {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Hidden layers: {hidden_sizes}")
    print(f"Output:        {args.output}")
    print(f"GPU:           {args.gpu}")
    print(f"Solver feedback: {args.solver_feedback}")
    print("=" * 60)

    if args.solver_feedback:
        # Generate data with solver feedback (soft labels)
        X, y_soft = generate_training_data_with_solver(
            n_samples=args.samples,
            timeout_ms=args.timeout
        )

        # Split into train/val
        n_val = min(1000, len(X) // 10)
        X_train, X_val = X[:-n_val], X[-n_val:]
        y_train, y_val = y_soft[:-n_val], y_soft[-n_val:]

        print(f"\nTrain samples: {len(X_train)}")
        print(f"Val samples:   {len(X_val)}")

        # Train with soft labels
        weights, biases, history = train_numpy_soft(
            X_train, y_train, epochs=args.epochs, batch_size=args.batch_size,
            learning_rate=args.lr, hidden_sizes=hidden_sizes
        )

        # Save model
        save_model(weights, biases, args.output, hidden_sizes)

        # Final evaluation
        print("\n" + "=" * 60)
        print("Final Evaluation")
        print("=" * 60)

        def predict(X, weights, biases):
            x = X
            for w, b in zip(weights[:-1], biases[:-1]):
                x = np.maximum(0, np.dot(x, w) + b)
            logits = np.dot(x, weights[-1]) + biases[-1]
            return np.argmax(logits, axis=1)

        train_pred = predict(X_train[:1000], weights, biases)
        train_true = np.argmax(y_train[:1000], axis=1)
        train_acc = np.mean(train_pred == train_true)

        val_pred = predict(X_val, weights, biases)
        val_true = np.argmax(y_val, axis=1)
        val_acc = np.mean(val_pred == val_true)

        print(f"Train accuracy: {train_acc:.3f}")
        print(f"Val accuracy:   {val_acc:.3f}")
        print(f"Final loss:     {history[-1]:.4f}")
        print("=" * 60)

    else:
        # Original fast training (hard labels)
        X, y = generate_training_data(n_samples=args.samples)

        # Split into train/val
        n_val = min(10000, len(X) // 10)
        X_train, X_val = X[:-n_val], X[-n_val:]
        y_train, y_val = y[:-n_val], y[-n_val:]

        print(f"\nTrain samples: {len(X_train)}")
        print(f"Val samples:   {len(X_val)}")

        # Train
        if args.gpu:
            weights, biases, history = train_pytorch(
                X_train, y_train, epochs=args.epochs, batch_size=args.batch_size,
                learning_rate=args.lr, hidden_sizes=hidden_sizes, device='cuda'
            )
        else:
            weights, biases, history = train_numpy(
                X_train, y_train, epochs=args.epochs, batch_size=args.batch_size,
                learning_rate=args.lr, hidden_sizes=hidden_sizes
            )

        # Save model
        save_model(weights, biases, args.output, hidden_sizes)

        # Final evaluation
        print("\n" + "=" * 60)
        print("Final Evaluation")
        print("=" * 60)

        def predict(X, weights, biases):
            x = X
            for w, b in zip(weights[:-1], biases[:-1]):
                x = np.maximum(0, np.dot(x, w) + b)
            logits = np.dot(x, weights[-1]) + biases[-1]
            return np.argmax(logits, axis=1)

        train_pred = predict(X_train[:5000], weights, biases)
        train_acc = np.mean(train_pred == y_train[:5000])

        val_pred = predict(X_val, weights, biases)
        val_acc = np.mean(val_pred == y_val)

        print(f"Train accuracy: {train_acc:.3f}")
        print(f"Val accuracy:   {val_acc:.3f}")
        print(f"Final loss:     {history[-1]:.4f}")
        print("=" * 60)


if __name__ == '__main__':
    main()
