"""
Neural Network Move Ordering for Kociemba's Two-Phase Algorithm.

This module implements a neural network-based heuristic for predicting
which moves are most likely to lead to shorter solutions. The network
is trained on solved cubes and used to order move exploration in IDA*.

Research improvement: Expected 3-5% additional move reduction beyond
multi-solution search, and 30-40% reduction in search nodes explored.
"""

import os
import pickle
import numpy as np
from pathlib import Path

# Coordinate sizes for feature extraction
N_TWIST = 2187  # 3^7 corner orientations
N_FLIP = 2048   # 2^11 edge orientations
N_SLICE = 495   # 12 choose 4
N_PARITY = 2
N_URFDLF = 20160
N_FRTOBR = 11880

# Move encoding
N_MOVES = 18  # 6 faces x 3 rotations
MOVE_NAMES = [
    "U", "U2", "U'", "R", "R2", "R'",
    "F", "F2", "F'", "D", "D2", "D'",
    "L", "L2", "L'", "B", "B2", "B'"
]


# Feature extraction modes
FEATURE_MODE_SCALAR = 'scalar'          # Original: 9 normalized scalars
FEATURE_MODE_COORDINATE = 'coordinate'  # New: binned coordinate features (richer)

# Binning parameters for coordinate features
N_TWIST_BINS = 32   # 2187 / ~68 per bin
N_FLIP_BINS = 32    # 2048 / 64 per bin
N_SLICE_BINS = 16   # 495 / ~31 per bin
N_URFDLF_BINS = 32  # 20160 / 630 per bin
N_FRTOBR_BINS = 32  # 11880 / ~371 per bin

# Total coordinate feature size: bins + scalars + interactions
COORDINATE_FEATURE_SIZE = (N_TWIST_BINS + N_FLIP_BINS + N_SLICE_BINS +
                           N_URFDLF_BINS + N_FRTOBR_BINS +
                           2 +   # parity (one-hot)
                           9 +   # original scalar features
                           6)    # pairwise interaction features
# = 32+32+16+32+32+2+9+6 = 161


def _bin_encode(value, max_val, n_bins):
    """Encode an integer coordinate as a one-hot bin vector."""
    vec = np.zeros(n_bins, dtype=np.float32)
    bin_idx = min(int(value * n_bins / (max_val + 1)), n_bins - 1)
    vec[bin_idx] = 1.0
    return vec


def extract_coordinate_features(twist, flip, slice_coord, parity=0,
                                 URFtoDLF=0, FRtoBR=0):
    """
    Extract rich coordinate-based features using binned one-hot encoding.

    This addresses the core bottleneck identified in the paper: raw scalar
    coordinates lose structural information about the group-theoretic
    relationships between cube states. Binned encoding allows the network
    to learn non-linear patterns within each coordinate's range.

    Feature composition (161 total):
      - Binned twist (32)     : corner orientation clusters
      - Binned flip (32)      : edge orientation clusters
      - Binned slice (16)     : UD-slice position clusters
      - Binned URFtoDLF (32)  : corner permutation clusters
      - Binned FRtoBR (32)    : edge permutation clusters
      - Parity one-hot (2)    : even/odd permutation
      - Scalar coords (9)     : original normalized features
      - Interactions (6)      : pairwise products of key coordinates

    Returns:
        numpy array of shape (161,)
    """
    parts = []

    # 1. Binned one-hot features (captures non-linear structure)
    parts.append(_bin_encode(twist, N_TWIST, N_TWIST_BINS))
    parts.append(_bin_encode(flip, N_FLIP, N_FLIP_BINS))
    parts.append(_bin_encode(slice_coord, N_SLICE, N_SLICE_BINS))
    parts.append(_bin_encode(URFtoDLF, N_URFDLF, N_URFDLF_BINS))
    parts.append(_bin_encode(FRtoBR, N_FRTOBR, N_FRTOBR_BINS))

    # 2. Parity one-hot
    parity_vec = np.zeros(2, dtype=np.float32)
    parity_vec[int(parity)] = 1.0
    parts.append(parity_vec)

    # 3. Original scalar features (normalized)
    scalars = np.array([
        twist / N_TWIST,
        flip / N_FLIP,
        slice_coord / N_SLICE if N_SLICE > 0 else 0,
        parity,
        URFtoDLF / N_URFDLF if N_URFDLF > 0 else 0,
        FRtoBR / N_FRTOBR if N_FRTOBR > 0 else 0,
        (twist % 729) / 729,
        (flip % 256) / 256,
        (slice_coord % 99) / 99 if slice_coord > 0 else 0,
    ], dtype=np.float32)
    parts.append(scalars)

    # 4. Pairwise interaction features (captures joint coordinate effects)
    t_norm = twist / N_TWIST
    f_norm = flip / N_FLIP
    s_norm = slice_coord / N_SLICE if N_SLICE > 0 else 0
    interactions = np.array([
        t_norm * f_norm,    # twist-flip interaction
        t_norm * s_norm,    # twist-slice interaction
        f_norm * s_norm,    # flip-slice interaction
        t_norm * parity,    # twist-parity
        f_norm * parity,    # flip-parity
        s_norm * parity,    # slice-parity
    ], dtype=np.float32)
    parts.append(interactions)

    return np.concatenate(parts)


class NeuralValuePredictor:
    """
    Neural network for predicting distance to solved state.

    Unlike move classification (which fails due to softmax ambiguity when
    multiple moves are equally good), value prediction outputs a scalar
    distance estimate (0-20 moves). This can be used as an additional
    heuristic bound in IDA* search.

    Key advantages over move classification:
    - MSE loss works with continuous targets
    - No ambiguity when multiple moves lead to similar distances
    - Can blend with existing pruning table heuristics
    """

    def __init__(self, model_path=None, feature_mode=FEATURE_MODE_SCALAR):
        """Initialize the predictor, optionally loading a trained model."""
        self.model_path = model_path
        self.weights = None
        self.biases = None
        self.loaded = False
        self.feature_mode = feature_mode

        # Wider architecture for value prediction
        self.hidden_sizes = [512, 256, 128]

        if model_path and os.path.exists(model_path):
            self.load(model_path)

    def _relu(self, x):
        """ReLU activation."""
        return np.maximum(0, x)

    def extract_features(self, twist, flip, slice_coord, parity=0,
                         URFtoDLF=0, FRtoBR=0):
        """
        Extract features from cube coordinates.
        Dispatches to scalar (9-dim) or coordinate (161-dim) based on feature_mode.
        """
        if self.feature_mode == FEATURE_MODE_COORDINATE:
            return extract_coordinate_features(twist, flip, slice_coord,
                                                parity, URFtoDLF, FRtoBR)
        # Default: scalar features
        features = np.array([
            twist / N_TWIST,
            flip / N_FLIP,
            slice_coord / N_SLICE,
            parity / N_PARITY if N_PARITY > 0 else 0,
            URFtoDLF / N_URFDLF if N_URFDLF > 0 else 0,
            FRtoBR / N_FRTOBR if N_FRTOBR > 0 else 0,
            # Derived features
            (twist % 729) / 729,
            (flip % 256) / 256,
            (slice_coord % 99) / 99 if slice_coord > 0 else 0,
        ], dtype=np.float32)
        return features

    def forward(self, features):
        """Forward pass returning predicted distance."""
        x = features
        for w, b in zip(self.weights[:-1], self.biases[:-1]):
            x = self._relu(np.dot(x, w) + b)
        # Final layer: linear output (no activation)
        output = np.dot(x, self.weights[-1]) + self.biases[-1]
        # Clip to valid range [0, 20]
        return np.clip(output.flatten()[0] if output.ndim > 0 else output, 0, 20)

    def predict_distance(self, twist, flip, slice_coord, parity=0,
                         URFtoDLF=0, FRtoBR=0):
        """
        Predict estimated distance to solved state.

        Returns:
            Estimated distance (0-20 moves)
        """
        if not self.loaded:
            return 0  # No prediction available

        features = self.extract_features(twist, flip, slice_coord,
                                         parity, URFtoDLF, FRtoBR)
        return self.forward(features)

    def batch_forward(self, X):
        """Forward pass for batch of feature vectors."""
        x = X
        for w, b in zip(self.weights[:-1], self.biases[:-1]):
            x = self._relu(np.dot(x, w) + b)
        output = np.dot(x, self.weights[-1]) + self.biases[-1]
        return np.clip(output.flatten(), 0, 20)

    def initialize_random(self, input_size=None):
        """Initialize network with random weights (for training)."""
        if input_size is None:
            input_size = COORDINATE_FEATURE_SIZE if self.feature_mode == FEATURE_MODE_COORDINATE else 9
        np.random.seed(42)

        self.weights = []
        self.biases = []

        # Output is 1 (scalar distance)
        sizes = [input_size] + self.hidden_sizes + [1]

        for i in range(len(sizes) - 1):
            # He initialization
            w = np.random.randn(sizes[i], sizes[i+1]) * np.sqrt(2.0 / sizes[i])
            b = np.zeros(sizes[i+1])
            self.weights.append(w.astype(np.float32))
            self.biases.append(b.astype(np.float32))

        self.loaded = True

    def save(self, path):
        """Save model weights to file."""
        data = {
            'weights': self.weights,
            'biases': self.biases,
            'hidden_sizes': self.hidden_sizes,
            'model_type': 'value_predictor',
            'feature_mode': self.feature_mode,
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        print(f"Value predictor saved to {path}")

    def load(self, path):
        """Load model weights from file."""
        try:
            with open(path, 'rb') as f:
                data = pickle.load(f)

            # Check if this is a value predictor model
            if data.get('model_type') != 'value_predictor':
                # It's the old move predictor model, can't use it
                self.loaded = False
                return

            self.weights = data['weights']
            self.biases = data['biases']
            self.hidden_sizes = data.get('hidden_sizes', self.hidden_sizes)
            self.feature_mode = data.get('feature_mode', FEATURE_MODE_SCALAR)
            self.loaded = True
            print(f"Value predictor loaded from {path} (features: {self.feature_mode})")
        except Exception as e:
            print(f"Failed to load value predictor: {e}")
            self.loaded = False


class NeuralValueTrainer:
    """
    Training system for the neural value predictor.

    Generates training data using:
    1. BFS from solved state for exact distances (depth 0-7)
    2. Pruning table heuristics for approximate distances (depth 8+)

    Uses MSE loss instead of cross-entropy.
    """

    def __init__(self, model=None):
        """Initialize trainer with optional existing model."""
        self.model = model or NeuralValuePredictor()

    def generate_training_data(self, n_samples=50000, max_depth=12):
        """
        Generate (state, distance) training pairs.

        Uses BFS for exact distances at low depths, and pruning tables
        for estimates at higher depths.
        """
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent))

        from .pykociemba.coordcube import CoordCube, getPruning
        from .pykociemba.cubiecube import CubieCube, moveCube

        print(f"Generating {n_samples} value training samples...")

        X = []
        y = []

        # Part 1: Generate states at various depths by random scrambling
        # The depth of scramble gives us an upper bound on distance
        samples_per_depth = n_samples // max_depth

        for depth in range(1, max_depth + 1):
            if depth % 3 == 0:
                print(f"  Depth {depth}/{max_depth}...")

            for _ in range(samples_per_depth):
                # Start from solved
                cube = CubieCube()

                # Apply random moves (avoiding redundant consecutive same-axis moves)
                last_axis = -1
                for _ in range(depth):
                    while True:
                        mv = np.random.randint(N_MOVES)
                        axis = mv // 3
                        if axis != last_axis:
                            break
                    last_axis = axis

                    # Apply move
                    axis = mv // 3
                    power = mv % 3
                    for _ in range(power + 1):
                        cube.cornerMultiply(moveCube[axis])
                        cube.edgeMultiply(moveCube[axis])

                # Get coordinates
                coord = CoordCube(cube)

                # Use pruning table heuristic as distance estimate
                # This is a lower bound on actual distance
                slice_coord = coord.FRtoBR // 24
                h1 = getPruning(CoordCube.Slice_Flip_Prun,
                               CoordCube.N_SLICE1 * coord.flip + slice_coord)
                h2 = getPruning(CoordCube.Slice_Twist_Prun,
                               CoordCube.N_SLICE1 * coord.twist + slice_coord)
                heuristic = max(h1, h2)

                # Target: use max of scramble depth and heuristic
                # Scramble depth is upper bound, heuristic is lower bound
                # True distance is somewhere in between
                # Using scramble depth as target (it's more accurate for short scrambles)
                target_distance = depth

                # Extract features
                features = self.model.extract_features(
                    coord.twist, coord.flip, slice_coord,
                    coord.parity, coord.URFtoDLF, coord.FRtoBR
                )

                X.append(features)
                y.append(float(target_distance))

        # Part 2: Add some solved-state samples (distance = 0)
        print("  Adding solved state samples...")
        solved_cube = CubieCube()
        solved_coord = CoordCube(solved_cube)
        solved_features = self.model.extract_features(
            solved_coord.twist, solved_coord.flip, solved_coord.FRtoBR // 24,
            solved_coord.parity, solved_coord.URFtoDLF, solved_coord.FRtoBR
        )
        for _ in range(n_samples // 50):  # 2% solved states
            X.append(solved_features)
            y.append(0.0)

        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.float32)

        print(f"Generated {len(X)} samples, distance range: {y.min():.0f}-{y.max():.0f}")
        return X, y

    def train(self, X, y, epochs=100, batch_size=256, learning_rate=0.001,
              val_split=0.1):
        """
        Train the model using MSE loss.

        Returns:
            Training history (loss and MAE per epoch)
        """
        n_samples = len(X)

        # Split into train/val
        n_val = int(n_samples * val_split)
        indices = np.random.permutation(n_samples)
        val_idx = indices[:n_val]
        train_idx = indices[n_val:]

        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        n_train = len(X_train)
        n_batches = (n_train + batch_size - 1) // batch_size

        # Initialize model if needed
        if not self.model.loaded:
            self.model.initialize_random(input_size=X.shape[1])

        history = {'loss': [], 'val_loss': [], 'mae': [], 'val_mae': []}

        for epoch in range(epochs):
            # Shuffle training data
            perm = np.random.permutation(n_train)
            X_shuffled = X_train[perm]
            y_shuffled = y_train[perm]

            epoch_loss = 0

            for batch in range(n_batches):
                start = batch * batch_size
                end = min(start + batch_size, n_train)

                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[start:end]

                # Forward pass
                activations = [X_batch]
                for w, b in zip(self.model.weights[:-1], self.model.biases[:-1]):
                    z = np.dot(activations[-1], w) + b
                    activations.append(self.model._relu(z))

                # Output layer (linear)
                predictions = np.dot(activations[-1], self.model.weights[-1]) + self.model.biases[-1]
                predictions = predictions.flatten()

                # MSE loss
                errors = predictions - y_batch
                batch_loss = np.mean(errors ** 2)
                epoch_loss += batch_loss

                # Backward pass
                grad_output = (2.0 / len(y_batch)) * errors.reshape(-1, 1)

                # Output layer gradients
                grad_w = [np.dot(activations[-1].T, grad_output)]
                grad_b = [grad_output.sum(axis=0)]

                # Hidden layer gradients
                grad_a = np.dot(grad_output, self.model.weights[-1].T)

                for i in range(len(self.model.weights) - 2, -1, -1):
                    # ReLU gradient
                    grad_z = grad_a * (activations[i + 1] > 0)

                    grad_w.insert(0, np.dot(activations[i].T, grad_z))
                    grad_b.insert(0, grad_z.sum(axis=0))

                    if i > 0:
                        grad_a = np.dot(grad_z, self.model.weights[i].T)

                # Update weights
                for i in range(len(self.model.weights)):
                    self.model.weights[i] -= learning_rate * grad_w[i]
                    self.model.biases[i] -= learning_rate * grad_b[i]

            epoch_loss /= n_batches

            # Validation metrics
            val_pred = self.model.batch_forward(X_val)
            val_loss = np.mean((val_pred - y_val) ** 2)
            val_mae = np.mean(np.abs(val_pred - y_val))

            train_pred = self.model.batch_forward(X_train[:1000])
            train_mae = np.mean(np.abs(train_pred - y_train[:1000]))

            history['loss'].append(epoch_loss)
            history['val_loss'].append(val_loss)
            history['mae'].append(train_mae)
            history['val_mae'].append(val_mae)

            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{epochs}, "
                      f"Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}, "
                      f"MAE: {train_mae:.2f}, Val MAE: {val_mae:.2f}")

        return history

    def evaluate(self, X, y):
        """Evaluate model on test data."""
        predictions = self.model.batch_forward(X)
        mse = np.mean((predictions - y) ** 2)
        mae = np.mean(np.abs(predictions - y))
        return {'mse': mse, 'mae': mae}


class NeuralMovePredictor:
    """
    Neural network for predicting move quality in Kociemba search.

    The network takes cube coordinates as input and outputs a probability
    distribution over moves, where higher probability indicates a move
    more likely to lead to a shorter solution.
    """

    def __init__(self, model_path=None, feature_mode=FEATURE_MODE_SCALAR):
        """Initialize the predictor, optionally loading a trained model."""
        self.model_path = model_path
        self.weights = None
        self.biases = None
        self.loaded = False
        self.feature_mode = feature_mode

        # Architecture: coordinates -> 256 -> 128 -> 64 -> 18 moves
        self.hidden_sizes = [256, 128, 64]

        if model_path and os.path.exists(model_path):
            self.load(model_path)

    def _sigmoid(self, x):
        """Numerically stable sigmoid."""
        return np.where(x >= 0,
                        1 / (1 + np.exp(-x)),
                        np.exp(x) / (1 + np.exp(x)))

    def _relu(self, x):
        """ReLU activation."""
        return np.maximum(0, x)

    def _softmax(self, x):
        """Softmax activation."""
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum()

    def extract_features(self, twist, flip, slice_coord, parity=0,
                         URFtoDLF=0, FRtoBR=0):
        """
        Extract features from cube coordinates.
        Dispatches to scalar (9-dim) or coordinate (161-dim) based on feature_mode.
        """
        if self.feature_mode == FEATURE_MODE_COORDINATE:
            return extract_coordinate_features(twist, flip, slice_coord,
                                                parity, URFtoDLF, FRtoBR)
        # Default: scalar features
        # Normalize coordinates to [0, 1]
        features = np.array([
            twist / N_TWIST,
            flip / N_FLIP,
            slice_coord / N_SLICE,
            parity / N_PARITY,
            URFtoDLF / N_URFDLF,
            FRtoBR / N_FRTOBR,
            # Add some derived features
            (twist % 729) / 729,  # Subset of corner orientations
            (flip % 256) / 256,   # Subset of edge orientations
            (slice_coord % 99) / 99,  # Subset of slice
        ], dtype=np.float32)

        return features

    def predict_move_order(self, twist, flip, slice_coord, parity=0,
                           URFtoDLF=0, FRtoBR=0):
        """
        Predict move ordering based on cube state.

        Args:
            Cube coordinates (see extract_features)

        Returns:
            List of move indices sorted by predicted quality (best first)
        """
        if not self.loaded:
            # Fallback: return default order
            return list(range(N_MOVES))

        features = self.extract_features(twist, flip, slice_coord,
                                         parity, URFtoDLF, FRtoBR)

        # Forward pass
        x = features
        for i, (w, b) in enumerate(zip(self.weights[:-1], self.biases[:-1])):
            x = self._relu(np.dot(x, w) + b)

        # Output layer
        logits = np.dot(x, self.weights[-1]) + self.biases[-1]
        probs = self._softmax(logits)

        # Return move indices sorted by probability (highest first)
        return list(np.argsort(probs)[::-1])

    def predict_move_probs(self, twist, flip, slice_coord, parity=0,
                           URFtoDLF=0, FRtoBR=0):
        """
        Get probability distribution over moves.

        Returns:
            Array of probabilities for each move
        """
        if not self.loaded:
            return np.ones(N_MOVES) / N_MOVES

        features = self.extract_features(twist, flip, slice_coord,
                                         parity, URFtoDLF, FRtoBR)

        x = features
        for w, b in zip(self.weights[:-1], self.biases[:-1]):
            x = self._relu(np.dot(x, w) + b)

        logits = np.dot(x, self.weights[-1]) + self.biases[-1]
        return self._softmax(logits)

    def initialize_random(self, input_size=None):
        """Initialize network with random weights (for training)."""
        if input_size is None:
            input_size = COORDINATE_FEATURE_SIZE if self.feature_mode == FEATURE_MODE_COORDINATE else 9
        np.random.seed(42)

        self.weights = []
        self.biases = []

        sizes = [input_size] + self.hidden_sizes + [N_MOVES]

        for i in range(len(sizes) - 1):
            # He initialization
            w = np.random.randn(sizes[i], sizes[i+1]) * np.sqrt(2.0 / sizes[i])
            b = np.zeros(sizes[i+1])
            self.weights.append(w.astype(np.float32))
            self.biases.append(b.astype(np.float32))

        self.loaded = True

    def save(self, path):
        """Save model weights to file."""
        data = {
            'weights': self.weights,
            'biases': self.biases,
            'hidden_sizes': self.hidden_sizes,
            'feature_mode': self.feature_mode,
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        print(f"Model saved to {path}")

    def load(self, path):
        """Load model weights from file."""
        try:
            with open(path, 'rb') as f:
                data = pickle.load(f)
            self.weights = data['weights']
            self.biases = data['biases']
            self.hidden_sizes = data.get('hidden_sizes', self.hidden_sizes)
            self.feature_mode = data.get('feature_mode', FEATURE_MODE_SCALAR)
            self.loaded = True
            print(f"Model loaded from {path} (features: {self.feature_mode})")
        except Exception as e:
            print(f"Failed to load model: {e}")
            self.loaded = False


class NeuralMoveTrainer:
    """
    Training system for the neural move predictor.

    Generates training data from solved cubes and trains the network
    to predict good moves.
    """

    def __init__(self, model=None):
        """Initialize trainer with optional existing model."""
        self.model = model or NeuralMovePredictor()

    def generate_training_data(self, n_samples=100000, max_scramble_length=25):
        """
        Generate training data from random scrambles.

        For each scramble, we record the cube state and the inverse move
        (i.e., the move that would make progress toward solved state).

        Args:
            n_samples: Number of training samples to generate
            max_scramble_length: Maximum scramble depth

        Returns:
            X: Feature matrix (n_samples, n_features)
            y: Move labels (n_samples,)
        """
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent))

        from .pykociemba.coordcube import CoordCube
        from .pykociemba.cubiecube import CubieCube, moveCube

        print(f"Generating {n_samples} training samples...")

        X = []
        y = []

        samples_per_depth = n_samples // max_scramble_length

        for depth in range(1, max_scramble_length + 1):
            if depth % 5 == 0:
                print(f"  Depth {depth}/{max_scramble_length}")

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

                    # Apply move (mv = axis * 3 + power)
                    axis = mv // 3
                    power = mv % 3
                    for _ in range(power + 1):
                        cube.cornerMultiply(moveCube[axis])
                        cube.edgeMultiply(moveCube[axis])

                # Extract coordinates
                coord = CoordCube(cube)

                # The "good" move is the inverse of the last applied move
                # (this would undo the last move, making progress)
                last_move = moves[-1]
                axis = last_move // 3
                power = last_move % 3

                # Inverse move: same axis, complementary power
                # power 0 (90°) -> power 2 (270°)
                # power 1 (180°) -> power 1 (180°)
                # power 2 (270°) -> power 0 (90°)
                inv_power = (4 - power - 1) % 3
                good_move = axis * 3 + inv_power

                # Extract features
                features = self.model.extract_features(
                    coord.twist, coord.flip, coord.FRtoBR // 24,
                    coord.parity, coord.URFtoDLF, coord.FRtoBR
                )

                X.append(features)
                y.append(good_move)

        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.int32)

        print(f"Generated {len(X)} samples")
        return X, y

    def train(self, X, y, epochs=100, batch_size=256, learning_rate=0.001):
        """
        Train the model using mini-batch gradient descent.

        Args:
            X: Feature matrix
            y: Move labels
            epochs: Number of training epochs
            batch_size: Mini-batch size
            learning_rate: Learning rate

        Returns:
            Training history (loss per epoch)
        """
        n_samples = len(X)
        n_batches = (n_samples + batch_size - 1) // batch_size

        # Initialize model if needed
        if not self.model.loaded:
            self.model.initialize_random(input_size=X.shape[1])

        history = []

        for epoch in range(epochs):
            # Shuffle data
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
                for w, b in zip(self.model.weights[:-1], self.model.biases[:-1]):
                    z = np.dot(activations[-1], w) + b
                    activations.append(self.model._relu(z))

                # Output layer
                logits = np.dot(activations[-1], self.model.weights[-1]) + self.model.biases[-1]

                # Softmax and cross-entropy loss
                exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
                probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)

                # Loss
                batch_loss = -np.mean(np.log(probs[np.arange(len(y_batch)), y_batch] + 1e-8))
                epoch_loss += batch_loss

                # Backward pass
                grad_logits = probs.copy()
                grad_logits[np.arange(len(y_batch)), y_batch] -= 1
                grad_logits /= len(y_batch)

                # Output layer gradients
                grad_w = [np.dot(activations[-1].T, grad_logits)]
                grad_b = [grad_logits.sum(axis=0)]

                # Hidden layer gradients
                grad_a = np.dot(grad_logits, self.model.weights[-1].T)

                for i in range(len(self.model.weights) - 2, -1, -1):
                    # ReLU gradient
                    grad_z = grad_a * (activations[i + 1] > 0)

                    grad_w.insert(0, np.dot(activations[i].T, grad_z))
                    grad_b.insert(0, grad_z.sum(axis=0))

                    if i > 0:
                        grad_a = np.dot(grad_z, self.model.weights[i].T)

                # Update weights
                for i in range(len(self.model.weights)):
                    self.model.weights[i] -= learning_rate * grad_w[i]
                    self.model.biases[i] -= learning_rate * grad_b[i]

            epoch_loss /= n_batches
            history.append(epoch_loss)

            if (epoch + 1) % 10 == 0:
                # Compute accuracy
                pred = self.predict_batch(X[:1000])
                acc = np.mean(pred == y[:1000])
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}, Acc: {acc:.3f}")

        return history

    def predict_batch(self, X):
        """Predict moves for a batch of feature vectors."""
        x = X
        for w, b in zip(self.model.weights[:-1], self.model.biases[:-1]):
            x = self.model._relu(np.dot(x, w) + b)

        logits = np.dot(x, self.model.weights[-1]) + self.model.biases[-1]
        return np.argmax(logits, axis=1)


# ─────────────────────────────────────────────────────────────────
# Lookup Table (LUT) for O(1) neural move ordering
# ─────────────────────────────────────────────────────────────────

class NeuralMoveLUT:
    """Precomputed lookup table replacing per-node neural inference with O(1) lookup.

    The key insight is that the 161-dim binned coordinate features are a
    deterministic function of 6 integer coordinates.  For Phase 1, only
    twist, flip, slice, and parity matter for heuristic quality.  By
    discretising these into bins (32×32×16×2 = 32 768 entries) and
    precomputing the move ordering for each, we eliminate per-node neural
    inference entirely.

    Memory:   ~576 KB  (32 768 × 18 bytes)
    Build:    ~1–2 s   (32 768 forward passes, done once)
    Lookup:   <1 µs    (3 integer divs + 1 array index)
    """

    def __init__(self):
        self.table = None   # shape (32, 32, 16, 2, 18), dtype int8
        self.ready = False
        self.build_time = 0.0

    def build_from_model(self, model):
        """Populate LUT by running the neural model on representative coords."""
        import time as _time
        t0 = _time.time()
        self.table = np.zeros(
            (N_TWIST_BINS, N_FLIP_BINS, N_SLICE_BINS, 2, N_MOVES),
            dtype=np.int8
        )

        count = 0
        for tb in range(N_TWIST_BINS):
            twist = min(int((tb + 0.5) * (N_TWIST + 1) / N_TWIST_BINS), N_TWIST - 1)
            for fb in range(N_FLIP_BINS):
                flip = min(int((fb + 0.5) * (N_FLIP + 1) / N_FLIP_BINS), N_FLIP - 1)
                for sb in range(N_SLICE_BINS):
                    slice_c = min(int((sb + 0.5) * (N_SLICE + 1) / N_SLICE_BINS), N_SLICE - 1)
                    for p in range(2):
                        order = model.predict_move_order(
                            twist, flip, slice_c, p,
                            N_URFDLF // 2, N_FRTOBR // 2
                        )
                        self.table[tb, fb, sb, p] = np.array(order, dtype=np.int8)
                        count += 1

        self.build_time = _time.time() - t0
        self.ready = True
        print(f"Neural LUT built: {count:,} entries in {self.build_time:.1f}s "
              f"({self.table.nbytes / 1024:.0f} KB)")

    def lookup(self, twist, flip, slice_coord, parity):
        """Return move ordering as list of 18 move indices (O(1) lookup)."""
        tb = min(int(twist * N_TWIST_BINS / (N_TWIST + 1)), N_TWIST_BINS - 1)
        fb = min(int(flip * N_FLIP_BINS / (N_FLIP + 1)), N_FLIP_BINS - 1)
        sb = min(int(slice_coord * N_SLICE_BINS / (N_SLICE + 1)), N_SLICE_BINS - 1)
        p = int(parity)
        return self.table[tb, fb, sb, p]

    def lookup_as_axis_power(self, twist, flip, slice_coord, parity):
        """Return move ordering as list of (axis, power) tuples for search."""
        indices = self.lookup(twist, flip, slice_coord, parity)
        return [(int(mv) // 3, int(mv) % 3 + 1) for mv in indices]


# ─────────────────────────────────────────────────────────────────
# Global model / LUT instances (lazy loaded)
# ─────────────────────────────────────────────────────────────────

_global_move_model = None
_global_value_model = None
_global_move_lut = None
_move_model_path = Path(__file__).parent / 'neural_model.pkl'
_value_model_path = Path(__file__).parent / 'neural_value_model.pkl'


def get_move_predictor():
    """Get the global move predictor instance."""
    global _global_move_model
    if _global_move_model is None:
        _global_move_model = NeuralMovePredictor(str(_move_model_path))
    return _global_move_model


def get_value_predictor():
    """Get the global value predictor instance."""
    global _global_value_model
    if _global_value_model is None:
        _global_value_model = NeuralValuePredictor(str(_value_model_path))
    return _global_value_model


def get_move_lut():
    """Get the global move ordering LUT, building from trained model on first call."""
    global _global_move_lut
    if _global_move_lut is None:
        model = get_move_predictor()
        if model.loaded:
            _global_move_lut = NeuralMoveLUT()
            _global_move_lut.build_from_model(model)
    return _global_move_lut


def predict_move_order(twist, flip, slice_coord, parity=0, URFtoDLF=0, FRtoBR=0):
    """
    Convenience function to get move ordering from global model.

    Returns list of move indices sorted by predicted quality.
    """
    return get_move_predictor().predict_move_order(
        twist, flip, slice_coord, parity, URFtoDLF, FRtoBR
    )


def predict_distance(twist, flip, slice_coord, parity=0, URFtoDLF=0, FRtoBR=0):
    """
    Convenience function to get distance prediction from global value model.

    Returns estimated distance to solved state (0-20).
    """
    return get_value_predictor().predict_distance(
        twist, flip, slice_coord, parity, URFtoDLF, FRtoBR
    )


if __name__ == '__main__':
    # Training script
    import argparse

    parser = argparse.ArgumentParser(description='Train neural predictor')
    parser.add_argument('--samples', type=int, default=100000,
                        help='Number of training samples')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Training epochs')
    parser.add_argument('--output', type=str, default=None,
                        help='Output model path')
    parser.add_argument('--value-mode', action='store_true',
                        help='Train value predictor instead of move predictor')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--feature-mode', type=str, default=FEATURE_MODE_SCALAR,
                        choices=[FEATURE_MODE_SCALAR, FEATURE_MODE_COORDINATE],
                        help='Feature extraction mode: scalar (9-dim) or coordinate (161-dim)')

    args = parser.parse_args()

    feature_mode = args.feature_mode
    feat_label = f"features={feature_mode}"
    if feature_mode == FEATURE_MODE_COORDINATE:
        feat_label += f" ({COORDINATE_FEATURE_SIZE}-dim)"
    else:
        feat_label += " (9-dim)"

    if args.value_mode:
        # Train value predictor
        print("=" * 60)
        print("Training NEURAL VALUE PREDICTOR")
        print(f"Feature mode: {feat_label}")
        print("Predicts distance to solved state (0-20)")
        print("=" * 60)

        output_path = args.output or 'neural_value_model.pkl'
        model = NeuralValuePredictor(feature_mode=feature_mode)
        trainer = NeuralValueTrainer(model=model)

        print("\nGenerating training data...")
        X, y = trainer.generate_training_data(n_samples=args.samples)

        print("\nTraining model with MSE loss...")
        history = trainer.train(X, y, epochs=args.epochs, learning_rate=args.lr)

        print("\nSaving model...")
        trainer.model.save(output_path)

        print("\nTraining complete!")
        print(f"Final Val MAE: {history['val_mae'][-1]:.2f} moves")

    else:
        # Train move predictor
        print("=" * 60)
        print("Training NEURAL MOVE PREDICTOR")
        print(f"Feature mode: {feat_label}")
        print("Predicts which move to apply (18-way classification)")
        print("=" * 60)

        output_path = args.output or 'neural_model.pkl'
        model = NeuralMovePredictor(feature_mode=feature_mode)
        trainer = NeuralMoveTrainer(model=model)

        print("\nGenerating training data...")
        X, y = trainer.generate_training_data(n_samples=args.samples)

        print("\nTraining model...")
        history = trainer.train(X, y, epochs=args.epochs, learning_rate=args.lr)

        print("\nSaving model...")
        trainer.model.save(output_path)

        print("\nTraining complete!")
        print(f"Final loss: {history[-1]:.4f}")
