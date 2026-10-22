"""
Unit test for ProcessDataModule with input blanking and dataset size reduction.

Tests:
1. Dataset size reduction with max_data_size parameter
2. Input blanking with Bernoulli(beta) sampling
3. Combined blanking and size reduction

Run with: pytest test/unit/test_dataloader.py -v
"""

import pytest
from os.path import dirname, abspath, join
import sys
import tempfile
import shutil
sys.path.append(dirname(dirname(abspath(__file__))))

from proT.training.dataloader import ProcessDataModule
import torch
import numpy as np


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test data."""
    tmp_dir = tempfile.mkdtemp()
    yield tmp_dir
    shutil.rmtree(tmp_dir)


def create_test_data(num_samples=100, seq_len=50, num_features=5):
    """
    Create synthetic test data without any NaNs initially.
    
    Args:
        num_samples: Number of samples in dataset
        seq_len: Length of each sequence
        num_features: Number of features
        
    Returns:
        tuple: (X_np, Y_np) numpy arrays
    """
    # Create data without NaNs
    X_np = np.random.randn(num_samples, seq_len, num_features).astype('float32')
    Y_np = np.random.randn(num_samples, seq_len, 1).astype('float32')
    
    return X_np, Y_np


def count_nans(tensor, feature_idx=0):
    """
    Count number of NaN values in a specific feature of a tensor.
    
    Args:
        tensor: torch.Tensor of shape (B, L, D)
        feature_idx: Index of feature to check for NaNs
        
    Returns:
        int: Total number of NaN values in the specified feature
    """
    return torch.isnan(tensor[:, :, feature_idx]).sum().item()


def test_no_blanking_no_reduction():
    """Test baseline: no blanking, no size reduction."""
    print("\n" + "="*60)
    print("TEST 1: No Blanking, No Size Reduction")
    print("="*60)
    
    # Setup
    num_samples = 100
    seq_len = 50
    num_features = 5
    
    # Create temporary directory and data
    temp_dir = tempfile.mkdtemp()
    X_np, Y_np = create_test_data(num_samples, seq_len, num_features)
    
    # Save data
    data_file = join(temp_dir, "test_data.npz")
    np.savez(data_file, x=X_np, y=Y_np)
    
    try:
        # Create datamodule without blanking
        dm = ProcessDataModule(
            data_dir=temp_dir,
            input_file="test_data.npz",
            target_file="test_data.npz",
            batch_size=10,
            num_workers=0,
            data_format="float32",
            max_data_size=None,
            input_p_blank=None,
            input_blanking_val_idx=0,
            seed=42
        )
        
        dm.prepare_data()
        dm.setup(stage=None)
        
        # Get dataset length
        actual_len = len(dm.train_ds) + len(dm.val_ds) + len(dm.test_ds)
        
        # Count NaNs in the original tensor (before splitting)
        # Since random_split creates Subset objects, we check the original tensor
        all_X = dm.X_tensor
        nan_count = count_nans(all_X, feature_idx=0)
        
        print(f"Expected length: {num_samples}")
        print(f"Actual length: {actual_len}")
        print(f"NaN count in feature 0: {nan_count}")
        
        # Assertions
        assert actual_len == num_samples, f"Dataset length mismatch: expected {num_samples}, got {actual_len}"
        assert nan_count == 0, f"Found {nan_count} NaNs when none expected"
        
        print("✓ Test passed!")
        
    finally:
        # Cleanup
        shutil.rmtree(temp_dir)


def test_with_blanking():
    """Test input blanking with Bernoulli(beta) sampling."""
    print("\n" + "="*60)
    print("TEST 2: With Input Blanking (beta=0.2)")
    print("="*60)
    
    # Setup
    num_samples = 100
    seq_len = 50
    num_features = 5
    beta = 0.2
    feature_idx = 0
    
    # Create temporary directory and data
    temp_dir = tempfile.mkdtemp()
    X_np, Y_np = create_test_data(num_samples, seq_len, num_features)
    
    # Save data
    data_file = join(temp_dir, "test_data.npz")
    np.savez(data_file, x=X_np, y=Y_np)
    
    try:
        # Create datamodule with blanking
        dm = ProcessDataModule(
            data_dir=temp_dir,
            input_file="test_data.npz",
            target_file="test_data.npz",
            batch_size=10,
            num_workers=0,
            data_format="float32",
            max_data_size=None,
            input_p_blank=beta,
            input_blanking_val_idx=feature_idx,
            seed=42
        )
        
        dm.prepare_data()
        dm.setup(stage=None)
        
        # Get all data tensor
        all_X = dm.X_tensor
        total_positions = all_X.shape[0] * all_X.shape[1]
        
        # Count NaNs in specified feature
        nan_count = count_nans(all_X, feature_idx=feature_idx)
        nan_ratio = nan_count / total_positions
        
        print(f"Total positions (B * L): {total_positions}")
        print(f"NaN count in feature {feature_idx}: {nan_count}")
        print(f"NaN ratio: {nan_ratio:.3f}")
        print(f"Expected ratio (beta): {beta}")
        print(f"Difference: {abs(nan_ratio - beta):.3f}")
        
        # Allow 5% tolerance in blanking ratio due to random sampling
        tolerance = 0.05
        assert abs(nan_ratio - beta) < tolerance, \
            f"NaN ratio {nan_ratio:.3f} deviates too much from expected {beta}"
        
        # Check that other features don't have NaNs
        for i in range(num_features):
            if i != feature_idx:
                other_nan_count = count_nans(all_X, feature_idx=i)
                assert other_nan_count == 0, \
                    f"Feature {i} has {other_nan_count} NaNs, but only feature {feature_idx} should be blanked"
        
        print("✓ Test passed!")
        
    finally:
        # Cleanup
        shutil.rmtree(temp_dir)


def test_with_size_reduction():
    """Test dataset size reduction with max_data_size."""
    print("\n" + "="*60)
    print("TEST 3: With Size Reduction (max_data_size=50)")
    print("="*60)
    
    # Setup
    num_samples = 100
    max_data_size = 50
    seq_len = 50
    num_features = 5
    
    # Create temporary directory and data
    temp_dir = tempfile.mkdtemp()
    X_np, Y_np = create_test_data(num_samples, seq_len, num_features)
    
    # Save data
    data_file = join(temp_dir, "test_data.npz")
    np.savez(data_file, x=X_np, y=Y_np)
    
    try:
        # Create datamodule with size reduction
        dm = ProcessDataModule(
            data_dir=temp_dir,
            input_file="test_data.npz",
            target_file="test_data.npz",
            batch_size=10,
            num_workers=0,
            data_format="float32",
            max_data_size=max_data_size,
            input_p_blank=None,
            input_blanking_val_idx=0,
            seed=42
        )
        
        dm.prepare_data()
        dm.setup(stage=None)
        
        # Check dataset length
        actual_len = len(dm.train_ds) + len(dm.val_ds) + len(dm.test_ds)
        
        print(f"Original dataset size: {num_samples}")
        print(f"Max data size: {max_data_size}")
        print(f"Actual dataset size: {actual_len}")
        
        assert actual_len == max_data_size, \
            f"Dataset length mismatch: expected {max_data_size}, got {actual_len}"
        
        print("✓ Test passed!")
        
    finally:
        # Cleanup
        shutil.rmtree(temp_dir)


def test_combined_blanking_and_reduction():
    """Test combined input blanking and size reduction."""
    print("\n" + "="*60)
    print("TEST 4: Combined Blanking and Size Reduction")
    print("="*60)
    
    # Setup
    num_samples = 100
    max_data_size = 60
    seq_len = 50
    num_features = 5
    beta = 0.3
    feature_idx = 1
    
    # Create temporary directory and data
    temp_dir = tempfile.mkdtemp()
    X_np, Y_np = create_test_data(num_samples, seq_len, num_features)
    
    # Save data
    data_file = join(temp_dir, "test_data.npz")
    np.savez(data_file, x=X_np, y=Y_np)
    
    try:
        # Create datamodule with both features
        dm = ProcessDataModule(
            data_dir=temp_dir,
            input_file="test_data.npz",
            target_file="test_data.npz",
            batch_size=10,
            num_workers=0,
            data_format="float32",
            max_data_size=max_data_size,
            input_p_blank=beta,
            input_blanking_val_idx=feature_idx,
            seed=42
        )
        
        dm.prepare_data()
        dm.setup(stage=None)
        
        # Check dataset length
        actual_len = len(dm.train_ds) + len(dm.val_ds) + len(dm.test_ds)
        
        # Count NaNs
        all_X = dm.X_tensor
        total_positions = all_X.shape[0] * all_X.shape[1]
        nan_count = count_nans(all_X, feature_idx=feature_idx)
        nan_ratio = nan_count / total_positions
        
        print(f"Original dataset size: {num_samples}")
        print(f"Reduced dataset size: {actual_len}")
        print(f"Expected size: {max_data_size}")
        print(f"Total positions in feature {feature_idx}: {total_positions}")
        print(f"NaN count: {nan_count}")
        print(f"NaN ratio: {nan_ratio:.3f}")
        print(f"Expected NaN ratio (beta): {beta}")
        
        # Assertions
        assert actual_len == max_data_size, \
            f"Dataset length mismatch: expected {max_data_size}, got {actual_len}"
        
        tolerance = 0.05
        assert abs(nan_ratio - beta) < tolerance, \
            f"NaN ratio {nan_ratio:.3f} deviates too much from expected {beta}"
        
        print("✓ Test passed!")
        
    finally:
        # Cleanup
        shutil.rmtree(temp_dir)


def test_presplit_data_with_blanking():
    """Test input blanking with pre-split train/test files."""
    print("\n" + "="*60)
    print("TEST 5: Pre-split Data with Blanking (train only)")
    print("="*60)
    
    # Setup
    num_train = 80
    num_test = 20
    seq_len = 50
    num_features = 5
    beta = 0.25
    feature_idx = 0
    
    # Create temporary directory and data
    temp_dir = tempfile.mkdtemp()
    X_train, Y_train = create_test_data(num_train, seq_len, num_features)
    X_test, Y_test = create_test_data(num_test, seq_len, num_features)
    
    # Save pre-split data
    train_file = join(temp_dir, "train.npz")
    test_file = join(temp_dir, "test.npz")
    np.savez(train_file, x=X_train, y=Y_train)
    np.savez(test_file, x=X_test, y=Y_test)
    
    try:
        # Create datamodule with pre-split data and blanking
        dm = ProcessDataModule(
            data_dir=temp_dir,
            input_file=None,
            target_file=None,
            batch_size=10,
            num_workers=0,
            data_format="float32",
            max_data_size=None,
            input_p_blank=beta,
            input_blanking_val_idx=feature_idx,
            seed=42,
            train_file="train.npz",
            test_file="test.npz"
        )
        
        dm.prepare_data()
        dm.setup(stage=None)
        
        # Check train data has NaNs
        train_X = dm.X_train_tensor
        train_total = train_X.shape[0] * train_X.shape[1]
        train_nan_count = count_nans(train_X, feature_idx=feature_idx)
        train_nan_ratio = train_nan_count / train_total
        
        # Check test data has NO NaNs (blanking should only affect train)
        test_X = dm.X_test_tensor
        test_nan_count = count_nans(test_X, feature_idx=feature_idx)
        
        print(f"Train data:")
        print(f"  Total positions: {train_total}")
        print(f"  NaN count: {train_nan_count}")
        print(f"  NaN ratio: {train_nan_ratio:.3f}")
        print(f"  Expected ratio (beta): {beta}")
        print(f"\nTest data:")
        print(f"  NaN count: {test_nan_count}")
        print(f"  Expected: 0 (no blanking on test data)")
        
        # Assertions
        tolerance = 0.05
        assert abs(train_nan_ratio - beta) < tolerance, \
            f"Train NaN ratio {train_nan_ratio:.3f} deviates from expected {beta}"
        
        assert test_nan_count == 0, \
            f"Test data has {test_nan_count} NaNs, but blanking should only affect train data"
        
        print("✓ Test passed!")
        
    finally:
        # Cleanup
        shutil.rmtree(temp_dir)


def main():
    """Run all tests."""
    print("\n" + "#"*60)
    print("# DATALOADER TESTS: Input Blanking & Size Reduction")
    print("#"*60)
    
    try:
        test_no_blanking_no_reduction()
        test_with_blanking()
        test_with_size_reduction()
        test_combined_blanking_and_reduction()
        test_presplit_data_with_blanking()
        
        print("\n" + "="*60)
        print("ALL TESTS PASSED! ✓")
        print("="*60 + "\n")
        
    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}\n")
        raise
    except Exception as e:
        print(f"\n✗ ERROR: {e}\n")
        raise


if __name__ == "__main__":
    main()
