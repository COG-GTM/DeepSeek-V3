import pytest
import os
import sys
import tempfile
import json
from unittest.mock import Mock, MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

@pytest.fixture
def mock_model_args():
    """Mock ModelArgs for testing."""
    mock_args = Mock()
    mock_args.dim = 4096
    mock_args.n_layers = 32
    mock_args.n_heads = 32
    mock_args.vocab_size = 102400
    mock_args.rope_theta = 10000.0
    mock_args.max_seq_len = 4096
    mock_args.n_routed_experts = 64
    mock_args.n_activated_experts = 6
    mock_args.n_shared_experts = 2
    mock_args.rope_factor = 1.0
    mock_args.mscale = 1.0
    mock_args.score_func = "sigmoid"
    mock_args.q_lora_rank = 1536
    mock_args.kv_lora_rank = 512
    mock_args.qk_rope_head_dim = 64
    mock_args.v_head_dim = 128
    mock_args.qk_nope_head_dim = 128
    return mock_args

@pytest.fixture
def mock_tokenizer():
    """Mock tokenizer for testing."""
    tokenizer = Mock()
    tokenizer.encode.return_value = [1, 2, 3, 4, 5]
    tokenizer.decode.return_value = "test response"
    tokenizer.eos_token_id = 2
    tokenizer.pad_token_id = 0
    return tokenizer

@pytest.fixture
def mock_gpqa_data():
    """Mock GPQA dataset for testing."""
    return [
        {
            "question": "What is the capital of France?",
            "choices": ["A) London", "B) Paris", "C) Berlin", "D) Madrid"],
            "answer": "B",
            "category": "geography"
        },
        {
            "question": "What is 2+2?",
            "choices": ["A) 3", "B) 4", "C) 5", "D) 6"],
            "answer": "B",
            "category": "math"
        }
    ]

@pytest.fixture
def temp_dir():
    """Create temporary directory for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir

@pytest.fixture
def mock_torch_tensor():
    """Mock torch tensor for testing."""
    tensor = Mock()
    tensor.size.return_value = Mock()
    tensor.size.return_value.__getitem__ = lambda self, idx: [10, 20][idx]
    tensor.dtype = "float32"
    tensor.device = "cpu"
    tensor.shape = [10, 20]
    return tensor

@pytest.fixture
def mock_safetensors_data():
    """Mock safetensors data for testing."""
    return {
        "weight1": mock_torch_tensor(),
        "weight2": mock_torch_tensor(),
        "weight1_scale_inv": mock_torch_tensor(),
        "weight2_scale_inv": mock_torch_tensor()
    }

@pytest.fixture
def mock_model_index():
    """Mock model index file for testing."""
    return {
        "metadata": {},
        "weight_map": {
            "weight1": "model-00001.safetensors",
            "weight2": "model-00001.safetensors",
            "weight1_scale_inv": "model-00001.safetensors",
            "weight2_scale_inv": "model-00001.safetensors"
        }
    }
