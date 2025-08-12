"""Utility functions for testing."""

import json
import os
import tempfile
from unittest.mock import Mock

def create_mock_dataset_file(data, filepath):
    """Create a mock dataset file for testing."""
    with open(filepath, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')

def create_mock_config_file(config, filepath):
    """Create a mock configuration file for testing."""
    with open(filepath, 'w') as f:
        json.dump(config, f, indent=2)

def create_mock_inference_wrapper():
    """Create a mock inference wrapper for testing."""
    wrapper = Mock()
    wrapper.generate.return_value = "B"
    wrapper.batch_generate.return_value = ["B", "A", "C"]
    return wrapper
