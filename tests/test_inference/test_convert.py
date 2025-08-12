import pytest
import os
import tempfile
import json
from unittest.mock import Mock, patch, MagicMock
import torch

from inference.convert import main

class TestConvert:
    
    @patch('inference.convert.glob.glob')
    @patch('inference.convert.load_file')
    @patch('inference.convert.save_file')
    @patch('inference.convert.shutil.copy2')
    def test_main_basic_conversion(self, mock_copy, mock_save, mock_load, mock_glob, temp_dir):
        """Test basic checkpoint conversion."""
        input_dir = os.path.join(temp_dir, "input")
        output_dir = os.path.join(temp_dir, "output")
        os.makedirs(input_dir, exist_ok=True)
        
        checkpoint_files = [
            os.path.join(input_dir, "pytorch_model-00001-of-00002.bin"),
            os.path.join(input_dir, "pytorch_model-00002-of-00002.bin")
        ]
        mock_glob.return_value = checkpoint_files
        
        mock_weights = {
            "model.layers.0.attention.wq.weight": torch.randn(100, 200),
            "model.layers.0.attention.wk.weight": torch.randn(100, 200),
            "model.layers.0.mlp.gate_proj.weight": torch.randn(150, 200)
        }
        mock_load.return_value = mock_weights
        
        main(input_dir, output_dir, n_experts=8, mp=2)
        
        assert mock_save.call_count == 2  # One for each MP rank
        
        mock_copy.assert_called()
    
    @patch('inference.convert.glob.glob')
    @patch('inference.convert.load_file')
    @patch('inference.convert.save_file')
    def test_main_with_expert_weights(self, mock_save, mock_load, mock_glob, temp_dir):
        """Test conversion with expert weights."""
        input_dir = os.path.join(temp_dir, "input")
        output_dir = os.path.join(temp_dir, "output")
        os.makedirs(input_dir, exist_ok=True)
        
        mock_glob.return_value = [os.path.join(input_dir, "pytorch_model.bin")]
        
        mock_weights = {
            "model.layers.0.mlp.experts.0.gate_proj.weight": torch.randn(100, 200),
            "model.layers.0.mlp.experts.1.gate_proj.weight": torch.randn(100, 200),
            "model.layers.0.mlp.shared_experts.gate_proj.weight": torch.randn(100, 200)
        }
        mock_load.return_value = mock_weights
        
        main(input_dir, output_dir, n_experts=2, mp=1)
        
        mock_save.assert_called()
    
    @patch('inference.convert.glob.glob')
    def test_main_no_checkpoint_files(self, mock_glob, temp_dir):
        """Test handling when no checkpoint files are found."""
        input_dir = os.path.join(temp_dir, "input")
        output_dir = os.path.join(temp_dir, "output")
        
        mock_glob.return_value = []
        
        try:
            main(input_dir, output_dir, n_experts=8, mp=2)
        except (FileNotFoundError, ValueError):
            pass
    
    @patch('inference.convert.glob.glob')
    @patch('inference.convert.load_file')
    @patch('inference.convert.save_file')
    def test_main_parameter_mapping(self, mock_save, mock_load, mock_glob, temp_dir):
        """Test parameter name mapping during conversion."""
        input_dir = os.path.join(temp_dir, "input")
        output_dir = os.path.join(temp_dir, "output")
        
        mock_glob.return_value = [os.path.join(input_dir, "pytorch_model.bin")]
        
        original_weights = {
            "model.embed_tokens.weight": torch.randn(1000, 512),
            "model.layers.0.self_attn.q_proj.weight": torch.randn(512, 512),
            "model.layers.0.self_attn.k_proj.weight": torch.randn(512, 512),
            "model.layers.0.self_attn.v_proj.weight": torch.randn(512, 512),
            "model.layers.0.self_attn.o_proj.weight": torch.randn(512, 512)
        }
        mock_load.return_value = original_weights
        
        main(input_dir, output_dir, n_experts=8, mp=1)
        
        mock_save.assert_called()
        
        saved_call = mock_save.call_args_list[0]
        saved_weights = saved_call[0][0]  # First argument to save_file
        
        assert any("tok_embeddings" in key for key in saved_weights.keys())
    
    @patch('inference.convert.glob.glob')
    @patch('inference.convert.load_file')
    @patch('inference.convert.save_file')
    @patch('os.makedirs')
    def test_main_creates_output_directory(self, mock_makedirs, mock_save, mock_load, mock_glob, temp_dir):
        """Test that output directory is created."""
        input_dir = os.path.join(temp_dir, "input")
        output_dir = os.path.join(temp_dir, "output")
        
        mock_glob.return_value = [os.path.join(input_dir, "pytorch_model.bin")]
        mock_load.return_value = {"test.weight": torch.randn(10, 10)}
        
        main(input_dir, output_dir, n_experts=8, mp=1)
        
        mock_makedirs.assert_called_with(output_dir, exist_ok=True)
    
    def test_main_with_different_mp_values(self, temp_dir):
        """Test conversion with different model parallelism values."""
        input_dir = os.path.join(temp_dir, "input")
        output_dir = os.path.join(temp_dir, "output")
        
        with patch('inference.convert.glob.glob') as mock_glob, \
             patch('inference.convert.load_file') as mock_load, \
             patch('inference.convert.save_file') as mock_save:
            
            mock_glob.return_value = [os.path.join(input_dir, "pytorch_model.bin")]
            mock_load.return_value = {"test.weight": torch.randn(100, 100)}
            
            main(input_dir, output_dir, n_experts=8, mp=1)
            assert mock_save.call_count == 1
            
            mock_save.reset_mock()
            
            main(input_dir, output_dir, n_experts=8, mp=4)
            assert mock_save.call_count == 4
