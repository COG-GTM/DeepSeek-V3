import pytest
import os
import tempfile
import json
from unittest.mock import Mock, patch, MagicMock
import torch

from inference.fp8_cast_bf16 import main

class TestFp8CastBf16:
    
    @patch('inference.fp8_cast_bf16.torch.set_default_dtype')
    @patch('inference.fp8_cast_bf16.os.makedirs')
    @patch('inference.fp8_cast_bf16.glob.glob')
    @patch('inference.fp8_cast_bf16.load_file')
    @patch('inference.fp8_cast_bf16.save_file')
    def test_main_basic_conversion(self, mock_save, mock_load, mock_glob, mock_makedirs, mock_set_dtype, temp_dir):
        """Test basic FP8 to BF16 conversion."""
        fp8_path = os.path.join(temp_dir, "fp8")
        bf16_path = os.path.join(temp_dir, "bf16")
        
        model_index = {
            "weight_map": {
                "layer.weight": "model-00001.safetensors",
                "layer.weight_scale_inv": "model-00001.safetensors",
                "other.weight": "model-00001.safetensors"
            }
        }
        
        with patch('builtins.open', create=True) as mock_open:
            mock_file = MagicMock()
            mock_open.return_value.__enter__.return_value = mock_file
            mock_file.read.return_value = json.dumps(model_index)
            
            mock_glob.return_value = [os.path.join(fp8_path, "model-00001.safetensors")]
            
            fp8_weight = torch.randint(0, 255, (100, 200), dtype=torch.uint8)
            fp8_weight.element_size = Mock(return_value=1)  # FP8 element size
            
            scale_inv = torch.randn(100)
            other_weight = torch.randn(50, 100)
            other_weight.element_size = Mock(return_value=4)  # BF16 element size
            
            mock_weights = {
                "layer.weight": fp8_weight,
                "layer.weight_scale_inv": scale_inv,
                "other.weight": other_weight
            }
            mock_load.return_value = mock_weights
            
            with patch('inference.fp8_cast_bf16.weight_dequant') as mock_dequant:
                mock_dequant.return_value = torch.randn(100, 200, dtype=torch.bfloat16)
                
                main(fp8_path, bf16_path)
                
                mock_set_dtype.assert_called_with(torch.bfloat16)
                mock_makedirs.assert_called_with(bf16_path, exist_ok=True)
                mock_dequant.assert_called_once_with(fp8_weight, scale_inv)
                mock_save.assert_called()
    
    @patch('inference.fp8_cast_bf16.glob.glob')
    @patch('inference.fp8_cast_bf16.load_file')
    def test_main_missing_scale_inv(self, mock_load, mock_glob, temp_dir):
        """Test handling of missing scale_inv tensors."""
        fp8_path = os.path.join(temp_dir, "fp8")
        bf16_path = os.path.join(temp_dir, "bf16")
        
        model_index = {
            "weight_map": {
                "layer.weight": "model-00001.safetensors"
            }
        }
        
        with patch('builtins.open', create=True) as mock_open:
            mock_file = MagicMock()
            mock_open.return_value.__enter__.return_value = mock_file
            mock_file.read.return_value = json.dumps(model_index)
            
            mock_glob.return_value = [os.path.join(fp8_path, "model-00001.safetensors")]
            
            fp8_weight = torch.randint(0, 255, (100, 200), dtype=torch.uint8)
            fp8_weight.element_size = Mock(return_value=1)
            
            mock_weights = {"layer.weight": fp8_weight}
            mock_load.return_value = mock_weights
            
            with patch('inference.fp8_cast_bf16.save_file') as mock_save, \
                 patch('builtins.print') as mock_print:
                
                main(fp8_path, bf16_path)
                
                warning_calls = [call for call in mock_print.call_args_list 
                               if "Warning: Missing scale_inv tensor" in str(call)]
                assert len(warning_calls) > 0
    
    @patch('inference.fp8_cast_bf16.glob.glob')
    @patch('inference.fp8_cast_bf16.load_file')
    @patch('inference.fp8_cast_bf16.save_file')
    def test_main_memory_management(self, mock_save, mock_load, mock_glob, temp_dir):
        """Test memory management with multiple files."""
        fp8_path = os.path.join(temp_dir, "fp8")
        bf16_path = os.path.join(temp_dir, "bf16")
        
        model_index = {"weight_map": {}}
        
        with patch('builtins.open', create=True) as mock_open:
            mock_file = MagicMock()
            mock_open.return_value.__enter__.return_value = mock_file
            mock_file.read.return_value = json.dumps(model_index)
            
            mock_glob.return_value = [
                os.path.join(fp8_path, "model-00001.safetensors"),
                os.path.join(fp8_path, "model-00002.safetensors"),
                os.path.join(fp8_path, "model-00003.safetensors"),
                os.path.join(fp8_path, "model-00004.safetensors")
            ]
            
            mock_load.return_value = {}
            
            with patch('torch.cuda.empty_cache') as mock_empty_cache:
                main(fp8_path, bf16_path)
                
                assert mock_empty_cache.call_count > 0
    
    @patch('inference.fp8_cast_bf16.glob.glob')
    @patch('inference.fp8_cast_bf16.load_file')
    @patch('inference.fp8_cast_bf16.save_file')
    def test_main_model_index_update(self, mock_save, mock_load, mock_glob, temp_dir):
        """Test that model index is properly updated."""
        fp8_path = os.path.join(temp_dir, "fp8")
        bf16_path = os.path.join(temp_dir, "bf16")
        
        model_index = {
            "weight_map": {
                "layer1.weight": "model-00001.safetensors",
                "layer1.weight_scale_inv": "model-00001.safetensors",
                "layer2.weight": "model-00001.safetensors",
                "layer2.weight_scale_inv": "model-00001.safetensors",
                "layer3.weight": "model-00001.safetensors"
            }
        }
        
        written_data = []
        
        def mock_write(data):
            written_data.append(data)
        
        with patch('builtins.open', create=True) as mock_open:
            mock_file = MagicMock()
            mock_open.return_value.__enter__.return_value = mock_file
            
            mock_file.read.return_value = json.dumps(model_index)
            
            mock_file.write.side_effect = mock_write
            
            mock_glob.return_value = [os.path.join(fp8_path, "model-00001.safetensors")]
            
            fp8_weight1 = torch.randint(0, 255, (10, 20), dtype=torch.uint8)
            fp8_weight1.element_size = Mock(return_value=1)
            fp8_weight2 = torch.randint(0, 255, (30, 40), dtype=torch.uint8)
            fp8_weight2.element_size = Mock(return_value=1)
            
            mock_weights = {
                "layer1.weight": fp8_weight1,
                "layer1.weight_scale_inv": torch.randn(10),
                "layer2.weight": fp8_weight2,
                "layer2.weight_scale_inv": torch.randn(30),
                "layer3.weight": torch.randn(50, 60)  # Non-FP8 weight
            }
            mock_load.return_value = mock_weights
            
            with patch('inference.fp8_cast_bf16.weight_dequant') as mock_dequant:
                mock_dequant.return_value = torch.randn(10, 20, dtype=torch.bfloat16)
                
                main(fp8_path, bf16_path)
                
                if written_data:
                    updated_index = json.loads(written_data[0])
                    weight_map = updated_index["weight_map"]
                    
                    assert "layer1.weight_scale_inv" not in weight_map
                    assert "layer2.weight_scale_inv" not in weight_map
                    
                    assert "layer1.weight" in weight_map
                    assert "layer2.weight" in weight_map
                    assert "layer3.weight" in weight_map

class TestGetTensorHelper:
    
    def test_get_tensor_function_exists(self):
        """Test that get_tensor helper function works correctly."""
        
        fp8_path = "/fake/path"
        bf16_path = "/fake/bf16"
        
        model_index = {
            "weight_map": {
                "test.weight": "model-00001.safetensors",
                "test.weight_scale_inv": "model-00002.safetensors"
            }
        }
        
        with patch('builtins.open', create=True) as mock_open, \
             patch('inference.fp8_cast_bf16.glob.glob') as mock_glob, \
             patch('inference.fp8_cast_bf16.load_file') as mock_load, \
             patch('inference.fp8_cast_bf16.save_file'):
            
            mock_file = MagicMock()
            mock_open.return_value.__enter__.return_value = mock_file
            mock_file.read.return_value = json.dumps(model_index)
            
            mock_glob.return_value = ["/fake/path/model-00001.safetensors"]
            
            def load_file_side_effect(filepath, device):
                if "model-00001" in filepath:
                    weight = torch.randint(0, 255, (10, 20), dtype=torch.uint8)
                    weight.element_size = Mock(return_value=1)
                    return {"test.weight": weight}
                elif "model-00002" in filepath:
                    return {"test.weight_scale_inv": torch.randn(10)}
                return {}
            
            mock_load.side_effect = load_file_side_effect
            
            with patch('inference.fp8_cast_bf16.weight_dequant') as mock_dequant:
                mock_dequant.return_value = torch.randn(10, 20, dtype=torch.bfloat16)
                
                main(fp8_path, bf16_path)
                
                mock_dequant.assert_called()
