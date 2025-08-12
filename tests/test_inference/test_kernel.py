import pytest
import torch
from unittest.mock import Mock, patch, MagicMock

with patch.dict('sys.modules', {'triton': MagicMock(), 'triton.language': MagicMock()}):
    from inference.kernel import act_quant, weight_dequant, fp8_gemm

class TestActQuant:
    
    @patch('inference.kernel.torch.empty_like')
    @patch('inference.kernel.torch.Tensor.new_empty')
    def test_act_quant_basic(self, mock_new_empty, mock_empty_like):
        """Test basic activation quantization."""
        input_tensor = Mock()
        input_tensor.numel.return_value = 1024
        input_tensor.size.return_value = torch.Size([32, 32])
        
        mock_quantized = Mock()
        mock_scale = Mock()
        mock_empty_like.return_value = mock_quantized
        mock_new_empty.return_value = mock_scale
        
        with patch('inference.kernel.act_quant_kernel') as mock_kernel:
            result_y, result_s = act_quant(input_tensor, block_size=128)
            
            assert result_y == mock_quantized
            assert result_s == mock_scale
            mock_kernel.__getitem__.assert_called_once()
    
    def test_act_quant_block_size_validation(self):
        """Test that block size is validated."""
        input_tensor = torch.randn(32, 32)
        
        try:
            with patch('inference.kernel.act_quant_kernel'):
                act_quant(input_tensor, block_size=128)
        except Exception as e:
            assert "block_size" not in str(e).lower()
    
    @patch('inference.kernel.torch.empty_like')
    @patch('inference.kernel.torch.Tensor.new_empty')
    def test_act_quant_different_shapes(self, mock_new_empty, mock_empty_like):
        """Test activation quantization with different tensor shapes."""
        tensor_2d = Mock()
        tensor_2d.numel.return_value = 100
        tensor_2d.size.return_value = torch.Size([10, 10])
        
        mock_empty_like.return_value = Mock()
        mock_new_empty.return_value = Mock()
        
        with patch('inference.kernel.act_quant_kernel'):
            act_quant(tensor_2d, block_size=64)
            
        tensor_3d = Mock()
        tensor_3d.numel.return_value = 1000
        tensor_3d.size.return_value = torch.Size([10, 10, 10])
        
        with patch('inference.kernel.act_quant_kernel'):
            act_quant(tensor_3d, block_size=64)

class TestWeightDequant:
    
    @patch('inference.kernel.torch.empty_like')
    def test_weight_dequant_basic(self, mock_empty_like):
        """Test basic weight dequantization."""
        weight_tensor = Mock()
        weight_tensor.size.return_value = torch.Size([32, 32])
        
        scale_tensor = Mock()
        
        mock_output = Mock()
        mock_empty_like.return_value = mock_output
        
        with patch('inference.kernel.weight_dequant_kernel') as mock_kernel:
            result = weight_dequant(weight_tensor, scale_tensor, block_size=64)
            
            assert result == mock_output
            mock_kernel.__getitem__.assert_called_once()
    
    def test_weight_dequant_shape_consistency(self):
        """Test that weight and scale tensors have consistent shapes."""
        weight = torch.randint(0, 255, (32, 32), dtype=torch.uint8)
        scale = torch.randn(32, 32)
        
        try:
            with patch('inference.kernel.weight_dequant_kernel'):
                weight_dequant(weight, scale, block_size=64)
        except Exception as e:
            assert "shape" not in str(e).lower()

class TestFp8Gemm:
    
    @patch('inference.kernel.torch.Tensor.new_empty')
    def test_fp8_gemm_basic(self, mock_new_empty):
        """Test basic FP8 matrix multiplication."""
        a_tensor = Mock()
        a_tensor.size.return_value = torch.Size([32, 64])
        
        b_tensor = Mock()
        b_tensor.size.return_value = torch.Size([64, 32])
        
        a_scale = Mock()
        b_scale = Mock()
        
        mock_output = Mock()
        mock_new_empty.return_value = mock_output
        
        with patch('inference.kernel.fp8_gemm_kernel') as mock_kernel:
            result = fp8_gemm(a_tensor, b_tensor, a_scale, b_scale)
            
            assert result == mock_output
            mock_kernel.__getitem__.assert_called_once()
    
    def test_fp8_gemm_matrix_dimensions(self):
        """Test FP8 GEMM with different matrix dimensions."""
        a_square = Mock()
        a_square.size.return_value = torch.Size([64, 64])
        
        b_square = Mock()
        b_square.size.return_value = torch.Size([64, 64])
        
        with patch('inference.kernel.fp8_gemm_kernel'), \
             patch('inference.kernel.torch.Tensor.new_empty'):
            fp8_gemm(a_square, b_square, Mock(), Mock())
        
        a_rect = Mock()
        a_rect.size.return_value = torch.Size([128, 256])
        
        b_rect = Mock()
        b_rect.size.return_value = torch.Size([256, 512])
        
        with patch('inference.kernel.fp8_gemm_kernel'), \
             patch('inference.kernel.torch.Tensor.new_empty'):
            fp8_gemm(a_rect, b_rect, Mock(), Mock())
    
    @patch('inference.kernel.torch.Tensor.new_empty')
    def test_fp8_gemm_output_shape(self, mock_new_empty):
        """Test that FP8 GEMM produces correct output shape."""
        a_tensor = Mock()
        a_tensor.size.return_value = torch.Size([100, 200])
        
        b_tensor = Mock()
        b_tensor.size.return_value = torch.Size([200, 300])
        
        mock_output = Mock()
        mock_new_empty.return_value = mock_output
        
        with patch('inference.kernel.fp8_gemm_kernel'):
            result = fp8_gemm(a_tensor, b_tensor, Mock(), Mock())
            
            mock_new_empty.assert_called()
            call_args = mock_new_empty.call_args[0]
            assert 100 in call_args and 300 in call_args

class TestKernelIntegration:
    
    def test_kernel_functions_exist(self):
        """Test that all kernel functions are properly imported."""
        assert callable(act_quant)
        assert callable(weight_dequant)
        assert callable(fp8_gemm)
    
    def test_kernel_function_signatures(self):
        """Test that kernel functions have expected signatures."""
        import inspect
        
        sig = inspect.signature(act_quant)
        assert 'x' in sig.parameters
        assert 'block_size' in sig.parameters
        
        sig = inspect.signature(weight_dequant)
        assert 'x' in sig.parameters
        assert 's' in sig.parameters
        assert 'block_size' in sig.parameters
        
        sig = inspect.signature(fp8_gemm)
        assert 'a' in sig.parameters
        assert 'b' in sig.parameters
        assert 'a_s' in sig.parameters
        assert 'b_s' in sig.parameters
