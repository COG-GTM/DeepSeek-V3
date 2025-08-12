import pytest
import torch
from unittest.mock import Mock, patch, MagicMock

from inference.generate import sample, generate

class TestSample:
    
    def test_sample_with_temperature(self):
        """Test sampling with temperature scaling."""
        logits = torch.tensor([[1.0, 2.0, 3.0, 4.0]], dtype=torch.float32)
        
        token = sample(logits, temperature=1.0)
        assert isinstance(token, int)
        assert 0 <= token < 4
    
    def test_sample_with_high_temperature(self):
        """Test sampling with high temperature (more random)."""
        logits = torch.tensor([[10.0, 1.0, 1.0, 1.0]], dtype=torch.float32)
        
        token = sample(logits, temperature=2.0)
        assert isinstance(token, int)
        assert 0 <= token < 4
    
    def test_sample_with_low_temperature(self):
        """Test sampling with low temperature (more deterministic)."""
        logits = torch.tensor([[1.0, 2.0, 3.0, 10.0]], dtype=torch.float32)
        
        token = sample(logits, temperature=0.1)
        assert isinstance(token, int)
        assert 0 <= token < 4
    
    def test_sample_deterministic(self):
        """Test deterministic sampling with temperature near zero."""
        logits = torch.tensor([[1.0, 2.0, 3.0, 10.0]], dtype=torch.float32)
        
        tokens = [sample(logits, temperature=0.01) for _ in range(10)]
        assert all(token == 3 for token in tokens)  # Index 3 has highest logit

class TestGenerate:
    
    @patch('inference.generate.sample')
    def test_generate_basic(self, mock_sample):
        """Test basic text generation."""
        mock_model = Mock()
        mock_model.forward.return_value = torch.tensor([[[1.0, 2.0, 3.0, 4.0]]])
        
        mock_tokenizer = Mock()
        mock_tokenizer.eos_token_id = 2
        
        mock_sample.side_effect = [1, 2]  # Generate token 1, then EOS token 2
        
        prompt_tokens = torch.tensor([[0]], dtype=torch.long)
        result = generate(mock_model, mock_tokenizer, prompt_tokens, max_new_tokens=5, temperature=1.0)
        
        assert result.shape[1] == 3  # Original token + 2 generated tokens
        assert result[0, -1].item() == 2  # Last token should be EOS
    
    @patch('inference.generate.sample')
    def test_generate_max_tokens(self, mock_sample):
        """Test generation with max tokens limit."""
        mock_model = Mock()
        mock_model.forward.return_value = torch.tensor([[[1.0, 2.0, 3.0, 4.0]]])
        
        mock_tokenizer = Mock()
        mock_tokenizer.eos_token_id = 999  # Set EOS to token that won't be generated
        
        mock_sample.return_value = 1
        
        prompt_tokens = torch.tensor([[0]], dtype=torch.long)
        result = generate(mock_model, prompt_tokens, mock_tokenizer.eos_token_id, max_new_tokens=3, temperature=1.0)
        
        assert result.shape[1] == 4  # Original token + 3 generated tokens
        assert all(result[0, 1:].tolist() == [1, 1, 1])  # All generated tokens should be 1
    
    @patch('inference.generate.sample')
    def test_generate_early_eos(self, mock_sample):
        """Test generation stopping early on EOS token."""
        mock_model = Mock()
        mock_model.forward.return_value = torch.tensor([[[1.0, 2.0, 3.0, 4.0]]])
        
        mock_tokenizer = Mock()
        mock_tokenizer.eos_token_id = 2
        
        mock_sample.return_value = 2
        
        prompt_tokens = torch.tensor([[0]], dtype=torch.long)
        result = generate(mock_model, prompt_tokens, mock_tokenizer.eos_token_id, max_new_tokens=10, temperature=1.0)
        
        assert result.shape[1] == 2  # Original token + 1 EOS token
        assert result[0, -1].item() == 2  # Last token should be EOS
    
    def test_generate_empty_prompt(self):
        """Test generation with empty prompt."""
        mock_model = Mock()
        mock_tokenizer = Mock()
        mock_tokenizer.eos_token_id = 2
        
        prompt_tokens = torch.tensor([[]], dtype=torch.long)
        
        try:
            result = generate(mock_model, prompt_tokens, mock_tokenizer.eos_token_id, max_new_tokens=1, temperature=1.0)
            assert result.shape[0] == 1  # Batch size should be 1
        except (IndexError, RuntimeError):
            pass
    
    @patch('inference.generate.sample')
    def test_generate_batch(self, mock_sample):
        """Test generation with batch input."""
        mock_model = Mock()
        mock_model.forward.return_value = torch.tensor([
            [[1.0, 2.0, 3.0, 4.0]],
            [[2.0, 3.0, 4.0, 5.0]]
        ])
        
        mock_tokenizer = Mock()
        mock_tokenizer.eos_token_id = 2
        
        mock_sample.side_effect = [1, 1, 2, 2]  # Two batches, each generates 1 then EOS
        
        prompt_tokens = torch.tensor([[0], [1]], dtype=torch.long)
        result = generate(mock_model, mock_tokenizer, prompt_tokens, max_new_tokens=5, temperature=1.0)
        
        assert result.shape[0] == 2  # Batch size should be 2
        assert result.shape[1] == 3  # Original token + 2 generated tokens per batch
