import pytest
import tempfile
import json
from unittest.mock import Mock, patch, MagicMock

from evaluation.utils.inference_wrapper import (
    NativeInferenceWrapper, 
    SGLangInferenceWrapper, 
    MockInferenceWrapper
)

class TestNativeInferenceWrapper:
    
    @patch('evaluation.utils.inference_wrapper.Transformer')
    @patch('evaluation.utils.inference_wrapper.AutoTokenizer')
    def test_init(self, mock_tokenizer_class, mock_transformer_class):
        """Test NativeInferenceWrapper initialization."""
        mock_tokenizer = Mock()
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        
        mock_model = Mock()
        mock_transformer_class.from_pretrained.return_value = mock_model
        
        wrapper = NativeInferenceWrapper("/fake/model", "/fake/config")
        
        assert wrapper.model_path == "/fake/model"
        assert wrapper.config_path == "/fake/config"
        assert wrapper.tokenizer == mock_tokenizer
        assert wrapper.model == mock_model
    
    @patch('evaluation.utils.inference_wrapper.Transformer')
    @patch('evaluation.utils.inference_wrapper.AutoTokenizer')
    @patch('evaluation.utils.inference_wrapper.generate')
    def test_generate(self, mock_generate, mock_tokenizer_class, mock_transformer_class):
        """Test text generation."""
        mock_tokenizer = Mock()
        mock_tokenizer.encode.return_value = [1, 2, 3]
        mock_tokenizer.decode.return_value = "Generated response"
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        
        mock_model = Mock()
        mock_transformer_class.from_pretrained.return_value = mock_model
        
        mock_generate.return_value = [4, 5, 6]
        
        wrapper = NativeInferenceWrapper("/fake/model", "/fake/config")
        result = wrapper.generate(["Test prompt"])
        
        assert result == "Generated response"
        mock_generate.assert_called_once()
    
    @patch('evaluation.utils.inference_wrapper.Transformer')
    @patch('evaluation.utils.inference_wrapper.AutoTokenizer')
    def test_batch_generate(self, mock_tokenizer_class, mock_transformer_class):
        """Test batch text generation."""
        mock_tokenizer = Mock()
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        
        mock_model = Mock()
        mock_transformer_class.from_pretrained.return_value = mock_model
        
        wrapper = NativeInferenceWrapper("/fake/model", "/fake/config")
        
        wrapper.generate = Mock(side_effect=["Response 1", "Response 2"])
        
        prompts = ["Prompt 1", "Prompt 2"]
        results = wrapper.generate(prompts)
        
        assert results == ["Response 1", "Response 2"]
        assert wrapper.generate.call_count == 2

class TestSGLangInferenceWrapper:
    
    def test_init(self):
        """Test SGLangInferenceWrapper initialization."""
        wrapper = SGLangInferenceWrapper("http://localhost:8000")
        assert hasattr(wrapper, 'base_url')
    
    @patch('requests.post')
    def test_generate(self, mock_post):
        """Test text generation via SGLang API."""
        mock_response = Mock()
        mock_response.json.return_value = {"text": "Generated response"}
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response
        
        wrapper = SGLangInferenceWrapper("http://localhost:8000")
        result = wrapper.generate(["Test prompt"])
        
        assert result == "Generated response"
        mock_post.assert_called_once()
    
    @patch('requests.post')
    def test_batch_generate(self, mock_post):
        """Test batch text generation via SGLang API."""
        mock_response = Mock()
        mock_response.json.return_value = {"texts": ["Response 1", "Response 2"]}
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response
        
        wrapper = SGLangInferenceWrapper("http://localhost:8000")
        prompts = ["Prompt 1", "Prompt 2"]
        results = wrapper.generate(prompts)
        
        assert results == ["Response 1", "Response 2"]
        mock_post.assert_called_once()
    
    @patch('requests.post')
    def test_generate_api_error(self, mock_post):
        """Test handling of API errors."""
        mock_post.side_effect = Exception("API Error")
        
        wrapper = SGLangInferenceWrapper("http://localhost:8000")
        
        with pytest.raises(Exception, match="API Error"):
            wrapper.generate(["Test prompt"])

class TestMockInferenceWrapper:
    
    def test_init_default(self):
        """Test MockInferenceWrapper initialization with defaults."""
        wrapper = MockInferenceWrapper()
        assert wrapper.dataset is None
        assert hasattr(wrapper, 'accuracy')
    
    def test_init_with_dataset(self, mock_gpqa_data, temp_dir):
        """Test MockInferenceWrapper initialization with dataset."""
        dataset_path = temp_dir + "/test_data.jsonl"
        with open(dataset_path, 'w') as f:
            for item in mock_gpqa_data:
                f.write(json.dumps(item) + '\n')
        
        wrapper = MockInferenceWrapper(dataset_path=dataset_path, accuracy=0.9)
        assert wrapper.dataset is not None
        assert hasattr(wrapper, 'accuracy')
        assert len(wrapper.dataset) == 2
    
    def test_load_dataset(self, mock_gpqa_data, temp_dir):
        """Test loading dataset from file."""
        dataset_path = temp_dir + "/test_data.jsonl"
        with open(dataset_path, 'w') as f:
            for item in mock_gpqa_data:
                f.write(json.dumps(item) + '\n')
        
        wrapper = MockInferenceWrapper()
        with patch.object(wrapper, '_load_dataset'):
            wrapper._load_dataset(dataset_path)
        
        assert len(wrapper.dataset) == 2
        assert wrapper.dataset[0]["question"] == "What is the capital of France?"
    
    def test_generate_with_dataset(self, mock_gpqa_data, temp_dir):
        """Test generation with loaded dataset."""
        dataset_path = temp_dir + "/test_data.jsonl"
        with open(dataset_path, 'w') as f:
            for item in mock_gpqa_data:
                f.write(json.dumps(item) + '\n')
        
        wrapper = MockInferenceWrapper(dataset_path=dataset_path, accuracy=1.0)
        
        prompt = "What is the capital of France?"
        result = wrapper.generate([prompt])
        
        assert result == "B"
    
    def test_generate_without_dataset(self):
        """Test generation without loaded dataset."""
        wrapper = MockInferenceWrapper(accuracy=0.5)
        
        result = wrapper.generate(["Random question"])
        
        assert result in ["A", "B", "C", "D"]
    
    def test_batch_generate(self, mock_gpqa_data, temp_dir):
        """Test batch generation."""
        dataset_path = temp_dir + "/test_data.jsonl"
        with open(dataset_path, 'w') as f:
            for item in mock_gpqa_data:
                f.write(json.dumps(item) + '\n')
        
        wrapper = MockInferenceWrapper(dataset_path=dataset_path, accuracy=1.0)
        
        prompts = [
            "What is the capital of France?",
            "What is 2+2?"
        ]
        results = wrapper.generate(prompts)
        
        assert len(results) == 2
        assert results[0] == "B"  # Correct answer for France question
        assert results[1] == "B"  # Correct answer for math question
    
    def test_find_matching_question(self, mock_gpqa_data, temp_dir):
        """Test finding matching question in dataset."""
        dataset_path = temp_dir + "/test_data.jsonl"
        with open(dataset_path, 'w') as f:
            for item in mock_gpqa_data:
                f.write(json.dumps(item) + '\n')
        
        wrapper = MockInferenceWrapper(dataset_path=dataset_path)
        
        with patch.object(wrapper, '_find_matching_question', side_effect=[
            {"answer": "B"}, {"answer": "B"}, None
        ]):
            match = wrapper._find_matching_question("What is the capital of France?")
            assert match is not None
            assert match["answer"] == "B"
            
            match = wrapper._find_matching_question("capital of France")
            assert match is not None
            
            match = wrapper._find_matching_question("Completely different question")
            assert match is None
