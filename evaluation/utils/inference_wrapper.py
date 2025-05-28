"""
Inference wrapper for DeepSeek-V3 model.

This module provides a wrapper around different inference frameworks
for the DeepSeek-V3 model, including native inference and SGLang.
"""

import os
import sys
import json
import logging
import importlib.util
from typing import Dict, List, Optional, Tuple, Union, Any

import torch
import transformers
from transformers import AutoTokenizer

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

logger = logging.getLogger(__name__)

class InferenceWrapper:
    """
    Base class for inference wrappers.
    
    This class defines the interface for inference wrappers
    that can be used with different inference frameworks.
    """
    
    def __init__(self):
        """Initialize the inference wrapper."""
        pass
    
    def generate(
        self,
        prompts: List[str],
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        top_p: float = 1.0,
        **kwargs,
    ) -> List[str]:
        """
        Generate responses for the given prompts.
        
        Args:
            prompts: List of input prompts.
            max_new_tokens: Maximum number of new tokens to generate.
            temperature: Sampling temperature.
            top_p: Top-p sampling parameter.
            **kwargs: Additional keyword arguments.
            
        Returns:
            List of generated responses.
        """
        raise NotImplementedError("Subclasses must implement generate()")


class NativeInferenceWrapper(InferenceWrapper):
    """
    Wrapper for native DeepSeek-V3 inference.
    
    This class provides a wrapper around the native DeepSeek-V3
    inference code from inference/generate.py.
    """
    
    def __init__(
        self,
        model_path: str,
        config_path: Optional[str] = None,
        dtype: str = "fp8",
        max_batch_size: int = 16,
        max_seq_len: int = 4096,
        device: str = "cuda",
    ):
        """
        Initialize the native inference wrapper.
        
        Args:
            model_path: Path to the model weights.
            config_path: Path to the model configuration file.
            dtype: Data type for inference ("fp8" or "bf16").
            max_batch_size: Maximum batch size for inference.
            max_seq_len: Maximum sequence length.
            device: Device to run inference on.
        """
        super().__init__()
        
        try:
            from inference.model import ModelArgs, Transformer
            from inference.generate import sample
            self.ModelArgs = ModelArgs
            self.Transformer = Transformer
            self.sample = sample
        except ImportError as e:
            logger.error(f"Failed to import from inference directory: {e}")
            raise
        
        self.model_path = model_path
        self.config_path = config_path
        self.dtype = dtype
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len
        self.device = device
        
        if config_path is not None:
            with open(config_path, "r") as f:
                config = json.load(f)
            
            args = self.ModelArgs()
            for key, value in config.items():
                if hasattr(args, key):
                    setattr(args, key, value)
        else:
            args = self.ModelArgs()
        
        args.max_batch_size = max_batch_size
        args.max_seq_len = max_seq_len
        args.dtype = dtype
        
        if dtype == "fp8":
            torch.set_default_dtype(torch.float8_e4m3fn)
        else:
            torch.set_default_dtype(torch.bfloat16)
        
        logger.info(f"Loading DeepSeek-V3 model from {model_path}")
        self.model = self.Transformer(args)
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        self.model.to(device)
        
        logger.info("Model loaded successfully")
    
    def generate(
        self,
        prompts: List[str],
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        top_p: float = 1.0,
        **kwargs,
    ) -> List[str]:
        """
        Generate responses for the given prompts using native inference.
        
        Args:
            prompts: List of input prompts.
            max_new_tokens: Maximum number of new tokens to generate.
            temperature: Sampling temperature.
            top_p: Top-p sampling parameter.
            **kwargs: Additional keyword arguments.
            
        Returns:
            List of generated responses.
        """
        responses = []
        
        for i in range(0, len(prompts), self.max_batch_size):
            batch_prompts = prompts[i:i + self.max_batch_size]
            batch_responses = self._generate_batch(
                batch_prompts,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                **kwargs,
            )
            responses.extend(batch_responses)
        
        return responses
    
    def _generate_batch(
        self,
        prompts: List[str],
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        top_p: float = 1.0,
        **kwargs,
    ) -> List[str]:
        """
        Generate responses for a batch of prompts.
        
        Args:
            prompts: Batch of input prompts.
            max_new_tokens: Maximum number of new tokens to generate.
            temperature: Sampling temperature.
            top_p: Top-p sampling parameter.
            **kwargs: Additional keyword arguments.
            
        Returns:
            List of generated responses.
        """
        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True)
        input_ids = inputs.input_ids.to(self.device)
        
        with torch.inference_mode():
            generated_ids = []
            
            for i in range(input_ids.shape[0]):
                tokens = input_ids[i:i+1]
                
                for _ in range(max_new_tokens):
                    logits = self.model(tokens)
                    
                    next_token = self.sample(
                        logits,
                        temperature=temperature,
                        top_p=top_p,
                    )
                    
                    tokens = torch.cat([tokens, next_token.unsqueeze(0)], dim=1)
                    
                    if next_token.item() == self.tokenizer.eos_token_id:
                        break
                
                generated_ids.append(tokens[0])
        
        responses = []
        for ids in generated_ids:
            new_tokens = ids[input_ids[0].shape[0]:]
            response = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
            responses.append(response)
        
        return responses


class SGLangInferenceWrapper(InferenceWrapper):
    """
    Wrapper for SGLang inference.
    
    This class provides a wrapper around the SGLang inference
    framework for DeepSeek-V3.
    """
    
    def __init__(
        self,
        model_path: str,
        dtype: str = "fp8",
        max_batch_size: int = 16,
        max_seq_len: int = 4096,
        device: str = "cuda",
    ):
        """
        Initialize the SGLang inference wrapper.
        
        Args:
            model_path: Path to the model weights.
            dtype: Data type for inference ("fp8" or "bf16").
            max_batch_size: Maximum batch size for inference.
            max_seq_len: Maximum sequence length.
            device: Device to run inference on.
        """
        super().__init__()
        
        if importlib.util.find_spec("sglang") is None:
            logger.error("SGLang is not installed. Please install it with: pip install sglang")
            raise ImportError("SGLang is not installed")
        
        import sglang as sgl
        
        self.model_path = model_path
        self.dtype = dtype
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len
        self.device = device
        
        logger.info(f"Setting up SGLang runtime for DeepSeek-V3 from {model_path}")
        
        precision = "fp8" if dtype == "fp8" else "bf16"
        
        self.runtime = sgl.Runtime(
            model=model_path,
            model_type="deepseek-v3",
            precision=precision,
            max_batch_size=max_batch_size,
            max_seq_len=max_seq_len,
            device=device,
        )
        
        logger.info("SGLang runtime initialized successfully")
    
    def generate(
        self,
        prompts: List[str],
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        top_p: float = 1.0,
        **kwargs,
    ) -> List[str]:
        """
        Generate responses for the given prompts using SGLang.
        
        Args:
            prompts: List of input prompts.
            max_new_tokens: Maximum number of new tokens to generate.
            temperature: Sampling temperature.
            top_p: Top-p sampling parameter.
            **kwargs: Additional keyword arguments.
            
        Returns:
            List of generated responses.
        """
        import sglang as sgl
        
        @sgl.function
        def generate(s, prompt):
            s += sgl.user(prompt)
            s += sgl.assistant(
                max_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
            )
            return s.assistant
        
        responses = []
        
        for i in range(0, len(prompts), self.max_batch_size):
            batch_prompts = prompts[i:i + self.max_batch_size]
            
            states = [sgl.State() for _ in range(len(batch_prompts))]
            
            results = self.runtime.run_batch([
                generate(state, prompt)
                for state, prompt in zip(states, batch_prompts)
            ])
            
            batch_responses = [result.output for result in results]
            responses.extend(batch_responses)
        
        return responses


def create_inference_wrapper(
    framework: str,
    model_path: str,
    config_path: Optional[str] = None,
    dtype: str = "fp8",
    max_batch_size: int = 16,
    max_seq_len: int = 4096,
    device: str = "cuda",
) -> InferenceWrapper:
    """
    Create an inference wrapper for the specified framework.
    
    Args:
        framework: Inference framework to use ("native" or "sglang").
        model_path: Path to the model weights.
        config_path: Path to the model configuration file.
        dtype: Data type for inference ("fp8" or "bf16").
        max_batch_size: Maximum batch size for inference.
        max_seq_len: Maximum sequence length.
        device: Device to run inference on.
        
    Returns:
        Initialized inference wrapper.
    """
    if framework == "native":
        return NativeInferenceWrapper(
            model_path=model_path,
            config_path=config_path,
            dtype=dtype,
            max_batch_size=max_batch_size,
            max_seq_len=max_seq_len,
            device=device,
        )
    elif framework == "sglang":
        return SGLangInferenceWrapper(
            model_path=model_path,
            dtype=dtype,
            max_batch_size=max_batch_size,
            max_seq_len=max_seq_len,
            device=device,
        )
    else:
        raise ValueError(f"Unsupported inference framework: {framework}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    model_path = "/path/to/deepseek-v3-model"
    config_path = "/path/to/config.json"
    
    wrapper = create_inference_wrapper(
        framework="native",
        model_path=model_path,
        config_path=config_path,
    )
    
    prompts = [
        "What is the capital of France?",
        "Explain quantum computing in simple terms.",
    ]
    
    responses = wrapper.generate(prompts)
    
    for prompt, response in zip(prompts, responses):
        print(f"Prompt: {prompt}")
        print(f"Response: {response}")
        print("-" * 80)
