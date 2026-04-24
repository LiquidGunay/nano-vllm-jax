"""LLM Engine for Qwen 3.5 JAX with continuous batching."""

import atexit
from time import perf_counter
from typing import List, Dict, Optional, Union
from tqdm.auto import tqdm

from nanovllm_jax.config import Qwen3_5Config
from nanovllm_jax.engine.sequence import Sequence, SamplingParams
from nanovllm_jax.engine.scheduler import Scheduler
from nanovllm_jax.engine.model_runner import ModelRunner
from nanovllm_jax.model import init_params, ModelParams
from nanovllm_jax.load_weights import load_weights_from_hf
import jax.random

try:
    from transformers import AutoTokenizer
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False


class LLMEngine:
    """LLM Engine for Qwen 3.5 JAX.
    
    Features:
    - Continuous batching (prefill + decode in same batch)
    - Paged KV cache with prefix caching
    - Simple Python-based scheduling (like nano-vllm)
    
    Usage:
        engine = LLMEngine("qwen-3.5-0.8b")
        outputs = engine.generate(
            prompts=["Hello, how are you?"],
            sampling_params=SamplingParams(max_tokens=100),
        )
    """

    def __init__(
        self, 
        model_path: str,
        **kwargs,
    ):
        """Initialize LLM engine.
        
        Args:
            model_path: Path to model weights or model identifier
            **kwargs: Additional config parameters
        """
        # Create config
        config_fields = {f.name for f in Qwen3_5Config.__dataclass_fields__.values()}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        self.config = Qwen3_5Config(**config_kwargs)
        
        # Initialize tokenizer
        if not HAS_TRANSFORMERS:
            raise ImportError(
                "transformers library is required. Install with: pip install transformers"
            )
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        
        # Set EOS token from tokenizer
        if self.config.eos is None:
            self.config.eos = self.tokenizer.eos_token_id
        
        # Initialize model parameters - try to load from HF
        try:
            print(f"Loading pretrained weights from {model_path}...")
            self.params = load_weights_from_hf(model_path, self.config)
            print("✓ Using pretrained weights")
        except Exception as e:
            print(f"Warning: Could not load pretrained weights ({e})")
            print("  Falling back to random initialization")
            key = jax.random.PRNGKey(42)
            self.params = init_params(key, self.config)
        
        # Initialize components
        self.scheduler = Scheduler(self.config)
        self.model_runner = ModelRunner(self.config, self.params)
        
        # Register cleanup
        atexit.register(self.exit)

    def exit(self):
        """Cleanup on exit."""
        del self.model_runner

    def add_request(
        self, 
        prompt: Union[str, List[int]], 
        sampling_params: SamplingParams,
    ):
        """Add a generation request.
        
        Args:
            prompt: Prompt text or token IDs
            sampling_params: Sampling parameters
        """
        # Tokenize if string
        if isinstance(prompt, str):
            # TODO: Use proper tokenizer
            # For now, simple placeholder
            prompt = self._tokenize(prompt)
        
        # Create sequence
        seq = Sequence(prompt, sampling_params)
        self.scheduler.add(seq)

    def step(self) -> tuple[List[tuple], int]:
        """Execute one scheduling step.
        
        Returns:
            Tuple of (outputs, num_tokens)
            - outputs: List of (seq_id, completion_tokens) for finished sequences
            - num_tokens: Number of tokens processed (positive for prefill, negative for decode)
        """
        # Schedule sequences
        seqs, is_prefill = self.scheduler.schedule()
        
        # Run model
        token_ids = self.model_runner.run(seqs, is_prefill)
        
        # Post-process
        self.scheduler.postprocess(seqs, token_ids)
        
        # Collect outputs
        outputs = [
            (seq.seq_id, seq.completion_token_ids) 
            for seq in seqs 
            if seq.is_finished
        ]
        
        # Track throughput
        if is_prefill:
            num_tokens = sum(len(seq) for seq in seqs)
        else:
            num_tokens = -len(seqs)  # Negative for decode
        
        return outputs, num_tokens

    def is_finished(self) -> bool:
        """Check if all requests are complete."""
        return self.scheduler.is_finished()

    def generate(
        self,
        prompts: List[Union[str, List[int]]],
        sampling_params: Union[SamplingParams, List[SamplingParams]] = None,
        use_tqdm: bool = True,
    ) -> List[Dict[str, any]]:
        """Generate completions for prompts.
        
        Args:
            prompts: List of prompts (text or token IDs)
            sampling_params: Sampling parameters (single or list)
            use_tqdm: Show progress bar
            
        Returns:
            List of dicts with 'text' and 'token_ids' keys
        """
        if sampling_params is None:
            sampling_params = SamplingParams()
        
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        
        # Setup progress bar
        if use_tqdm:
            pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True)
        
        # Add all requests
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)
        
        # Generation loop
        outputs = {}
        prefill_throughput = decode_throughput = 0.0
        
        while not self.is_finished():
            t = perf_counter()
            output, num_tokens = self.step()
            
            # Update progress
            if use_tqdm:
                if num_tokens > 0:
                    prefill_throughput = num_tokens / (perf_counter() - t)
                else:
                    decode_throughput = -num_tokens / (perf_counter() - t)
                
                pbar.set_postfix({
                    "Prefill": f"{int(prefill_throughput)} tok/s",
                    "Decode": f"{int(decode_throughput)} tok/s",
                })
            
            # Collect finished outputs
            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids
                if use_tqdm:
                    pbar.update(1)
        
        # Sort by seq_id and decode
        outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
        results = [
            {"text": self._detokenize(token_ids), "token_ids": token_ids}
            for token_ids in outputs
        ]
        
        if use_tqdm:
            pbar.close()
        
        return results

    def _tokenize(self, text: str) -> List[int]:
        """Tokenize text using Qwen tokenizer."""
        return self.tokenizer.encode(text, add_special_tokens=False)

    def _detokenize(self, token_ids: List[int]) -> str:
        """Detokenize token IDs using Qwen tokenizer."""
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)
