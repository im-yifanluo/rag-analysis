"""vLLM reader wrapper used by server-side experiment runs."""

from __future__ import annotations

import os

from hamlet_qa.core.prompts import fallback_chat_prompt


class VLLMReader:
    """Minimal vLLM wrapper for one local reader model."""

    def __init__(
        self,
        model_name: str,
        temperature: float = 0.0,
        max_new_tokens: int = 512,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.90,
        device: str = "cuda",
    ):
        if device.startswith("cuda:"):
            os.environ["CUDA_VISIBLE_DEVICES"] = device.split(":", 1)[1]
        elif device == "cpu":
            os.environ["CUDA_VISIBLE_DEVICES"] = ""

        # vLLM v1 launches its EngineCore in a child process. If CUDA is already
        # initialized in the parent (vLLM probes it on import), a forked child
        # raises "Cannot re-initialize CUDA in forked subprocess". Force spawn so
        # the engine core starts with a clean CUDA context. setdefault lets an
        # explicit env override still win.
        os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

        from vllm import LLM, SamplingParams

        self.model_name = model_name
        self.device = device
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.llm = LLM(
            model=model_name,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            trust_remote_code=True,
        )
        self.tokenizer = self.llm.get_tokenizer()
        self.model_max_context = self._resolve_model_max_context()
        self.sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_new_tokens,
        )

    def _resolve_model_max_context(self) -> int | None:
        value = getattr(self.tokenizer, "model_max_length", None)
        if isinstance(value, int) and 0 < value < 10_000_000:
            return value
        llm_engine = getattr(self.llm, "llm_engine", None)
        model_config = getattr(llm_engine, "model_config", None)
        max_model_len = getattr(model_config, "max_model_len", None)
        if isinstance(max_model_len, int) and max_model_len > 0:
            return max_model_len
        return None

    def format_prompt(self, system_prompt: str, user_prompt: str) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        if hasattr(self.tokenizer, "apply_chat_template"):
            kwargs = {
                "tokenize": False,
                "add_generation_prompt": True,
            }
            if "qwen3" in self.model_name.lower():
                kwargs["enable_thinking"] = False
            try:
                return self.tokenizer.apply_chat_template(messages, **kwargs)
            except TypeError:
                kwargs.pop("enable_thinking", None)
                return self.tokenizer.apply_chat_template(messages, **kwargs)
        return fallback_chat_prompt(system_prompt, user_prompt)

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int | None = None,
    ) -> str:
        prompt = self.format_prompt(system_prompt, user_prompt)
        sampling_params = self.sampling_params
        if max_tokens is not None and max_tokens != self.max_new_tokens:
            from vllm import SamplingParams

            sampling_params = SamplingParams(
                temperature=self.temperature,
                max_tokens=max_tokens,
            )
        output = self.llm.generate([prompt], sampling_params)[0]
        return output.outputs[0].text.strip()

    def score_completion(self, full_prompt: str, completion: str) -> dict:
        """Log-likelihood of `completion` given `full_prompt` via prompt_logprobs.

        Used by the oracle CI value metric: utility is the mean token-level
        log-probability of the gold answer under the reader.
        """
        from vllm import SamplingParams

        prefix_tokens = len(self.tokenizer.encode(full_prompt))
        scored_text = full_prompt + completion
        total_tokens = len(self.tokenizer.encode(scored_text))
        completion_tokens = max(1, total_tokens - prefix_tokens)

        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=1,
            prompt_logprobs=0,
        )
        output = self.llm.generate([scored_text], sampling_params)[0]
        prompt_logprobs = output.prompt_logprobs or []
        tail = prompt_logprobs[-completion_tokens:]
        sum_logprob = 0.0
        counted = 0
        for entry in tail:
            if not entry:
                continue
            logprob_obj = next(iter(entry.values()))
            logprob = getattr(logprob_obj, "logprob", None)
            sum_logprob += float(logprob if logprob is not None else logprob_obj)
            counted += 1
        mean_logprob = sum_logprob / counted if counted else 0.0
        return {
            "sum_logprob": sum_logprob,
            "num_tokens": counted,
            "mean_logprob": mean_logprob,
        }
