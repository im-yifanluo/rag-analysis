"""vLLM reader wrapper used by server-side experiment runs."""

from __future__ import annotations

from hamlet_qa.prompts import fallback_chat_prompt


class VLLMReader:
    """Minimal vLLM wrapper for one local reader model."""

    def __init__(
        self,
        model_name: str,
        temperature: float = 0.0,
        max_new_tokens: int = 512,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.90,
    ):
        from vllm import LLM, SamplingParams

        self.model_name = model_name
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

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        prompt = self.format_prompt(system_prompt, user_prompt)
        output = self.llm.generate([prompt], self.sampling_params)[0]
        return output.outputs[0].text.strip()
