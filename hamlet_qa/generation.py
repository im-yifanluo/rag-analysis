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
        self.sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_new_tokens,
        )

    def format_prompt(self, system_prompt: str, user_prompt: str) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        if hasattr(self.tokenizer, "apply_chat_template"):
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        return fallback_chat_prompt(system_prompt, user_prompt)

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        prompt = self.format_prompt(system_prompt, user_prompt)
        output = self.llm.generate([prompt], self.sampling_params)[0]
        return output.outputs[0].text.strip()
