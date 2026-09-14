"""
The language model: LFM2.5-1.2B-Instruct by Liquid AI, from a local folder,
loaded once with transformers (trust_remote_code=False, local_files_only=True)
in bfloat16 on the GPU. Nothing is downloaded here. The weights stay local and
are not redistributed; the LFM Open License v1.0 is kept beside them.

generate() samples one reply to the chat messages with a recorded seed and
returns (text, new_tokens, finished, seconds). `finished` is True when the
model ended its reply itself (the end-of-turn token) rather than being cut at
max_new_tokens.
"""
import time

# CHOSEN sampling settings; the engine passes them per call and records them
MAX_NEW_TOKENS = 60
TEMPERATURE = 0.4            # the model card suggests 0.1; 0.4 still gives a retry a different sample.
                             # At 0.8 a local probe on the dry runs' readouts drifted from the readout far more often
TOP_P = 0.9
TOP_K = 0                    # 0: no top-k cut, so top_p alone limits the choice
REPETITION_PENALTY = 1.05    # the model card's recommended value


def is_cuda_oom(exc):
    """True for a CUDA out-of-memory error (torch's class, or its message on older builds)."""
    return type(exc).__name__ == "OutOfMemoryError" or "CUDA out of memory" in str(exc)


class LFMModel:
    def __init__(self, model_dir, device="cuda", say=print):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.device = device
        t0 = time.time()
        self.tok = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True,
                                                 trust_remote_code=False)
        self.model = AutoModelForCausalLM.from_pretrained(
            str(model_dir), dtype=torch.bfloat16, local_files_only=True,
            trust_remote_code=False).to(device).eval()
        self.end_ids = {int(i) for i in (self.tok.eos_token_id,
                                         self.tok.convert_tokens_to_ids("<|im_end|>")) if i is not None}
        self.load_s = time.time() - t0
        self.name = "LFM2.5-1.2B-Instruct"
        say(f"language model ready: {self.name} in {self.load_s:.1f} s on {device}")

    def generate(self, messages, seed, max_new_tokens=MAX_NEW_TOKENS, temperature=TEMPERATURE,
                 top_p=TOP_P, top_k=TOP_K, repetition_penalty=REPETITION_PENALTY):
        torch = self.torch
        from transformers import set_seed

        enc = self.tok.apply_chat_template(messages, add_generation_prompt=True,
                                           return_tensors="pt", return_dict=True)
        enc = {k: v.to(self.device) for k, v in enc.items()}
        n_in = int(enc["input_ids"].shape[1])
        set_seed(int(seed))
        t0 = time.time()
        with torch.no_grad():
            out = self.model.generate(**enc, max_new_tokens=int(max_new_tokens), do_sample=True,
                                      temperature=float(temperature), top_p=float(top_p),
                                      top_k=int(top_k), repetition_penalty=float(repetition_penalty),
                                      pad_token_id=self.tok.pad_token_id)
        if self.device != "cpu":
            torch.cuda.synchronize()
        secs = time.time() - t0
        new = out[0, n_in:].tolist()
        finished = bool(new) and new[-1] in self.end_ids
        text = self.tok.decode(new, skip_special_tokens=True)
        return text, len(new), finished, secs

    def prompt_tokens(self, messages):
        enc = self.tok.apply_chat_template(messages, add_generation_prompt=True, return_dict=True)
        return len(enc["input_ids"])
