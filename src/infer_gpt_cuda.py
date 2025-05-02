import onnxruntime as ort
from transformers import GPT2Tokenizer
import numpy as np


def load_session(onnx_path: str):
    # Enable all available optimizations in ORT
    sess_opts = ort.SessionOptions()
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    # Register your custom ops library (if still needed on GPU)
    so_path = "/home/mcw/sidharth/ONNX_DEV/ops/cuda/build/libcustom_cuda_ops.so"
    sess_opts.register_custom_ops_library(so_path)

    # Create session with CUDA first, CPU as fallback
    providers = [
        (
            "CUDAExecutionProvider",
            {
                "device_id": 0,
                # you can tune these CUDA_EP options if needed:
                # "arena_extend_strategy": "kNextPowerOfTwo",
                # "cudnn_conv_algo_search": "EXHAUSTIVE",
            },
        ),
        "CPUExecutionProvider",
    ]
    session = ort.InferenceSession(onnx_path, sess_opts, providers)
    print("Active EPs:", session.get_providers())
    return session


def generate_with_onnx(session, tokenizer, prompt: str, max_length: int = 50):
    # Tokenize prompt
    inputs = tokenizer(prompt, return_tensors="np")
    input_ids = inputs["input_ids"].astype(np.int64)

    # Prepare feeds
    feeds = {"input_ids": input_ids}

    # Run the ONNX model
    outputs = session.run(None, feeds)
    logits = outputs[0]  # shape: [batch_size, seq_len, vocab_size]

    # Greedy decoding
    generated = input_ids.tolist()[0]
    for _ in range(max_length - input_ids.shape[1]):
        last_logits = logits[0, -1, :]
        next_token = int(np.argmax(last_logits))
        generated.append(next_token)

        # Prepare next step
        input_ids = np.array([generated], dtype=np.int64)
        feeds = {"input_ids": input_ids}
        outputs = session.run(None, feeds)
        logits = outputs[0]

    return tokenizer.decode(generated, skip_special_tokens=True)


if __name__ == "__main__":
    onnx_model = "/home/mcw/sidharth/ONNX_DEV/models/onnx_models/gpt2_124M.onnx"
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    session = load_session(onnx_model)

    while True:
        prompt = input("Enter a prompt (or 'exit' to quit): ")
        if prompt.strip().lower() == "exit":
            break
        result = generate_with_onnx(session, tokenizer, prompt, max_length=100)
        print("Generated:", result)
