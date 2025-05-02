import onnxruntime as ort
from transformers import GPT2Tokenizer
import numpy as np


def load_session(onnx_path: str):
    # Enable all available optimizations in ORT
    sess_opts = ort.SessionOptions()

    # Custom ops library path
    so_path = "/home/mcw/sidharth/ONNX_DEV/ops/cpu/build/libcustom_op.so"
    sess_opts.register_custom_ops_library(so_path)
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(onnx_path, sess_opts)


def generate_with_onnx(session, tokenizer, prompt: str, max_length: int = 50):
    # Tokenize prompt
    inputs = tokenizer(prompt, return_tensors="np")
    input_ids = inputs["input_ids"].astype(np.int64)

    # Prepare feeds
    feeds = {"input_ids": input_ids}

    # Run the ONNX model
    outputs = session.run(None, feeds)
    logits = outputs[0]  # shape: [batch_size, seq_len, vocab_size]

    # Greedy decoding: take argmax at each step
    generated = input_ids.tolist()[0]
    for _ in range(max_length - input_ids.shape[1]):
        last_token_logits = logits[0, -1, :]
        next_token = int(np.argmax(last_token_logits))
        generated.append(next_token)

        # Update input for next iteration
        input_ids = np.array([generated], dtype=np.int64)
        feeds = {"input_ids": input_ids}
        outputs = session.run(None, feeds)
        logits = outputs[0]

    return tokenizer.decode(generated, skip_special_tokens=True)


if __name__ == "__main__":
    onnx_model = "/home/mcw/sidharth/ONNX_DEV/models/onnx_models/gpt2_124M_custom.onnx"
    prompt = "Once upon a time"
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    session = load_session(onnx_model)

    while True:

        prompt = input("Enter a prompt (or 'exit' to quit): ")
        if prompt.lower() == "exit":
            break

        result = generate_with_onnx(session, tokenizer, prompt, max_length=100)
        print("Generated:", result)
