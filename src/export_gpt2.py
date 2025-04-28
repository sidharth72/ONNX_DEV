## Script: export_gpt2_to_onnx.py
"""
Converts Hugging Face GPT-2 (PyTorch) to ONNX with dynamic axes.
Usage: python export_gpt2_to_onnx.py --model gpt2 --output gpt2.onnx
"""
import argparse
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

def export_to_onnx(model_name: str, onnx_path: str, opset: int = 13):
    # Load model and tokenizer
    model = GPT2LMHeadModel.from_pretrained(model_name).eval()
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)

    # Prepare dummy input
    dummy_text = "Hello, ONNX world!"
    inputs = tokenizer(dummy_text, return_tensors="pt")
    input_ids = inputs["input_ids"]

    # Define dynamic axes
    # Dynamic axes that can change during inference
    # For example, batch size and sequence length can vary
    # in real-world scenarios.
    dynamic_axes = {
        "input_ids": {0: "batch_size", 1: "sequence_length"},
        "attention_mask": {0: "batch_size", 1: "sequence_length"},
        "output": {0: "batch_size", 1: "sequence_length"}
    }

    # Export
    torch.onnx.export(
        model,
        (input_ids, ),
        onnx_path,
        input_names=["input_ids"],
        output_names=["output"],
        dynamic_axes=dynamic_axes,
        opset_version=opset,
        do_constant_folding=True, # Precomputing constant values to optimize the model inference speed
    )
    print(f"ONNX model saved to {onnx_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gpt2", help="HuggingFace model name")
    parser.add_argument("--output", type=str, required=True, help="Path to save ONNX model")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version")
    args = parser.parse_args()
    export_to_onnx(args.model, args.output, args.opset)