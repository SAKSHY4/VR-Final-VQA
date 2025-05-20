#!/usr/bin/env python3
import os
import argparse
import pandas as pd
from PIL import Image
from tqdm import tqdm
import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration, BitsAndBytesConfig
from peft import PeftModel

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=str, required=True, help='Path to image folder')
    parser.add_argument('--csv_path', type=str, required=True, help='Path to image-metadata CSV')
    args = parser.parse_args()

    # Define paths - use absolute path for reliable loading
    adapter_path = "./adapter/checkpoint-3937"  # Or the specific checkpoint you want to use
    base_model = "Intel/llava-gemma-2b"

    # Load metadata CSV
    print(f"Reading metadata from {args.csv_path}")
    df = pd.read_csv(args.csv_path)
    print(f"Found {len(df)} questions")

    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Configure 4-bit quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
        quantize_kv_cache=True
    )

    # Load processor and model
    print("Loading processor...")
    processor = AutoProcessor.from_pretrained(
        base_model,
        trust_remote_code=True,
        use_fast=True
    )

    # Load base model with quantization
    print(f"Loading base model from {base_model}...")
    base_model = LlavaForConditionalGeneration.from_pretrained(
        base_model,
        trust_remote_code=True,
        quantization_config=bnb_config,
        torch_dtype=torch.float16,
        device_map="auto",
        use_flash_attention_2=True
    )
    base_model.config.use_cache = True

    # Load adapter weights
    print(f"Loading adapter from {adapter_path}")
    model = PeftModel.from_pretrained(
        base_model,
        adapter_path,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    model.eval()
    
    # CRITICAL: Configure processor correctly
    processor.patch_size = model.config.vision_config.patch_size
    processor.vision_feature_select_strategy = getattr(model.config, "vision_feature_select_strategy", "default")
    processor.vision_feature_layer = getattr(model.config, "vision_feature_layer", -1)
    processor.num_additional_image_tokens = 1
    
    print("✅ Model loaded successfully")

    # Process each sample
    generated_answers = []
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        image_path = os.path.join(args.image_dir, row['image_name'])
        question = str(row['question'])
        true_answer = str(row['answer'])
        
        try:
            # Load image
            image = Image.open(image_path).convert("RGB")
            
            # Format prompt with chat template
            messages = [
                {"role": "user", "content": f"<image>\n{question}\nAnswer with just one word."}
            ]
            prompt = processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            
            # Process inputs
            inputs = processor(text=prompt, images=image, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Generate answer
            with torch.no_grad():
                out_ids = model.generate(
                    **inputs,
                    max_new_tokens=10,
                    do_sample=False,
                    repetition_penalty=1.5
                )
            
            # Extract only the newly generated tokens
            input_length = inputs["input_ids"].shape[1]
            generated_ids = out_ids[0, input_length:]
            
            # Decode just the new tokens
            raw_answer = processor.tokenizer.decode(
                generated_ids,
                skip_special_tokens=True
            )
            
            # Extract first word as final answer
            answer_text = raw_answer.strip().lower()
            final_answer = answer_text.split()[0] if answer_text else ""
            
        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")
            final_answer = "error"
        
        generated_answers.append(final_answer)

    # Add predictions to dataframe and save
    df["generated_answer"] = generated_answers
    df.to_csv("results.csv", index=False)
    print("Results saved to results.csv")

if __name__ == "__main__":
    main()