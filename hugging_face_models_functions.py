from transformers import AutoModelForCausalLM, AutoTokenizer
import torch



def hugging_face_models_functions(model_name, input_text, system_instruction=None):
    """
    This is a function to run Hugging Face models. 
    """
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    target_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype).to(device)


    messages = []
    if system_instruction:
        messages.append({"role": "system", "content": system_instruction})
    messages.append({"role": "user", "content": input_text})

    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True
    ).to(device)

    # Extract the actual tensor from the BatchEncoding dictionary
    input_ids = inputs["input_ids"]

    # Now input_ids is a standard tensor, and [0] gets the first sequence
    decoded_text = tokenizer.decode(input_ids[0])
    print(decoded_text)
    
    # Generate tokens from the model
    outputs = target_model.generate(
        **inputs, 
        max_new_tokens=256,
        do_sample=True,
        temperature=0.7
    )

    # Extract ONLY the newly generated tokens (skipping the prompt)
    prompt_length = inputs["input_ids"].shape[-1]
    new_tokens = outputs[0][prompt_length:]

    # Decode the response to text
    response = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return response
    
if __name__ == "__main__":
    # Example usage
    model_name = "HuggingFaceTB/SmolLM2-360M-Instruct"
    input_text = "What is the capital of France?"
    hugging_face_models_functions(model_name, input_text)