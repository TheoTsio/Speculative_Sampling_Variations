from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os
import torch
from huggingface_hub import login

def reward_function(prompt, answer, reward_model_name):
    """
    Computes the reward based on the predictions and targets.

    Args:
        predictions (list): A list of predicted values.
        targets (list): A list of target values.
    """
    
    tokenizer = AutoTokenizer.from_pretrained(reward_model_name)
    model = AutoModelForSequenceClassification.from_pretrained(reward_model_name)    
        
    
    conversation = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": answer}
    ]
    
    inputs = tokenizer.apply_chat_template(
        conversation, 
        tokenize=True, 
        return_tensors="pt",
        return_dict=True
    ).to(model.device)
    
    # 5. Run inference
    model.eval()
    with torch.no_grad():
        # Notice the ** to unpack the dictionary into input_ids and attention_mask
        score = model(**inputs).logits[0][0].item()
        
    return score


if __name__ == "__main__":
    # Log in to Hugging Face Hub
    login(token=os.environ.get("HF_TOKEN"))
    reward_model_name = "Skywork/Skywork-Reward-Llama-3.1-8B"
    
    # Example usage
    prompt = "What is the capital of France?"
    answer = "The capital of France is Paris."
    
    score = reward_function(prompt, answer, reward_model_name)
    print(f"Reward score: {score}")
    

