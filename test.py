from transformers import AutoModelForCausalLM, AutoTokenizer

# Load trained model and tokenizer
model_dir = "./greeting_model"
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForCausalLM.from_pretrained(model_dir)

test_inputs = [
    "Hãy chào chú giúp tôi.",
    "Bạn có thể chào bác không?",
    "Làm ơn chào cô!",
    "Hãy chào cô đi!"
]

for input_text in test_inputs:
    inputs = tokenizer(input_text, return_tensors="pt")
    outputs = model.generate(
        **inputs,
        max_length=50,
        temperature=0.7,  # Increase this value for more diversity
        top_k=50,
        top_p=0.8,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Input: {input_text}\nResponse: {response}\n")
