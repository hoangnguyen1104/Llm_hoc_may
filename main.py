from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import Dataset

# Step 1: Prepare Data for Training
data = {
    "input_text": [
        "Hãy chào cô đi!",
        "Xin chào, bạn là ai?",
        "Chào ông nhé!",
        "Làm ơn chào chú!",
        "Chào bác!",
        "Bạn có thể nói lời chào giúp tôi không?",
        "Chào cả nhà!"
    ],
    "output_text": [
        "Chào cô!",
        "Xin chào bạn!",
        "Chào ông!",
        "Chào chú!",
        "Chào bác!",
        "Chào mọi người!",
        "Chào các bạn!"
    ]
}

dataset = Dataset.from_dict(data)

# Step 2: Define Tokenizer and Model
tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")

# Set pad_token
tokenizer.pad_token = tokenizer.eos_token

# Tokenize Data
def tokenize_function(example):
    # # Ensure input_text and output_text are strings (if they are lists, convert to strings)
    # input_text = " ".join(example["input_text"]) if isinstance(example["input_text"], list) else example["input_text"]
    # output_text = " ".join(example["output_text"]) if isinstance(example["output_text"], list) else example["output_text"]
    #
    # # Combine input and output into one string (for causal LM)
    # text = input_text + " " + output_text
    # return tokenizer(
    #     text,
    #     padding="longest",  # Use longest for variable-length sequences
    #     truncation=True,
    #     max_length=50
    # )
    return tokenizer(
        example["input_text"],
        text_target=example["output_text"],
        padding="longest",
        truncation=True,
        max_length=50
    )

tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Step 3: Training Arguments
training_args = TrainingArguments(
    output_dir="./greeting_model",
    evaluation_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=2,
    num_train_epochs=3,  # Start with fewer epochs for quick testing
    save_strategy="epoch",
    logging_dir="./logs",
    logging_steps=500,  # Log every 500 steps
    save_steps=1000,  # Save model every 1000 steps
    report_to="tensorboard"  # Optional: Use TensorBoard for visualization
)

# Split dataset into training and evaluation datasets
train_test_split = tokenized_dataset.train_test_split(test_size=0.2)
train_dataset = train_test_split["train"]
eval_dataset = train_test_split["test"]

# Update Trainer setup
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,  # Add evaluation dataset
)

# Step 5: Train the Model
trainer.train()

# Save the Trained Model
trainer.save_model("./greeting_model")
tokenizer.save_pretrained("./greeting_model")

print("Training complete. Model saved at ./greeting_model")
