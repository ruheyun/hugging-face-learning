from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import re

model_name = "Qwen/Qwen3-0.6B"

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)

messages = [
    {
        "role": "system",
        "content": "You are a helpful assistant. Do not output thinking process."
    },
    {
        "role": "system",
        "content": (
            "You can access and use the full conversation history. "
            "Answer questions about previous messages accurately."
        )
    }
]

while True:

    user_input = input("User: ")

    if user_input.lower() in ["exit", "quit"]:
        break

    messages.append(
        {"role": "user", "content": user_input}
    )

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False   
    )

    inputs = tokenizer(
        text,
        return_tensors="pt"
    ).to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        do_sample=True,
        temperature=0.7,
        top_p=0.9
    )

    generated_ids = outputs[0][inputs.input_ids.shape[-1]:]

    response = tokenizer.decode(
        generated_ids,
        skip_special_tokens=True
    )

    response = re.sub(
        r"<think>.*?</think>",
        "",
        response,
        flags=re.DOTALL
    ).strip()

    print("Assistant:", response)

    messages.append(
        {"role": "assistant", "content": response}
    )
