# Legal Language Model

This is a fine-tuned version of the **SmolLM2-135M-Instruct** model, trained on legal texts from the **Indian-Law** dataset by [vishnun0027](https://huggingface.co/datasets/vishnun0027/Indian-Law) on Hugging Face.

## 🤗 Model Availability

**The model is publicly available on Hugging Face:** [saicharan1010/SmolLM2-FT-legal-india](https://huggingface.co/saicharan1010/SmolLM2-FT-legal-india)

## Model Information

- **Base Model**: HuggingFaceTB/SmolLM2-135M-Instruct
- **Dataset**: Indian-Law (25,600 instruction-response pairs after cleaning)
- **Training**: Fine-tuned using SFT (Supervised Fine-Tuning)

## Training Details

- **Training Steps**: 1,000 steps
- **Batch Size**: 16
- **Learning Rate**: 5e-5
- **Final Training Loss**: 1.086

## Performance Evaluation

Evaluation on 1,280 test samples showed improved legal reasoning compared to the base model:

- **BLEU Score**: 0.126 (compared to base model's 0.121)
- **ROUGE-L F-Score**: 0.304

## Usage

Run the model using the `transformers` library:

```python
from transformers import pipeline, AutoTokenizer

# Load tokenizer and create pipeline
tokenizer = AutoTokenizer.from_pretrained("saicharan1010/SmolLM2-FT-legal-india")
pipe = pipeline("text-generation", model="saicharan1010/SmolLM2-FT-legal-india")

# Format with chat template
prompt = "Can a Vakalatnama be revoked or withdrawn in India?"
messages = [{"role": "user", "content": prompt}]
formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False)

# Generate response
response = pipe(formatted_prompt, max_new_tokens=200)
print(response[0]['generated_text'])
```

This model is specifically optimized for legal language tasks in the Indian context. It shows improved understanding of Indian legal terminology and concepts compared to the base model.
