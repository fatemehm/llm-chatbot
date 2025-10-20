#!/usr/bin/env python3
"""
Deploy LLM Chatbot to Hugging Face Hub
"""
import os
import sys
from datetime import datetime
from pathlib import Path

try:
    from huggingface_hub import HfApi, create_repo, upload_folder
except ImportError:
    print(" huggingface_hub not installed!")
    print("Run: pip install huggingface_hub")
    sys.exit(1)


def get_hf_token():
    """Get Hugging Face token from environment"""
    token = os.getenv("HF_TOKEN")
    if not token:
        raise ValueError("HF_TOKEN environment variable not set")
    return token


def create_model_card(repo_id, metrics=None):
    """Generate model card content"""

    card_content = f"""---
language: en
license: apache-2.0
tags:
- conversational
- chatbot
- llm
- fine-tuned
- tech-support
datasets:
- custom
model-index:
- name: {repo_id.split('/')[-1]}
  results: []
---

#  LLM Tech Support Chatbot

Fine-tuned language model for technical support conversations.

## Model Details

- **Developed by**: Your Organization
- **Model type**: Conversational Language Model
- **Language**: English
- **License**: Apache 2.0
- **Use case**: Technical support, customer service

## Training Data

Fine-tuned on technical support Q&A dataset covering:
- Technical troubleshooting
- Billing inquiries
- General customer support

## Usage

### With Transformers
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "{repo_id}"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Generate response
inputs = tokenizer("How do I reset my password?", return_tensors="pt")
outputs = model.generate(**inputs, max_length=100)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

### With Inference API
```python
import requests

API_URL = "https://api-inference.huggingface.co/models/{repo_id}"
headers = {{"Authorization": "Bearer YOUR_HF_TOKEN"}}

def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()

output = query({{"inputs": "How do I reset my password?"}})
print(output)
```

## Deployment Info

- **Last deployed**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
- **Automated deployment**: Via GitHub Actions
- **Drift monitoring**: Enabled

## Limitations

- Best for technical support domain
- May require fine-tuning for other domains
- Performance varies on out-of-distribution inputs

## Citation
```bibtex
@misc{{llm-chatbot,
  author = {{Your Organization}},
  title = {{LLM Tech Support Chatbot}},
  year = {{{datetime.now().year}}},
  publisher = {{Hugging Face}},
  howpublished = {{\\url{{https://huggingface.co/{repo_id}}}}}
}}
```

---

*Automatically deployed via GitHub Actions*
"""

    return card_content


def deploy_model(model_path, repo_id, token, private=False):
    """Deploy model to Hugging Face Hub"""

    print(f"\n Deploying to Hugging Face Hub")
    print(f"Repository: {repo_id}")
    print(f"Model path: {model_path}")
    print(f"Private: {private}")

    api = HfApi(token=token)

    # Create repository if it doesn't exist
    try:
        print(f"\n Creating/verifying repository...")
        create_repo(repo_id=repo_id, token=token, private=private, exist_ok=True, repo_type="model")
        print(f" Repository ready: https://huggingface.co/{repo_id}")
    except Exception as e:
        print(f" Repository: {e}")

    # Generate and save model card
    try:
        print(f"\n Creating model card...")
        model_card = create_model_card(repo_id)

        readme_path = model_path / "README.md"
        with open(readme_path, "w") as f:
            f.write(model_card)
        print(f" Model card created")
    except Exception as e:
        print(f" Model card: {e}")

    # Upload model files
    try:
        print(f"\n Uploading model files...")

        if not model_path.exists():
            raise FileNotFoundError(f"Model path does not exist: {model_path}")

        # Upload entire folder
        upload_folder(
            folder_path=str(model_path),
            repo_id=repo_id,
            repo_type="model",
            token=token,
            commit_message=f"Deploy model - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        )

        print(f" Model uploaded successfully!")
        print(f"\n Model URL: https://huggingface.co/{repo_id}")

        return True

    except Exception as e:
        print(f" Upload failed: {e}")
        return False


def main():
    """Main deployment workflow"""

    print("=" * 60)
    print(" Hugging Face Deployment Pipeline")
    print("=" * 60)

    try:
        # Configuration from environment variables
        MODEL_PATH = Path(os.getenv("MODEL_PATH", "models/distilgpt2-lora"))
        REPO_ID = os.getenv("HF_REPO_ID", "your-username/llm-chatbot")
        PRIVATE = os.getenv("HF_PRIVATE", "false").lower() == "true"

        print(f"\n Configuration:")
        print(f"   Model path: {MODEL_PATH}")
        print(f"   HF Repo: {REPO_ID}")
        print(f"   Private: {PRIVATE}")

        # Get token
        token = get_hf_token()
        print(f"✅ HF token loaded")

        # Check if model exists
        if not MODEL_PATH.exists():
            print(f"\n Model not found at {MODEL_PATH}")
            print("Available models:")
            models_dir = Path("models")
            if models_dir.exists():
                for model in models_dir.iterdir():
                    if model.is_dir():
                        print(f"   - {model}")
            sys.exit(1)

        # Deploy model
        success = deploy_model(model_path=MODEL_PATH, repo_id=REPO_ID, token=token, private=PRIVATE)

        if success:
            print("\n" + "=" * 60)
            print(" DEPLOYMENT SUCCESSFUL")
            print("=" * 60)
            print(f"\n Model URL: https://huggingface.co/{REPO_ID}")
            print(f" Model card: https://huggingface.co/{REPO_ID}#model-card")
            print(f" Files: https://huggingface.co/{REPO_ID}/tree/main")
            sys.exit(0)
        else:
            sys.exit(1)

    except Exception as e:
        print(f"\n DEPLOYMENT FAILED: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
