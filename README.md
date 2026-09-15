# fine-tuning-sentence-transformers
A Very Simple Demo of Fine Tuning Sentence Transformers

This repository provides a practical demonstration of how to fine-tune a Sentence Transformer model on a custom dataset and then use the fine-tuned model to generate sentence embeddings. The scripts utilize the PyTorch library and Sentence Transformers for this purpose.

This project is a simple example of how to fine-tune a Sentence Transformer model. It is not designed for large-scale or real-world applications.

## GPU

Training and inference pick a device in `device.py`: CUDA if `torch.cuda.is_available()`, otherwise CPU. The scripts print `Using GPU: <name>` or `Using CPU (CUDA not available)` at startup.

You do not need to set `device=` yourself. To force a specific GPU on the host, set `CUDA_VISIBLE_DEVICES` before running, for example `CUDA_VISIBLE_DEVICES=0`.

### Docker

The image installs PyTorch `2.3.1+cu121`. Compose passes NVIDIA GPUs into every service with `gpus: all`.

Host requirements:

- NVIDIA driver (CUDA 12.1 wheels typically need driver ≥ 525)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
- WSL2 GPU support if Docker runs on Windows

Rebuild after changing the Dockerfile (CPU vs CUDA wheels):

```bash
docker compose build
```

Confirm the container sees a GPU:

```bash
docker compose run --rm shell python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"
```

If CUDA is not visible, training still runs on CPU.

### Local (venv)

`requirements.txt` does not install PyTorch. For GPU locally, install a CUDA build first, then the rest:

```bash
pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

CPU-only local install:

```bash
pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

## Docker Compose

Build the image:

```bash
docker compose build
```

Fine-tune the model in a one-off container:

```bash
docker compose run --rm train
```

Run inference with the fine-tuned model:

```bash
docker compose run --rm inference
```

Open an interactive shell in the container:

```bash
docker compose run --rm shell
```

Run an arbitrary CLI command in the container:

```bash
docker compose run --rm shell python --version
docker compose run --rm shell python 02_using_your_model.py
docker compose run --rm shell python 03_compare_input_output.py --rank \
    --input "I can't log in to my account." \
    --output "Unable to access my account."
```

The project directory is bind-mounted at `/app`, so host scripts (including
`03_compare_input_output.py`) and `tuned_models` are used without rebuilding.
Rebuild the image after `Dockerfile` or `requirements.txt` changes:

```bash
docker compose build
```

Downloaded Hugging Face models are retained in the `huggingface-cache` Docker volume.

If the container cannot reach `huggingface.co` (common in mainland China),
the Compose file and Dockerfile set `HF_ENDPOINT=https://hf-mirror.com`.
Override that variable if you use another mirror or a VPN.

## Setup and Installation
To set up and run the example, follow these steps:

1. Clone this repository: `git clone https://github.com/adiekaye/fine-tuning-sentence-transformers.git`
2. Navigate to the project directory: cd sentence-transformer-tuning
3. Create a virtual environment: `python -m venv venv`
4. Activate the virtual environment:
- For Windows: `venv\Scripts\activate`
- For macOS/Linux: `source venv/bin/activate`
5. Install PyTorch (CUDA or CPU, see [GPU](#gpu)), then `pip install -r requirements.txt`
6. Run the script `01_tuning_your_model.py` to fine-tune the model: `python 01_tuning_your_model.py`
7. Run the evaluation scripts (see [Testing](#testing)): `python 02_using_your_model.py` and `python 03_compare_input_output.py`
8. To exit the virtual environment, type `deactivate` in the terminal.

Note: The virtual environment and requirements installation steps are optional but recommended to ensure compatibility and avoid conflicts with other Python packages you may have installed.

## Contents
- `01_tuning_your_model.py`: Script that fine-tunes a Sentence Transformer model on a custom dataset.
- `02_using_your_model.py`: Script that compares original vs fine-tuned embeddings on fixed related/unrelated pairs.
- `03_compare_input_output.py`: Script that scores your own input/output pairs and optionally ranks a candidate pool.
- `device.py`: Chooses CUDA when available, otherwise CPU, and prints the device in use.
- `requirements.txt`: Lists the required libraries for this project (PyTorch is installed separately; see [GPU](#gpu)).
- `README.md`: Provides instructions for setting up and running the example, and explains the contents of the repository.
- `.gitignore`: A simple Git configuration file to ignore the virtual environment directory and other non-essential files.
- `/tuned_models`: A directory to store your fine tuned models.
- `/tuned_models/.gitignore`: A gitignore file to make sure you don't accidentally commit your fine tuned model.

## Usage
The scripts are executable as they are. You might need to adjust the path of the fine-tuned model (`tuned_models/fine_tuned_model`) depending on your directory structure.

The `01_tuning_your_model.py` script will train a Sentence Transformer model using the specified training examples and then save the fine-tuned model.

## Testing
Fine-tuning is trained with triplet loss, so the useful signal is **relative** similarity: related intents should get closer, unrelated intents should get farther. Comparing the same sentence’s vector before vs after fine-tuning is a weak check (that cosine often stays near 1.0).

Run these after `01_tuning_your_model.py` has written `tuned_models/fine_tuned_model`.

### Pair similarity (`02_using_your_model.py`)

Loads the original `all-mpnet-base-v2` model and the fine-tuned model, then prints cosine similarity for a fixed list of related and unrelated sentence pairs, plus a `delta` (fine-tuned minus original).

```bash
python 02_using_your_model.py
```

```bash
docker compose run --rm inference
```

What to look for:

- Related pairs (same intent): `delta` is usually **positive**.
- Unrelated pairs (different intents): `delta` is usually **negative**.

### Custom input / output (`03_compare_input_output.py`)

Treat **input** as the query (user utterance) and **output** as the text you expect to match (paraphrase, FAQ, or candidate document). The script reports original vs fine-tuned cosine similarity for each pair.

One pair:

```bash
python 03_compare_input_output.py \
    --input "I can't log in to my account." \
    --output "Unable to access my account."
```

Several pairs (repeat `--input` / `--output`):

```bash
python 03_compare_input_output.py \
    --input "How do I reset my password?" \
    --output "I forgot my password. What do I do?" \
    --input "How do I reset my password?" \
    --output "My order hasn't arrived yet."
```

Retrieval-style check: rank a built-in candidate pool for each input. If you also pass `--output`, the expected text is marked and its rank is compared.

```bash
python 03_compare_input_output.py --rank \
    --input "I can't log in to my account." \
    --output "Unable to access my account."
```

With Docker:

```bash
docker compose run --rm shell python 03_compare_input_output.py --rank \
    --input "I can't log in to my account." \
    --output "Unable to access my account."
```

With no arguments, the script runs a built-in demo set of related and unrelated pairs.

What to look for:

- Matching input/output: higher cosine and a **smaller** rank number after fine-tuning.
- Mismatched input/output: lower cosine after fine-tuning.

Optional flags: `--top-k`, `--original-model`, `--fine-tuned-model`.

## License
This project is licensed under the MIT License.
