# neural-chameleons

A replication of the [Neural Chameleons paper](https://arxiv.org/pdf/2512.11949), and hopefully some project extensions.

- Generate dataset for eliciting concepts
- Train probes to detect benign concepts in activations
- Train the chameleons to evade probe detection
- Train new probes *post hoc* to detect both benign and safety-related concepts in chameleons
- Evaluate chameleons on both model performance and probe evasion
- Project extensions!!

## Setup

### 1. Clone and install

```bash
git clone https://github.com/<your-org>/neural-chameleons.git
cd neural-chameleons
pip install -e .
```

> **RunPod note:** The RunPod PyTorch template ships with a CUDA-enabled PyTorch already installed. `pip install -e .` will reuse it rather than downloading a new build.

### 2. Configure environment variables

Create a `.env` file in the project root:

```bash
# Required for the Groq judge model (free at console.groq.com)
GROQ_API_KEY=<your-groq-key>
```

### 3. Launch the dataset generation notebook

```bash
jupyter notebook src/generate_dataset.ipynb
```

Or on a headless RunPod server:

```bash
jupyter notebook --no-browser --ip=0.0.0.0 --port=8888 src/generate_dataset.ipynb
```

Then open the proxied URL shown in the terminal output.

## Project structure

```
neural-chameleons/
├── data/                   # Generated datasets
├── src/
│   ├── generate_dataset.ipynb  # Dataset generation (start here)
│   ├── train.py                # Chameleon fine-tuning
│   ├── eval.py                 # Evaluation
│   └── utils.py                # Probes and losses
├── pyproject.toml          # Package definition and dependencies
└── .env                    # API keys (not committed)
```

## RunPod inference setup

The dataset generation notebook can use either Groq (fast, free, no GPU needed) or Gemma-2-9B loaded directly on a RunPod GPU via `transformers.pipeline`:

1. Launch a pod with an A100 or similar GPU (~20 GB VRAM for Gemma-2-9B in bfloat16). Use the **RunPod PyTorch** template.
2. Clone the repo and run `pip install -e .` on the pod.
3. In `src/generate_dataset.ipynb`, swap the `generate_client` assignment: comment out the Groq line and uncomment the `pipeline` block:

```python
generate_client = pipeline(
    "text-generation",
    model=GENERATE_MODEL,
    dtype=torch.bfloat16,
    device_map="auto",
)
```

The model (`IlyaGusev/gemma-2-9b-it-abliterated`) will be downloaded from HuggingFace on first run.
