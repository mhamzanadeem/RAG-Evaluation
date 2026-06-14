<a id="readme-top"></a>

## RAG in the Wild — Case Study Assignment

Concise summary: research code to evaluate several Retrieval-Augmented Generation (RAG) pipelines (RAG Fusion, HyDE, CRAG, Graph RAG) on a real-world-style corpus. This repo contains backend pipelines, a small React frontend, dataset instructions, and evaluation tooling used for the assignment.

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#about">About</a></li>
    <li><a href="#built-with">Built With</a></li>
    <li><a href="#getting-started">Getting Started</a></li>
    <li><a href="#configuration">Configuration</a></li>
    <li><a href="#dataset">Dataset</a></li>
    <li><a href="#running">Running</a></li>
    <li><a href="#project-structure">Project Structure</a></li>
    <li><a href="#for-recruiters-ats">For Recruiters / ATS</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
  </ol>
</details>


## About

This codebase implements and evaluates multiple RAG approaches on the CRAG Task 1 & 2 dev dataset. It includes:

- Python pipelines for retrieval, generation and evaluation (`src/` and `src/pipelines/`).
- A small Flask backend (`backend/app.py`) used by the frontend to call pipelines.
- A React frontend in `frontend/` to visualize results and try pipelines interactively.
- An evaluation harness `run_evaluation.py` to run automated metrics across datasets.

This README explains setup and typical workflows so a beginner can run the project end-to-end.

<p align="right">(<a href="#readme-top">back to top</a>)</p>


### Built With

- Python 3.9+ (core pipelines)
- Flask (backend)
- React + Vite (frontend)
- Common ML / NLP libraries (see `requirements.txt`)

<p align="right">(<a href="#readme-top">back to top</a>)</p>


## Getting Started

Follow these steps to get a local copy running. The commands assume you are in the repository root.

### Prerequisites

- Python 3.9 or later
- Node.js 18+ (for frontend development)
- Git

### Installation (backend)

1. Create and activate a Python virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

2. Install Python dependencies:

```bash
pip install -r requirements.txt
```

3. Copy the example config:

```bash
cp config/config.example.yaml config/config.yaml
```

4. Edit `config/config.yaml` to point `dataset_path` to the dataset file and configure model/embedding names.

### Installation (frontend)

```bash
cd frontend
npm install
cd ..
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>


## Configuration

- Keep secrets out of source control. `config/config.example.yaml` is provided as a template — do not commit `config/config.yaml` if it contains API keys.
- Recommended config fields:
  - `dataset_path`: `dataset/crag_task_1_and_2_dev_v4.jsonl`
  - `embedding_model`: model used for embeddings (e.g. `all-MiniLM-L6-v2`)
  - `generation_model`: model name for the LLM used for generation
  - `top_k`: retrieval size per query

<p align="right">(<a href="#readme-top">back to top</a>)</p>


## Dataset

This assignment uses the CRAG dev dataset. Place `crag_task_1_and_2_dev_v4.jsonl` in the `dataset/` folder.

- Download (compressed): https://github.com/facebookresearch/CRAG/raw/refs/heads/main/data/crag_task_1_and_2_dev_v4.jsonl.bz2
- Decompress (example):

```bash
bzip2 -d crag_task_1_and_2_dev_v4.jsonl.bz2
mv crag_task_1_and_2_dev_v4.jsonl dataset/
```

See `docs/dataset.md` for the schema and details.

<p align="right">(<a href="#readme-top">back to top</a>)</p>


## Running

### Run evaluation (automated metrics)

From project root:

```bash
python run_evaluation.py
```

Output is written to `results/evaluation_results.json`.

### Run backend (development)

Start the Flask backend:

```bash
python backend/app.py
```

This loads pipeline components and exposes endpoints the frontend can call.

### Run frontend (development)

```bash
cd frontend
npm run dev
```

Open the URL shown by Vite (typically `http://localhost:3000`).

<p align="right">(<a href="#readme-top">back to top</a>)</p>


## Project Structure

- `backend/` — Flask app and backend endpoints
- `config/` — example configuration files
- `dataset/` — place dataset files here
- `frontend/` — React app (Vite)
- `src/` — Python pipelines: `corpus.py`, `retrieval.py`, `generation.py`, `evaluation.py` and `pipelines/`
- `run_evaluation.py` — high-level evaluation runner
- `results/` — evaluation outputs

<p align="right">(<a href="#readme-top">back to top</a>)</p>


## For Recruiters / ATS

Skills and keywords: Retrieval-Augmented Generation (RAG), information retrieval, embeddings, Python, Flask, React, Vite, evaluation metrics, JSONL datasets, NLP pipelines, model integration.

Quick demo commands to validate the project for interviews:

```bash
# run a quick evaluation
python run_evaluation.py --limit 10

# start backend
python backend/app.py

# run frontend
cd frontend && npm run dev
```

Add these steps to your verification checklist when reviewing the repository.

<p align="right">(<a href="#readme-top">back to top</a>)</p>


## Contributing

If you find issues or want to add improvements:

1. Fork the repo
2. Create a feature branch
3. Open a pull request with a clear description and tests/examples

Please keep `config/config.yaml` out of PRs when it contains secrets.

<p align="right">(<a href="#readme-top">back to top</a>)</p>


## Contact

Author: Muhammad Hamza Nadeem

