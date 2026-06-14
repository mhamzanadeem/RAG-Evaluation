<!-- # RAG in the Wild — Case Study Assignment

This assignment is framed as a **case study**: you work with a real-world-style corpus (web search results across multiple domains) and implement four advanced RAG strategies—RAG Fusion, HyDE, CRAG, and Graph RAG—to see which best handles noisy retrieval and varied question types. See **ASSIGNMENT.md** for the full scenario and requirements.

## Requirements

- Python 3.9+
- Node.js 18+ (for the React frontend)

---

## Setup

### Python (backend and pipelines)

```bash
pip install -r requirements.txt
```

Copy `config/config.example.yaml` to `config/config.yaml` and set:

- `dataset_path` — path to `dataset/crag_task_1_and_2_dev_v4.jsonl`
- `embedding_model` — e.g. `all-MiniLM-L6-v2`
- `generation_model` — model name for the LLM you use for answer generation (see below)
- `top_k` — number of chunks to retrieve per query

**LLM / API policy:** **Do not use an OpenAI API key.** Use a **Groq** API key, or a **free** option such as **Google Gemini** (free tier), or another free/local LLM.

Do not commit `config.yaml` if it contains API keys.

### Frontend (React)

```bash
cd frontend
npm install
```

---

## Dataset

This assignment uses the **CRAG Task 1 & 2 dev v4** dataset. 
Download the dataset and place it in the `dataset/` folder yourself.

- **Download (Task 1 & 2, compressed):** [crag_task_1_and_2_dev_v4.jsonl.bz2](https://github.com/facebookresearch/CRAG/raw/refs/heads/main/data/crag_task_1_and_2_dev_v4.jsonl.bz2)
- Decompress the file (e.g. with 7-Zip or `bzip2 -d`), then put the resulting `crag_task_1_and_2_dev_v4.jsonl` inside the `dataset/` folder.
- **Path after setup:** `dataset/crag_task_1_and_2_dev_v4.jsonl`
- **Format:** One JSON object per line. Fields: `query`, `answer`, `alt_ans`, `search_results` (list of up to 5 items; each has `page_snippet`).
- **Schema:** See `docs/dataset.md`.

All `page_snippet` texts from all rows form the global corpus. Build one embedding index from this corpus; all four pipelines retrieve from it.

---

## Running the project

**Evaluation (run from project root):**

```bash
python run_evaluation.py
```

**Frontend (run from project root):**

```bash
cd frontend
npm run dev
```

Open the URL shown (e.g. http://localhost:3000). You need a backend that loads the index and runs the selected pipeline; the React app will call that backend.

---

## Folder structure

Do not change the folder structure. Required layout and the full case-study description are in `ASSIGNMENT.md`. -->


<!-- Project README for RAG case-study assignment -->

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


## License

Check `LICENSE` (if present) or assume assignment-specific usage. Do not include third-party API keys in the repository.

<p align="right">(<a href="#readme-top">back to top</a>)</p>


## Contact

Author: Muhammad Hamza Nadeem

Project repository: (local assignment workspace)

Questions or issues: open an issue in the repo or contact the author directly.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- Improved compatibility of back to top link: See: https://github.com/othneildrew/Best-README-Template/pull/73 -->
<a id="readme-top"></a>
<!--
*** Thanks for checking out the Best-README-Template. If you have a suggestion
*** that would make this better, please fork the repo and create a pull request
*** or simply open an issue with the tag "enhancement".
*** Don't forget to give the project a star!
*** Thanks again! Now go create something AMAZING! :D
-->



<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->
[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![Unlicense License][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url]



<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/othneildrew/Best-README-Template">
    <img src="images/logo.png" alt="Logo" width="80" height="80">
  </a>

  <h3 align="center">Best-README-Template</h3>

  <p align="center">
    An awesome README template to jumpstart your projects!
    <br />
    <a href="https://github.com/othneildrew/Best-README-Template"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href="https://github.com/othneildrew/Best-README-Template">View Demo</a>
    &middot;
    <a href="https://github.com/othneildrew/Best-README-Template/issues/new?labels=bug&template=bug-report---.md">Report Bug</a>
    &middot;
    <a href="https://github.com/othneildrew/Best-README-Template/issues/new?labels=enhancement&template=feature-request---.md">Request Feature</a>
  </p>
</div>



<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project

[![Product Name Screen Shot][product-screenshot]](https://example.com)

There are many great README templates available on GitHub; however, I didn't find one that really suited my needs so I created this enhanced one. I want to create a README template so amazing that it'll be the last one you ever need -- I think this is it.

Here's why:
* Your time should be focused on creating something amazing. A project that solves a problem and helps others
* You shouldn't be doing the same tasks over and over like creating a README from scratch
* You should implement DRY principles to the rest of your life :smile:

Of course, no one template will serve all projects since your needs may be different. So I'll be adding more in the near future. You may also suggest changes by forking this repo and creating a pull request or opening an issue. Thanks to all the people who have contributed to expanding this template!

Use the `BLANK_README.md` to get started.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



### Built With

This section should list any major frameworks/libraries used to bootstrap your project. Leave any add-ons/plugins for the acknowledgements section. Here are a few examples.

* [![Next][Next.js]][Next-url]
* [![React][React.js]][React-url]
* [![Vue][Vue.js]][Vue-url]
* [![Angular][Angular.io]][Angular-url]
* [![Svelte][Svelte.dev]][Svelte-url]
* [![Laravel][Laravel.com]][Laravel-url]
* [![Bootstrap][Bootstrap.com]][Bootstrap-url]
* [![JQuery][JQuery.com]][JQuery-url]

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- GETTING STARTED -->
## Getting Started

This is an example of how you may give instructions on setting up your project locally.
To get a local copy up and running follow these simple example steps.

### Prerequisites

This is an example of how to list things you need to use the software and how to install them.
* npm
  ```sh
  npm install npm@latest -g
  ```

### Installation

_Below is an example of how you can instruct your audience on installing and setting up your app. This template doesn't rely on any external dependencies or services._

1. Get a free API Key at [https://example.com](https://example.com)
2. Clone the repo
   ```sh
   git clone https://github.com/github_username/repo_name.git
   ```
3. Install NPM packages
   ```sh
   npm install
   ```
4. Enter your API in `config.js`
   ```js
   const API_KEY = 'ENTER YOUR API';
   ```
5. Change git remote url to avoid accidental pushes to base project
   ```sh
   git remote set-url origin github_username/repo_name
   git remote -v # confirm the changes
   ```

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- USAGE EXAMPLES -->
## Usage

Use this space to show useful examples of how a project can be used. Additional screenshots, code examples and demos work well in this space. You may also link to more resources.

_For more examples, please refer to the [Documentation](https://example.com)_

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- ROADMAP -->
## Roadmap

- [x] Add Changelog
- [x] Add back to top links
- [ ] Add Additional Templates w/ Examples
- [ ] Add "components" document to easily copy & paste sections of the readme
- [ ] Multi-language Support
    - [ ] Chinese
    - [ ] Spanish

See the [open issues](https://github.com/othneildrew/Best-README-Template/issues) for a full list of proposed features (and known issues).

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- CONTRIBUTING -->
## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".
Don't forget to give the project a star! Thanks again!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Top contributors:

<a href="https://github.com/othneildrew/Best-README-Template/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=othneildrew/Best-README-Template" alt="contrib.rocks image" />
</a>

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- LICENSE -->
## License

Distributed under the Unlicense License. See `LICENSE.txt` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- CONTACT -->
## Contact

Your Name - [@your_twitter](https://twitter.com/your_username) - email@example.com

Project Link: [https://github.com/your_username/repo_name](https://github.com/your_username/repo_name)

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

Use this space to list resources you find helpful and would like to give credit to. I've included a few of my favorites to kick things off!

* [Choose an Open Source License](https://choosealicense.com)
* [GitHub Emoji Cheat Sheet](https://www.webpagefx.com/tools/emoji-cheat-sheet)
* [Malven's Flexbox Cheatsheet](https://flexbox.malven.co/)
* [Malven's Grid Cheatsheet](https://grid.malven.co/)
* [Img Shields](https://shields.io)
* [GitHub Pages](https://pages.github.com)
* [Font Awesome](https://fontawesome.com)
* [React Icons](https://react-icons.github.io/react-icons/search)

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/othneildrew/Best-README-Template.svg?style=for-the-badge
[contributors-url]: https://github.com/othneildrew/Best-README-Template/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/othneildrew/Best-README-Template.svg?style=for-the-badge
[forks-url]: https://github.com/othneildrew/Best-README-Template/network/members
[stars-shield]: https://img.shields.io/github/stars/othneildrew/Best-README-Template.svg?style=for-the-badge
[stars-url]: https://github.com/othneildrew/Best-README-Template/stargazers
[issues-shield]: https://img.shields.io/github/issues/othneildrew/Best-README-Template.svg?style=for-the-badge
[issues-url]: https://github.com/othneildrew/Best-README-Template/issues
[license-shield]: https://img.shields.io/github/license/othneildrew/Best-README-Template.svg?style=for-the-badge
[license-url]: https://github.com/othneildrew/Best-README-Template/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/othneildrew
[product-screenshot]: images/screenshot.png
[Next.js]: https://img.shields.io/badge/next.js-000000?style=for-the-badge&logo=nextdotjs&logoColor=white
[Next-url]: https://nextjs.org/
[React.js]: https://img.shields.io/badge/React-20232A?style=for-the-badge&logo=react&logoColor=61DAFB
[React-url]: https://reactjs.org/
[Vue.js]: https://img.shields.io/badge/Vue.js-35495E?style=for-the-badge&logo=vuedotjs&logoColor=4FC08D
[Vue-url]: https://vuejs.org/
[Angular.io]: https://img.shields.io/badge/Angular-DD0031?style=for-the-badge&logo=angular&logoColor=white
[Angular-url]: https://angular.io/
[Svelte.dev]: https://img.shields.io/badge/Svelte-4A4A55?style=for-the-badge&logo=svelte&logoColor=FF3E00
[Svelte-url]: https://svelte.dev/
[Laravel.com]: https://img.shields.io/badge/Laravel-FF2D20?style=for-the-badge&logo=laravel&logoColor=white
[Laravel-url]: https://laravel.com
[Bootstrap.com]: https://img.shields.io/badge/Bootstrap-563D7C?style=for-the-badge&logo=bootstrap&logoColor=white
[Bootstrap-url]: https://getbootstrap.com
[JQuery.com]: https://img.shields.io/badge/jQuery-0769AD?style=for-the-badge&logo=jquery&logoColor=white
[JQuery-url]: https://jquery.com 