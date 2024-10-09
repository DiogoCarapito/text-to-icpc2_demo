# text-to-icpc2 demo

Streamlit demo of text-to-icpc2 classification model to clasify diagnosis into icpc2 codes.

Model available here: [https://huggingface.co/diogocarapito/text-to-icpc2](https://huggingface.co/diogocarapito/text-to-icpc2)

Training dataset here: [https://huggingface.co/datasets/diogocarapito/text-to-icpc2](https://huggingface.co/datasets/diogocarapito/text-to-icpc2)

Code here: [https://github.com/DiogoCarapito/text-to-icpc2](https://github.com/DiogoCarapito/text-to-icpc2)

[![Github Actions Workflow](https://github.com/DiogoCarapito/text-to-icpc2_demo/actions/workflows/main.yaml/badge.svg)](https://github.com/DiogoCarapito/text-to-icpc2_demo/actions/workflows/main.yaml)

## cheat sheet

### venv

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

### Dockerfile

#### build

```bash
docker build -t app:latest .
````

#### check image id

```bash
docker images
````

#### run with image id

```bash
docker run -p 8501:8501 app:latest
````
