# text-to-icpc2 demo
Streamlit demo of text-to-icpc2 classification model to clasify diagnosis into icpc2 codes

[![Github Actions Workflow](https://github.com/DiogoCarapito/text-to-icpc2_demo/actions/workflows/main.yaml/badge.svg)](https://github.com/DiogoCarapito/text-to-icpc2_demo/actions/workflows/main.yaml)



## cheat sheet

### venv

create venv

```bash
python3 -m venv .venv
```

activate venv

```bash
source .venv/bin/activate
```

### Dockerfile

#### build

```bash
docker build -t Home:latest .
````

#### check image id

```bash
docker images
````

#### run with image id

```bash
docker run -p 8501:8501 Home:latest
````
