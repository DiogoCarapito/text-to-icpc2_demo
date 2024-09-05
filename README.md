[![Github Actions Workflow](https://github.com/DiogoCarapito/streamlit_app_template/actions/workflows/main.yaml/badge.svg)](https://github.com/DiogoCarapito/streamlit_app_template/actions/workflows/main.yaml)

# streamlit_app_template
Streamlit python project template

## cheat sheet

###  venv
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

