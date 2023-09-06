
# Fake News Classifier ML Web App using FastAPI

This project aims to build a Machine Learning model for classifying fake news articles. We use FastAPI to serve the model as a web application.

## Table of Contents

- [Requirements](#requirements)
- [Installation](#installation)
  - [Create a Virtual Environment](#create-a-virtual-environment)
  - [Activate the Virtual Environment](#activate-the-virtual-environment)
  - [Install Libraries](#install-libraries)
- [Training the Model](#training-the-model)
- [Starting the FastAPI Application](#starting-the-fastapi-application)

## Requirements

- Python 3.x
- pip
- Virtual environment (optional but recommended)

## Installation

### Create a Virtual Environment

To isolate the dependencies, it is recommended to create a virtual environment. Run the following command to create a new virtual environment in a directory named `.venv`:

```bash
python -m venv .venv
```

### Activate the Virtual Environment

Once the virtual environment is created, you need to activate it.

- **For macOS and Linux:**

    ```bash
    source .venv/bin/activate
    ```

- **For Windows:**

    ```bash
    .\.venv\Scripts\activate
    ```

### Install Libraries

After activating the virtual environment, install the required libraries by running:

```bash
pip install -r requirements.txt
```

## Training the Model

Once the libraries are installed, execute `train.py` to train your Machine Learning model. The model will be saved in Pickle format.

```bash
python train.py
```

## Starting the FastAPI Application

After training the model, start the FastAPI application by running:

```bash
uvicorn main:app --reload
```

The FastAPI application will now be running and you can access it via `http://127.0.0.1:8000/docs`.

---

Remember to update the `requirements.txt` file with the specific packages and versions you are using.
