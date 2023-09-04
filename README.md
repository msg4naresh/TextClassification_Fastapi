# TextClassification_Fastapi
Building Fake News classifier ML Web app using Fastapi

Installation & Setup
Follow these steps to get the development environment running.

Create a Virtual Environment: Run the following command to create a new virtual environment in a directory named .venv.
"python -m venv .venv"
Activate the Virtual Environment: If you're using a Mac, activate the virtual environment with the following command:
"source .venv/bin/activate"
Install the necessary libraries by running the following command: 
"pip install -r requirements.txt"
Run the Training Script: Execute train.py to train your model. The model will be saved in Pickle format.
"python train.py"
Start the FastAPI Application: Run main.py using the following uvicorn command to start your FastAPI application:
"uvicorn main:app --reload"

