- This is what you need to do for each project

1. Think of a problem that's interesting for you and find a dataset for that
2. Describe this problem and explain how a model could be used
3. Prepare the data and doing EDA, analyze important features
4. Train multiple models, tune their performance and select the best model
5. Export the notebook into a script
6. Put your model into a web service and deploy it locally with Docker
7. Bonus points for deploying the service to the cloud

Deliverables
For a project, you repository/folder should contain the following:

README.md with
Description of the problem
Instructions on how to run the project

Data
    You should either commit the dataset you used or have clear instructions how to download the dataset

Notebook (suggested name - notebook.ipynb) with
    Data preparation and data cleaning
    EDA, feature importance analysis
    Model selection process and parameter tuning


Script train.py (suggested name)
    Training the final model
    Saving it to a file (e.g. pickle) or saving it with specialized software (BentoML)

Script predict.py (suggested name)
    Loading the model
    Serving it via a web service (with Flask or specialized software - BentoML, KServe, etc)

Files with dependencies
    Pipenv and Pipenv.lock if you use Pipenv
    or equivalents: conda environment file, requirements.txt or pyproject.toml

Dockerfile for running the service

Deployment
    URL to the service you deployed or
    Video or image of how you interact with the deployed service

Evaluation Criteria
The project will be evaluated using these criteria:

Problem description
0 points: Problem is not described
1 point: Problem is described in README birefly without much details
2 points: Problem is described in README with enough context, so it's clear what the problem is and how the solution will be used
EDA
0 points: No EDA
1 point: Basic EDA (looking at min-max values, checking for missing values)
2 points: Extensive EDA (ranges of values, missing values, analysis of target variable, feature importance analysis) For images: analyzing the content of the images. For texts: frequent words, word clouds, etc
Model training
0 points: No model training
1 point: Trained only one model, no parameter tuning
2 points: Trained multiple models (linear and tree-based). For neural networks: tried multiple variations - with dropout or without, with extra inner layers or without
3 points: Trained multiple models and tuned their parameters. For neural networks: same as previous, but also with tuning: adjusting learning rate, dropout rate, size of the inner layer, etc.
Exporting notebook to script
0 points: No script for training a model
1 point: The logic for training the model is exported to a separate script
Reproducibility
0 points: Not possitble to execute the notebook and the training script. Data is missing or it's not easiliy accessible
1 point: It's possible to re-execute the notebook and the training script without errors. The dataset is committed in the project repository or there are clear instructions on how to download the data
Model deployment
0 points: Model is not deployed
1 point: Model is deployed (with Flask, BentoML or a similar framework)
Dependency and enviroment management
0 points: No dependency management
1 point: Provided a file with dependencies (requirements.txt, pipfile, bentofile.yaml with dependencies, etc)
2 points: Provided a file with dependencies and used virtual environment. README says how to install the dependencies and how to activate the env
Containerization
0 points: No containerization
1 point: Dockerfile is provided or a tool that creates a docker image is used (e.g. BentoML)
2 points: The application is containerized and the README describes how to build a container and how to run it
Cloud deployment
0 points: No deployment to the cloud
1 point: Docs describe clearly (with code) how to deploy the service to cloud or kubernetes cluster (local or remote)
2 points: There's code for deployment to cloud or kubernetes cluster (local or remote). There's a URL for testing - or video/screenshot of testing it