# mmarcus-thuillier.personal-rugby-prediction

## Project Overview

**Scope:** Establish a method for accurate rugby game point spread prediction

**Objective:** Develop an xgboost model with inputs including team, location, and previous performance to predict final game results.

### Results

- Developed ML model that outputs a prediction on a test set.

### ML model

- Data in ```Artefacts/``` is used in the model training and prediction

## Structure

    execute.sh
        - bash script to run the project
    requirements.txt
        - file with library requirements
    main.ipynb
        - Notebook for development
    Code/
        main.py
            - Script that calls functions into main module and executes project
        dataingestion.py
            - Script to pull in data and preprocess both training and testing datasets
        modeltraining.py
            - Script to train model and save artifacts and test output
        Config/
            config.yaml
                - config information for model run
        Artefacts/
            model.pkl
                - ML model
            train_data.csv
            test_data.csv
            output.csv

## Update model run

1. Run the model through main.py.

## Future Work envisioned

1. Automate data pull to move away from flat files
2. Iterate on the model to avoid overfitting (additional features, external dataset)
3. Automate data push to a database, away from flat files
4. Automate a model refresh when new results come into the dataset