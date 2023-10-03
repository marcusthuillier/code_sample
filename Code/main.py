import yaml

from dataingestion import DataIngestion
from modeltraining import ModelTraining

with open("CONFIG/config.yaml", "r") as yamlfile:
    cfg = yaml.safe_load(yamlfile)

def main():
    dataingestion = DataIngestion(cfg)
    train_data, test_data = dataingestion.finalize_dataset()
    we_have_actuals = True
    modeltraining = ModelTraining(cfg, train_data, test_data, we_have_actuals)
    modeltraining.train_model()
    
if __name__ == "__main__":
    main()