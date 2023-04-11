from flask import Flask
from ThyroidPrediction.entity.artifact_entity import DataIngestionArtifact
from ThyroidPrediction.entity.config_entity import DataValidationConfig, BaseDataIngestionConfig


from ThyroidPrediction.logger import logging
from ThyroidPrediction.exception import ThyroidException
from ThyroidPrediction. component import data_ingestion, data_transformation, data_validation
app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def index():
        
    logging.info("split data")
    
    return data_ingestion.DataIngestion(data_ingestion_config=BaseDataIngestionConfig).initiate_data_ingestion()
    
    #return data_validation.DataValidation(data_validation_config=DataValidationConfig,data_ingestion_artifact= DataIngestionArtifact).get_and_save_data_drift_report()

if __name__ == "__main__":
    app.run(debug=True)