from flask import Flask
#from sklearn.pipeline import Pipeline
from ThyroidPrediction.config.configuration import Configuration
from ThyroidPrediction.constant import CURRENT_TIME_STAMP
from ThyroidPrediction.entity.artifact_entity import DataIngestionArtifact
from ThyroidPrediction.entity.config_entity import DataValidationConfig, BaseDataIngestionConfig

import os
import sys
from ThyroidPrediction.logger import logging
from ThyroidPrediction.exception import ThyroidException
from ThyroidPrediction. component import data_ingestion, data_transformation, data_validation
from ThyroidPrediction.pipeline.pipeline import Pipeline
app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def index():
    try:

        
        logging.info("split data")

        config_path = os.path.join("config","config.yaml")
        pipeline = Pipeline(Configuration(config_file_path= config_path))
        pipeline.run_pipeline()
        
        #data_validation_config = Configuration().get_data_ingestion_config()
        
        #data_validation_config = Configuration().get_data_transformation_config()
        #print(data_validation_config)
        #schema_file_path = r"C:\Users\HIMANSHU\ML_Projects\config\schema.yaml"
        #file_path = r"C:\Users\HIMANSHU\ML_Projects\housing\artifact\data_ingestion\2023-03-19-14-06-27\ingested_data\train\housing.csv"
        #df = DataTransformation.load_data(file_path=file_path, schema_file_path=schema_file_path)
        #print(df.columns)
        #print(df.dtypes)
        #return Configuration(config_file_path=config_path, time_stamp= CURRENT_TIME_STAMP).get_base_data_transformation_config()
        #return data_validation.DataValidation(data_ingestion_artifact=DataIngestionArtifact, data_validation_config=DataValidationConfig).initiate_data_validation()
    
    except Exception as e:
        raise ThyroidException(e,sys) from e
    

    #return data_ingestion.DataIngestion(data_ingestion_config=BaseDataIngestionConfig).initiate_data_ingestion()
    #return data_validation.DataValidation(data_validation_config=DataValidationConfig,data_ingestion_artifact= DataIngestionArtifact).get_and_save_data_drift_report()

if __name__ == "__main__":
    app.run(debug=True)
