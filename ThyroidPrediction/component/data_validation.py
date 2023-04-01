from ThyroidPrediction.logger import logging
from ThyroidPrediction.exception import ThyroidException
from ThyroidPrediction.entity.config_entity import DataValidationConfig
from ThyroidPrediction.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact


from evidently import ColumnMapping
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset
from evidently.metrics import *
from evidently.tests import *

import pandas as pd
import os, sys
import json


class DataValidation:

    def __init__(self, data_validation_config:DataValidationConfig,data_ingestion_artifact: DataIngestionArtifact):
        try:
            
            self.data_validation_config = data_validation_config
            self.data_ingestion_artifact = data_ingestion_artifact

        except Exception as e:
            raise ThyroidException(e,sys) from e
        

    def get_train_and_test_df(self):
        try:
            train_df = pd.read_csv(self.data_ingestion_artifact.train_file_path)
            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)

            return train_df, test_df
        
        except Exception as e:
            raise ThyroidException(e, sys) from e

    def is_train_test_file_exists(self) -> bool:
        try:
            logging.info(f"Checking if train and test csv file is available")
            is_train_file_exists = False
            is_test_file_exists = False

            train_file_path = self.data_ingestion_artifact.train_file_path
            test_file_path = self.data_ingestion_artifact.test_file_path

            is_train_file_exists = os.path.exists(train_file_path)
            is_test_file_exists = os.path.exists(test_file_path)
            
            is_available = is_train_file_exists and is_test_file_exists

            logging.info(f"Is train adn test file exists? --> {is_available}")

            if not is_available:
                training_file = self.data_ingestion_artifact.train_file_path
                testing_file = self.data_ingestion_artifact.test_file_path

                message = f"Training file [{training_file}] or Testing file [{testing_file}] is not present"
                
                logging.info(message)
                raise Exception(message)


            return is_available
        except Exception as e:
            raise ThyroidException(e, sys) from e


    def validate_dataset_schema(self) -> bool:
        try:
            validation_status = False
            #Assigment validate training and testing dataset using schema file
                #1. Number of Column
                #2. Check the value of ocean proximity 
                # acceptable values     <1H OCEAN
                # INLAND
                # ISLAND
                # NEAR BAY
                # NEAR OCEAN
                #3. Check column names


            validation_status = True
            return validation_status
        except Exception as e:
            raise ThyroidException(e, sys) from e

   


    def get_and_save_data_drift_report(self):
        try:           
            #####THIS METHOD IS WORKING BUT DASHBOARD AND MODEL_PROFILE HAS BEEN DEPRICATED ######
            #profile = Profile(sections=[DataDriftProfileSection()])
            #train_df, test_df = self.get_train_and_test_df()
            #profile.calculate(train_df, test_df)
            #profile.json()
            #report = json.loads(profile.json())
            
            ########### NEW METHOD ###############
            train_df, test_df = self.get_train_and_test_df()
            report = Report(metrics=[ DataDriftPreset(),])
            report.run(reference_data=train_df, current_data=test_df)

            ########### SAVING JSON FILE ########
            report_file_path = self.data_validation_config.report_file_path
            report_dir = os.path.dirname(report_file_path)
            os.makedirs(report_dir, exist_ok=True)            
            report.save_json(filename= report_file_path,)

            ############## SAVING HTML FILE #########
            report_page_file_path = self.data_validation_config.report_page_file_path
            report_page_dir = os.path.dirname(report_page_file_path)
            os.makedirs(report_page_dir, exist_ok=True)            
            report.save_html(filename= report_page_file_path)
            #####################################################################
            
            #report_file_path = self.data_validation_config.report_file_path
            #report_dir = os.path.dirname(report_file_path)
            #os.makedirs(report_dir, exist_ok=True)
            
            #with open(report_file_path, "w") as report_file:
            #    json.dump(report, report_file, indent=6)

            return report

        except Exception as e:
            raise ThyroidException(e, sys) from e


    def save_data_drift_report_page(self):
        try:
            ######### DEPRICATED METHOD ########
            #dashboard = Dashboard(tabs=[DataDriftTab()])
            #train_df, test_df = self.get_train_and_test_df()
            #dashboard.calculate(train_df, test_df)
            ################################

            #report_page_file_path = self.data_validation_config.report_page_file_path
            #report_page_dir = os.path.dirname(report_page_file_path)
            #os.makedirs(report_page_dir, exist_ok=True)

            #dashboard.save(report_page_file_path)
            
            #report.save_html(filename= report_file_path)
            pass
            
        except Exception as e:
            raise ThyroidException(e, sys) from e

    def is_data_drift_found(self) -> bool:
        try:
            report  = self.get_and_save_data_drift_report()
            self.save_data_drift_report_page()

            return True
        except Exception as e:
            raise ThyroidException(e, sys) from e


    def initiate_data_validation(self) -> DataValidationArtifact:
        try:
            self.is_train_test_file_exists()            
            self.validate_dataset_schema()
            self.is_data_drift_found()

            data_validation_artifact = DataValidationArtifact(schema_file_path= self.data_validation_config.schema_file_path,
                                                              report_file_path= self.data_validation_config.report_file_path,
                                                              report_page_file_path= self.data_validation_config.report_page_file_path,
                                                              is_validated= True,
                                                              message= "Data Validation Performed Sucessfully")
            
            logging.info(f"Data validation artifact: {data_validation_artifact}")

            return data_validation_artifact
        except Exception as e:
            raise ThyroidException(e, sys) from e