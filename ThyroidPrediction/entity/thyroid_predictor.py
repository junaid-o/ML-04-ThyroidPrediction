import os
import sys
from ThyroidPrediction.logger import logging
from ThyroidPrediction.exception import ThyroidException
from ThyroidPrediction.util.util import load_object

import pandas as pd


class HousingData:

    def __init__(self, age: float, TSH: float, T3: float, TT4: float, T4U: float, FTI: float,
                 sex, on_thyroxine, query_on_thyroxine, on_antithyroid_medication, sick, pregnant,
                 thyroid_surgery, I131_treatment, query_hypothyroid, query_hyperthyroid, lithium, goitre,
                 tumor, hypopituitary, psych, referral_source_SVHC, referral_source_SVHD, referral_source_SVI,
                 referral_source_other, major_class_encoded
                 ):

        logging.info(f"{'>>' * 30} HousingData log started {'<<' * 30} ")

        try:
            self.age = age
            self.TSH = TSH
            self.T3 = T3
            self.TT4 = TT4
            self.T4U = T4U
            self.FTI = FTI
            self.sex = sex
            self.on_thyroxine = on_thyroxine
            self.query_on_thyroxine = query_on_thyroxine
            self.on_antithyroid_medication = on_antithyroid_medication
            self.sick = sick
            self.pregnant = pregnant
            self.thyroid_surgery = thyroid_surgery
            self.I131_treatment = I131_treatment
            self.query_hypothyroid = query_hypothyroid
            self.query_hyperthyroid = query_hyperthyroid
            self.lithium = lithium
            self.goitre = goitre
            self.tumor = tumor
            self.hypopituitary = hypopituitary
            self.psych = psych
            self.referral_source_SVHC = referral_source_SVHC
            self.referral_source_SVHD = referral_source_SVHD
            self.referral_source_SVI = referral_source_SVI
            self.referral_source_other = referral_source_other
            self.major_class_encoded = major_class_encoded
            


        except Exception as e:
            raise ThyroidException(e, sys) from e

    def get_housing_input_data_frame(self):

        try:
            housing_input_dict = self.get_housing_data_as_dict()
            return pd.DataFrame(housing_input_dict)
        except Exception as e:
            raise ThyroidException(e, sys) from e

    def get_housing_data_as_dict(self):
        try:
            logging.info(f"getting housing data as_dict")

            input_data = {"age": [self.age],
                          "TSH": [self.TSH],
                          "T3": [self.T3],
                          "TT4": [self.TT4],
                          "T4U": [self.T4U],
                          "FTI": [self.FTI],
                          "sex":[self.sex],
                          "on_thyroxine":[self.on_thyroxine],
                          "query_on_thyroxine":[self.query_on_thyroxine],
                          "on_antithyroid_medication":[self.on_antithyroid_medication],
                          "sick":[self.sick],
                          "pregnant":[self.pregnant],
                          "thyroid_surgery":[self.thyroid_surgery],
                          "I131_treatment":[self.I131_treatment],
                          "query_hypothyroid":[self.query_hypothyroid],
                          "query_hyperthyroid":[self.query_hyperthyroid],
                          "lithium":[self.lithium],
                          "goitre":[self.goitre],
                          "tumor":[self.tumor],
                          "hypopituitary":[self.hypopituitary],
                          "psych":[self.psych],
                          "referral_source_SVHC":[self.referral_source_SVHC],
                          "referral_source_SVHD":[self.referral_source_SVHD],
                          "referral_source_SVI":[self.referral_source_SVI],
                          "referral_source_other":[self.referral_source_other],
                          "major_class_encoded":[self.major_class_encoded]}
           
            return input_data
        except Exception as e:
            raise ThyroidException(e, sys)


class HousingPredictor:

    def __init__(self, model_dir: str):
        try:
            logging.info(f"{'>>' * 30} HousingPredictor log started {'<<' * 30} ")

            self.model_dir = model_dir
        except Exception as e:
            raise ThyroidException(e, sys) from e

    def get_latest_model_path(self):
        try:
            logging.info(f"getting latest model path")

            folder_name = list(map(int, os.listdir(self.model_dir)))
            latest_model_dir = os.path.join(self.model_dir, f"{max(folder_name)}")
            file_name = os.listdir(latest_model_dir)[0]
            latest_model_path = os.path.join(latest_model_dir, file_name)

            logging.info(f"latest model path: [ {latest_model_path} ]")
            
            return latest_model_path
        except Exception as e:
            raise ThyroidException(e, sys) from e

    def predict(self, X):
        try:
            logging.info(f"Making Predictions")

            model_path = self.get_latest_model_path()
            model = load_object(file_path=model_path)
            
            logging.info(f"Model objct loaded from path: [ {model_path} ]")

            median_house_value = model.predict(X)
            logging.info(f"Prdictions: [ {median_house_value} ]")
            return median_house_value
        except Exception as e:
            raise ThyroidException(e, sys) from e