from sklearn.model_selection import train_test_split
from ThyroidPrediction.exception import ThyroidException
from ThyroidPrediction.logger import logging
from ThyroidPrediction.entity.config_entity import BaseDataIngestionConfig, BaseDataTransformationConfig, DataTransformationConfig 
from ThyroidPrediction.entity.artifact_entity import BaseDataIngestionArtifact, BaseDataTransformationArtifact, DataIngestionArtifact, DataValidationArtifact,DataTransformationArtifact
from ThyroidPrediction.constant import *
from ThyroidPrediction.util.util import read_yaml_file, save_object, save_numpy_array_data, load_data

from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTENC,RandomOverSampler,KMeansSMOTE
import pandas as pd
import numpy as np
import dill
import sys,os
from cgi import test


#class DataTransformation:
#
#    def __init__(self, data_transformation_config: DataTransformationConfig, data_ingestion_artifact: DataIngestionArtifact, data_validation_artifact: DataValidationArtifact):
#    
#        try:
#            logging.info(f"{'>>' * 30}Data Transformation log started.{'<<' * 30} ")
#    
#            self.data_transformation_config= data_transformation_config
#            self.data_ingestion_artifact = data_ingestion_artifact
#            self.data_validation_artifact = data_validation_artifact
#
#        except Exception as e:
#            raise ThyroidException(e,sys) from e
#   
#
#    def get_data_transformer_object(self) -> ColumnTransformer:
#
#        try:
#            schema_file_path = self.data_validation_artifact.schema_file_path
#
#            dataset_schema = read_yaml_file(file_path=schema_file_path)
#
#            numerical_columns = dataset_schema[NUMERICAL_COLUMN_KEY]
#            categorical_columns = dataset_schema[CATEGORICAL_COLUMN_KEY]
#
#            num_pipeline = Pipeline(steps=[('imputer', SimpleImputer(strategy="median")),
#                                           ('feature_generator', FeatureGenerator(add_bedrooms_per_room=self.data_transformation_config.add_bedroom_per_room,
#                                                                                   columns=numerical_columns)),
#                                           ('scaler', StandardScaler())
#                                           ])
#
#            cat_pipeline = Pipeline(steps=[('impute', SimpleImputer(strategy="most_frequent")),
#                                           ('one_hot_encoder', OneHotEncoder()),
#                                           ('scaler', StandardScaler(with_mean=False))
#                                           ])
#
#            logging.info(f"Categorical columns: {categorical_columns}")
#            logging.info(f"Numerical columns: {numerical_columns}")
#
#
#            preprocessing = ColumnTransformer([('num_pipeline', num_pipeline, numerical_columns),
#                                               ('cat_pipeline', cat_pipeline, categorical_columns),
#                                               ])
#            return preprocessing
#
#        except Exception as e:
#            raise ThyroidException(e,sys) from e   
#
#
#    def initiate_data_transformation(self)->DataTransformationArtifact:
#        try:
#            logging.info(f"Obtaining preprocessing object.")
#
#            preprocessing_obj = self.get_data_transformer_object()
#
#            logging.info(f"Obtaining training and test file path.")
#
#            train_file_path = self.data_ingestion_artifact.train_file_path
#            test_file_path = self.data_ingestion_artifact.test_file_path
#            schema_file_path = self.data_validation_artifact.schema_file_path
#            
#            logging.info(f"Loading training and test data as pandas dataframe.")
#
#            train_df = load_data(file_path=train_file_path, schema_file_path=schema_file_path)
#            test_df = load_data(file_path=test_file_path, schema_file_path=schema_file_path)
#
#            schema = read_yaml_file(file_path=schema_file_path)
#            target_column_name = schema[TARGET_COLUMN_KEY]
#
#
#            logging.info(f"Splitting input and target feature from training and testing dataframe.")
#
#            input_feature_train_df = train_df.drop(columns=[target_column_name],axis=1)
#            target_feature_train_df = train_df[target_column_name]
#
#            input_feature_test_df = test_df.drop(columns=[target_column_name],axis=1)
#            target_feature_test_df = test_df[target_column_name]
#            
#
#            logging.info(f"Applying preprocessing object on training dataframe and testing dataframe")
#
#            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
#            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)
#
#            train_arr = np.c_[ input_feature_train_arr, np.array(target_feature_train_df)]
#            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]
#            
#            transformed_train_dir = self.data_transformation_config.transformed_train_dir
#            transformed_test_dir = self.data_transformation_config.transformed_test_dir
#
#            train_file_name = os.path.basename(train_file_path).replace(".csv",".npz")
#            test_file_name = os.path.basename(test_file_path).replace(".csv",".npz")
#
#            transformed_train_file_path = os.path.join(transformed_train_dir, train_file_name)
#            transformed_test_file_path = os.path.join(transformed_test_dir, test_file_name)
#
#            logging.info(f"Saving transformed training and testing array.")
#            
#            save_numpy_array_data(file_path=transformed_train_file_path,array=train_arr)
#            save_numpy_array_data(file_path=transformed_test_file_path,array=test_arr)
#
#            preprocessing_obj_file_path = self.data_transformation_config.preprocessed_object_file_path
#
#            logging.info(f"Saving preprocessing object.")
#
#            save_object(file_path=preprocessing_obj_file_path,obj=preprocessing_obj)
#
#            data_transformation_artifact = DataTransformationArtifact(is_transformed=True, message="Data transformation successfull.", transformed_train_file_path=transformed_train_file_path,
#                                                                      transformed_test_file_path=transformed_test_file_path, preprocessed_object_file_path=preprocessing_obj_file_path)
#
#            logging.info(f"Data transformationa artifact: {data_transformation_artifact}")
#
#            return data_transformation_artifact
#        except Exception as e:
#            raise ThyroidException(e,sys) from e
#


class DataTransformation:

    def __init__(self, data_transformation_config: BaseDataTransformationConfig, data_ingestion_artifact: DataIngestionArtifact, base_data_ingestion: BaseDataIngestionConfig):
    
        try:
            logging.info(f"{'>>' * 30} Data Transformation log started {'<<' * 30} ")
    
            self.data_transformation_config= data_transformation_config
            self.data_ingestion_artifact = data_ingestion_artifact
            #self.data_validation_artifact = data_validation_artifact
            
            self.processed_data_dir_path = base_data_ingestion.processed_data_dir
            #self.processed_data_dir_path = "ThyroidPrediction/dataset_base/Processed_Dataset"

            self.cleaned_data_dir = base_data_ingestion.cleaned_data_dir
            
        except Exception as e:
            raise ThyroidException(e,sys) from e


    def get_resampled_data(self):
        try:
            logging.info("Importing Train and Test files for Resampling of Data")

            train_file_path = self.data_ingestion_artifact.train_file_path
            test_file_path = self.data_ingestion_artifact.test_file_path

            train = pd.read_csv(train_file_path)
            test = pd.read_csv(test_file_path)

            print("========= DataTransformation V1: train.columns =========")
            print(train.columns)
            print("================"*5)
            

            #################################   RESAMPLING  #################################################
            #X = df_combined_grouped.drop(["Class","Class_encoded",'major_class','major_class_encoded'], axis=1)
            #y = df_combined_grouped["major_class_encoded"]

            #X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2, shuffle=True, stratify= y, random_state=2023 )

            X_train = train.drop(['major_class_encoded'], axis=1)
            y_train = train["major_class_encoded"]

            ###############################################################
            # 
            # Note that we only apply the random oversampler on
            #  the training data and
            #  not on the test data.
            #################################################################


            categorical_features = ['sex','on_thyroxine','query_on_thyroxine','on_antithyroid_medication','sick','pregnant',
                                    'thyroid_surgery','I131_treatment','query_hypothyroid','query_hyperthyroid','lithium',
                                    'goitre','tumor','hypopituitary','psych']

            continuous_features = train.drop(categorical_features, axis=1)

            categorical_features_indices = [train.columns.get_loc(col) for col in categorical_features]

                     
            # Create an instance of RandomOverSampler
            random_over_sampler = RandomOverSampler(random_state=2023)

            

            X_resampled_random, y_resampled_random = random_over_sampler.fit_resample(X_train, y_train)

            X_resampled_random = pd.DataFrame(data = X_resampled_random, columns = X_train.columns)
            y_resampled_random = pd.DataFrame(y_resampled_random, columns= ["major_class_encoded"])

            df_resample_random = pd.concat([X_resampled_random,y_resampled_random], axis=1)


            #class_mapping = {0: 'T toxic',
            #                 1: 'compensated hypothyroid',
            #                 2: 'decreased binding protein',
            #                 3: 'discordant',
            #                 4: 'goitre',
            #                 5: 'hyperthyroid',
            #                 6: 'increased binding protein',
            #                 7: 'negative',
            #                 8: 'overreplacement',
            #                 9: 'primary hypothyroid',
            #                 10: 'replacement therapy',
            #                 11: 'secondary hypothyroid',
            #                 12: 'secondary toxic',
            #                 13: 'sick',
            #                 14: 'underreplacement'}
            #
            #df_resample_random['Class_label'] = df_resample_random['Class_encoded'].replace(class_mapping)


            ## Define the major class conditions
            #conditions = [
            #    df_resample_random['Class_label'].isin(['compensated hypothyroid', 'primary hypothyroid', 'secondary hypothyroid']),
            #    df_resample_random['Class_label'].isin(['hyperthyroid', 'T toxic', 'secondary toxic']),
            #    df_resample_random['Class_label'].isin(['replacement therapy', 'underreplacement', 'overreplacement']),
            #    df_resample_random['Class_label'].isin(['goitre']),
            #    df_resample_random['Class_label'].isin(['increased binding protein', 'decreased binding protein']),
            #    df_resample_random['Class_label'].isin(['sick']),
            #    df_resample_random['Class_label'].isin(['discordant'])]
            #

            ## Define the major class labels
            #class_labels = ['hypothyroid', 'hyperthyroid', 'replacement therapy',
            #                 'goitre', 'binding protein', 'sick', 'discordant']

            ## Add the major class column to the dataframe based on the conditions
            #df_resample_random['major_class'] = np.select(conditions, class_labels, default='negative')
            ##df_resample_random.drop("Class_label", axis=1, inplace=True)
            #
            ##df_combined_grouped = df_resample_random.copy()

            #df_resample_random["major_class_encoded"] = LabelEncoder().fit_transform(df_resample_random["major_class"])


            #resample_data_dir = os.path.join(self.processed_data_dir_path ,"Resampled_Dataset")
            #os.makedirs(resample_data_dir, exist_ok=True)
            #resample_data_file_path = os.path.join(resample_data_dir, "ResampleData_major.csv")

            #df_resample_random.to_csv(resample_data_file_path, index=False)
            
            ##############################################################################################################
            ############################################################################################################

            logging.info("Exporting Resampled Train data.....")

            train_resample_dir = os.path.join(self.data_transformation_config.resampled_data_dir, "train")

            print("========= DataTransformation V2: train_resample_dir =========")
            print(train_resample_dir)
            print("================"*5)


            os.makedirs(train_resample_dir, exist_ok=True)
            train_resample_file_path = os.path.join(train_resample_dir, "train_resample_major.csv")
            df_resample_random.to_csv(train_resample_file_path, index=False)
            
            logging.info(f"Resampled train file exportd: [ {train_file_path} ]")


            test_non_resampled = test
            test_resample_dir = os.path.join(self.data_transformation_config.resampled_data_dir , "test")
            os.makedirs(test_resample_dir, exist_ok=True)
            
            test_non_resample_file_path = os.path.join(test_resample_dir, "test_non_resample_major.csv")
            
            test_non_resampled.to_csv(test_non_resample_file_path, index=False)
            logging.info(f"Non-Resampled test file copied to: [ {test_non_resample_file_path} ]")

            #############################################################################################################
            #train_resample_dir = os.path.join(self.processed_data_dir_path ,"Resampled_Dataset","train_resampled")
            #os.makedirs(train_resample_dir, exist_ok=True)
            #train_resample_file_path = os.path.join(train_resample_dir, "train_resample_major.csv")
            #df_resample_random.to_csv(train_resample_file_path, index=False)


            #test_non_resampled = pd.concat([X_test,y_test], axis=1)
            #test_resample_dir = os.path.join(self.processed_data_dir_path ,"Resampled_Dataset","test_resampled")
            #os.makedirs(test_resample_dir, exist_ok=True)
            #test_non_resample_file_path = os.path.join(test_resample_dir, "test_non_resample_major.csv")
            #test_non_resampled.to_csv(test_non_resample_file_path, index=False)

            #return df_resample_random.head().to_html()
            data_transformation_artifact = BaseDataTransformationArtifact(is_transformed=True, message="Data transformation successfull.",
                                                                          transformed_resampled_train_file_path = train_resample_file_path,
                                                                          transformed_non_resampled_test_file_path= test_non_resample_file_path)

            logging.info(f"Data transformationa artifact: {data_transformation_artifact}")

            return data_transformation_artifact

        except Exception as e:
            raise ThyroidException(e,sys) from e

    def initiate_data_transformation(self):
        try:
            return self.get_resampled_data()
        
        except Exception as e:
            raise ThyroidException(e,sys) from e
    

    def __del__(self):
        logging.info(f"{'>>'*30} Data Transformation log completed {'<<'*30} \n\n")