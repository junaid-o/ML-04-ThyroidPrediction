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

    def __init__(self, data_transformation_config: BaseDataTransformationConfig, data_ingestion_artifact: BaseDataIngestionArtifact, base_data_ingestion: BaseDataIngestionConfig):
    
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


    def get_data_transformer_object(self):

        try:
            
            processesd_data_dir_path = self.processed_data_dir_path
            cleaned_data_file_path = os.path.join(self.cleaned_data_dir,"df_combined_cleaned.csv")
            
            print("===== Cleand Data File Path ======"*20)
            print("\n\n",cleaned_data_file_path)
            
            df_combined = pd.read_csv(cleaned_data_file_path)

            #######################################    MISSING VALUE IMPUATION    ##########################################################

            df_combined['sex'] = SimpleImputer(missing_values=np.nan, strategy="most_frequent").fit_transform(df_combined[["sex"]].values)
            df_combined['age'] = SimpleImputer(missing_values=np.nan, strategy="most_frequent").fit_transform(df_combined[["age"]].values)

            df_combined['TSH'] = SimpleImputer(missing_values=np.nan, strategy="median").fit_transform(df_combined[["TSH"]].values)
            df_combined['T3'] = SimpleImputer(missing_values=np.nan, strategy="median").fit_transform(df_combined[["T3"]].values)
            df_combined['TT4'] = SimpleImputer(missing_values=np.nan, strategy="median").fit_transform(df_combined[["TT4"]].values)
            df_combined['T4U'] = SimpleImputer(missing_values=np.nan, strategy="median").fit_transform(df_combined[["T4U"]].values)
            df_combined['FTI'] = SimpleImputer(missing_values=np.nan, strategy="median").fit_transform(df_combined[["FTI"]].values)


            ###################### HANDLINNG CATEGOICAL VARIABLES ###########################

            df_combined_plot = df_combined.copy()
            
            columns_list = df_combined.columns.to_list()
            
            for feature in columns_list:
                if len(df_combined[feature].unique()) <= 3:

                    #print(df_combined[feature].unique() )
                    value1 = df_combined[feature].unique()[0]
                    value2 = df_combined[feature].unique()[1]
                    df_combined[feature] = df_combined[feature].map({f'{value1}':0,f'{value2}':1})
                    print(feature, df_combined[feature].unique())

            df_combined = pd.get_dummies(data  = df_combined, columns=['referral_source'], drop_first=True)
            df_combined["Class_encoded"] = LabelEncoder().fit_transform(df_combined["Class"])


            return df_combined, df_combined_plot
        except Exception as e:
            raise ThyroidException(e,sys) from e
        

    def outliers_handling(self):
        try:

            ############################## OUTLIERS HANDLING ###############################

            df_combined, df_combined_plot = self.get_data_transformer_object()
            
            def outliers_fence(col):
                Q1 = df_combined[col].quantile(q=0.25)
                Q3 = df_combined[col].quantile(q=0.75)
                IQR = Q3 - Q1

                lower_fence = Q1 - 1.5*IQR
                upper_fence = Q3 + 1.5*IQR
                return lower_fence, upper_fence

            lower_fence1, upper_fence1 = outliers_fence(col='TSH')
            lower_fence2, upper_fence2 = outliers_fence(col='T3')
            lower_fence3, upper_fence3 = outliers_fence(col='TT4')
            lower_fence4, upper_fence4 = outliers_fence(col='T4U')
            lower_fence5, upper_fence5 = outliers_fence(col='FTI')

            # Winsorize the data just replace outliers with corresponding fence

            df_combined['TSH'] = np.where(df_combined["TSH"] < lower_fence1, lower_fence1, df_combined["TSH"])
            df_combined["TSH"] = np.where(df_combined["TSH"] > upper_fence1, upper_fence1, df_combined["TSH"])

            df_combined['T3'] = np.where(df_combined["T3"] < lower_fence2, lower_fence2, df_combined["T3"])
            df_combined["T3"] = np.where(df_combined["T3"] > upper_fence2, upper_fence2, df_combined["T3"])

            df_combined['TT4'] = np.where(df_combined["TT4"] < lower_fence3, lower_fence3, df_combined["TT4"])
            df_combined["TT4"] = np.where(df_combined["TT4"] > upper_fence3, upper_fence3, df_combined["TT4"])

            df_combined['T4U'] = np.where(df_combined["T4U"] < lower_fence4, lower_fence4, df_combined["T4U"])
            df_combined["T4U"] = np.where(df_combined["T4U"] > upper_fence4, upper_fence4, df_combined["T4U"])

            df_combined['FTI'] = np.where(df_combined["FTI"] < lower_fence5, lower_fence5, df_combined["FTI"])
            df_combined["FTI"] = np.where(df_combined["FTI"] > upper_fence5, upper_fence5, df_combined["FTI"])
        
            return df_combined, df_combined_plot
                
        except Exception as e:
            raise ThyroidException(e,sys) from e   

    def get_target_by_major_class(self):
        try:

            ##################################### MAJOR CLASS CREATION   ############################################################
            
            df_combined_class_labels, df_combined_plot = self.outliers_handling()
            df_combined_class_labels["Class_label"] = df_combined_plot['Class']

            df = df_combined_class_labels
            # Define the major class conditions
            conditions = [
                df['Class_label'].isin(['compensated hypothyroid', 'primary hypothyroid', 'secondary hypothyroid']),
                df['Class_label'].isin(['hyperthyroid', 'T toxic', 'secondary toxic']),
                df['Class_label'].isin(['replacement therapy', 'underreplacement', 'overreplacement']),
                df['Class_label'].isin(['goitre']),
                df['Class_label'].isin(['increased binding protein', 'decreased binding protein']),
                df['Class_label'].isin(['sick']),
                df['Class_label'].isin(['discordant'])
            ]

            # Define the major class labels
            class_labels = [
                'hypothyroid',
                'hyperthyroid',
                'replacement therapy',
                'goitre',
                'binding protein',
                'sick',
                'discordant'
            ]

            # Add the major class column to the dataframe based on the conditions
            df['major_class'] = np.select(conditions, class_labels, default='negative')
            df.drop("Class_label", axis=1, inplace=True)
            
            df_combined_grouped = df.copy()
            
            df_combined_grouped["major_class_encoded"] = LabelEncoder().fit_transform(df_combined_grouped["major_class"])
            transformed_data_dir = os.path.join(self.data_transformation_config.transformed_data_dir)
            #transformed_data_dir = os.path.join(self.processed_data_dir_path,"Transformed_Data")
            os.makedirs(transformed_data_dir, exist_ok=True)

            transformed_data_file_path = os.path.join(transformed_data_dir,"df_transformed_major_class.csv")
            df_combined_grouped.to_csv(transformed_data_file_path, index=False)

            return df_combined_grouped
        
        except Exception as e:
            raise ThyroidException(e,sys) from e


    def get_resampled_data(self):
        try:
            df_combined_grouped = self.get_target_by_major_class()

            #################################   RESAMPLING  #################################################
            X = df_combined_grouped.drop(["Class","Class_encoded",'major_class','major_class_encoded'], axis=1)
            y = df_combined_grouped["major_class_encoded"]

            X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2, shuffle=True, stratify= y, random_state=2023 )

            ###############################################################
            # 
            # Note that we only apply the random oversampler on
            #  the training data and
            #  not on the test data.
            #################################################################


            categorical_features = ['sex','on_thyroxine','query_on_thyroxine','on_antithyroid_medication','sick','pregnant',
                                    'thyroid_surgery','I131_treatment','query_hypothyroid','query_hyperthyroid','lithium',
                                    'goitre','tumor','hypopituitary','psych']

            continuous_features = df_combined_grouped.drop(categorical_features, axis=1)

            categorical_features_indices = [df_combined_grouped.columns.get_loc(col) for col in categorical_features]

                     
            # Create an instance of RandomOverSampler
            random_over_sampler = RandomOverSampler(random_state=2023)

            

            X_resampled_random, y_resampled_random = random_over_sampler.fit_resample(X_train, y_train)

            X_resampled_random = pd.DataFrame(data = X_resampled_random, columns = X.columns)
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
            train_resample_dir = os.path.join(self.data_transformation_config.train_resampled_dir)
            os.makedirs(train_resample_dir, exist_ok=True)
            train_resample_file_path = os.path.join(train_resample_dir, "train_resample_major.csv")
            df_resample_random.to_csv(train_resample_file_path, index=False)


            test_non_resampled = pd.concat([X_test,y_test], axis=1)
            test_resample_dir = os.path.join(self.data_transformation_config.test_non_resampled_dir)
            os.makedirs(test_resample_dir, exist_ok=True)
            test_non_resample_file_path = os.path.join(test_resample_dir, "test_non_resample_major.csv")
            test_non_resampled.to_csv(test_non_resample_file_path, index=False)

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
        logging.info(f"{'>>'*30}Data Transformation log completed.{'<<'*30} \n\n")