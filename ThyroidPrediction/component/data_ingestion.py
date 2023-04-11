from ThyroidPrediction.entity.config_entity import DataIngestionConfig, BaseDataIngestionConfig
from ThyroidPrediction.exception import ThyroidException
from ThyroidPrediction.logger import logging
from ThyroidPrediction.entity.artifact_entity import DataIngestionArtifact, BaseDataIngestionArtifact
import os, sys

import tarfile
from six.moves import urllib
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from sklearn.preprocessing import LabelEncoder

import os
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTENC,RandomOverSampler,KMeansSMOTE


class DataIngestion:

    def __init__(self, data_ingestion_config:BaseDataIngestionConfig):
        try:
            logging.info(f"{'='*20} DATA INGESTION LOG STARTED.{'='*20}")
            
            #self.data_ingestion_config = data_ingestion_config
            self.base_data_ingestion_config = data_ingestion_config
            self.base_dataset_path = r"ThyroidPrediction\dataset_base"
    
        except Exception as e:
            raise ThyroidException(e, sys) from e


    #def download_housing_data(self) -> str:
    #    try:
    #        # extract remote url for downloading dataset
    #        download_url = self.data_ingestion_config.dataset_download_url
    #
    #        # folder location to download file
    #        tgz_donwload_dir = self.data_ingestion_config.tgz_download_dir
    #        
    #        #######################################
    #        if os.path.exists(tgz_donwload_dir):
    #            os.remove(tgz_donwload_dir)
    #
    #        os.makedirs(tgz_donwload_dir, exist_ok=True)
    #        
    #        #####################
    #
    #        housing_file_name = os.path.basename(download_url)
    #        tgz_file_path = os.path.join(tgz_donwload_dir, housing_file_name)
    #
    #        logging.info(f"Downloading file from {download_url} into [{tgz_file_path}]")
    #        
    #        urllib.request.urlretrieve(download_url,tgz_file_path)
    #        
    #        logging.info(f"File: [{tgz_file_path}] has been downloaded sucessfully")
    #        
    #        return tgz_file_path
    #
    #    except Exception as e:
    #        raise ThyroidException(e,sys) from e        

    #def extract_tgz_file(self, tgz_file_path: str):
    #    try:
    #        raw_data_dir = self.data_ingestion_config.raw_data_dir
    #
    #        #######################################
    #        if os.path.exists(raw_data_dir):
    #            os.remove(raw_data_dir)
    #
    #        os.makedirs(raw_data_dir, exist_ok=True)
    #
    #        logging.info(f"Extracting tgz file: [{tgz_file_path}] into dir: [{tgz_file_path}]")
    #        with tarfile.open(tgz_file_path) as housing_tgz_file_obj:
    #            housing_tgz_file_obj.extractall(path= raw_data_dir)
    #
    #        logging.info(f"Extraction completed for tgz file: [{tgz_file_path}]")
    #
    #    except Exception as e:
    #        raise ThyroidException(e, sys) from e

    


    #def split_data_as_train_test(self) -> DataIngestionArtifact:
    #
    #   try:
    #       raw_data_dir = self.data_ingestion_config.raw_data_dir
    #       file_name = os.listdir(raw_data_dir)[0]
    #       housing_file_path = os.path.join(raw_data_dir, file_name)
    #       
    #       
    #       housing_file_path = self.file_path
    #       logging.info(f"Reading csv file:[{housing_file_path}]")
    #
    #       thyroid_data_frame = pd.read_csv(housing_file_path)
    #
    #    
    #       logging.info(f"Splitting Data into Train-Test")
    #       
    #       start_train_set = None
    #       start_test_set = None
    #       
    #       split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    #
    #       X = thyroid_data_frame.drop(['major_class', 'major_class_endcoded'],axis=1)
    #       y = thyroid_data_frame.drop(['major_class'],axis=1)['major_class_endcoded']
    #       
    #       for train_index, test_index in split.split(X, y):
    #           
    #           start_train_set = thyroid_data_frame.loc[train_index].drop(["major_class"], axis=1)
    #           start_test_set = thyroid_data_frame.loc[test_index].drop(["major_class"], axis=1)
    #
    #       
    #       
    #       train_file_path = os.path.join(self.data_ingestion_config.ingested_train_dir, file_name)
    #       test_file_path = os.path.join(self.data_ingestion_config.ingested_test_dir, file_name)
    #       if start_train_set is not None:
    #
    #           os.makedirs(self.data_ingestion_config.ingested_train_dir, exist_ok=True)
    #           
    #           logging.info(f"Exporting training data to file:[{train_file_path}]")
    #           
    #           start_train_set.to_csv(train_file_path, index=False)
    #
    #       if start_test_set is not None:
    #
    #           os.makedirs(self.data_ingestion_config.ingested_test_dir, exist_ok=True)
    #           logging.info(f"Exporting test data to file:[{test_file_path}]")
    #           start_test_set.to_csv(test_file_path, index=False)
    #
    #       
    #       data_ingestion_artifact = DataIngestionArtifact(train_file_path=train_file_path, test_file_path=test_file_path,
    #                                                       is_ingested=True, message= f"DataIngestion Completed Successfully")
    #       
    #       logging.info(f"Data Ingestion Artifact:[{data_ingestion_artifact}]")
    #       return data_ingestion_artifact
    #
    #   except Exception as e:
    #       raise ThyroidException(e, sys) from e
        
    def get_base_data(self):
        try:
            pd.set_option('display.max_columns', None)

            # Path to the top-level directory
            #dir_path = "ThyroidPrediction/dataset_base/Raw_Dataset"

            dataset_base = self.base_dataset_path
            raw_data_dir = self.base_data_ingestion_config.raw_data_dir
            print('=='*20)
            print(raw_data_dir)
            print('=='*20)
            raw_data_dir_path = os.path.join(dataset_base, raw_data_dir)
            print('=='*20)
            print(raw_data_dir_path)
            print('=='*20)            
            
            os.makedirs(raw_data_dir, exist_ok=True)



            csv_files = []

            columns_list = ['age', 'sex','on_thyroxine','query_on_thyroxine','on_antithyroid_medication', 'sick', 'pregnant', 'thyroid_surgery',
                            'I131_treatment', 'query_hypothyroid', 'query_hyperthyroid', 'lithium', 'goitre', 'tumor', 'hypopituitary','psych', 'TSH_measured',
                            'TSH','T3_measured','T3', 'TT4_measured', 'TT4', 'T4U_measured','T4U', 'FTI_measured', 'FTI', 'TBG_measured', 'TBG', 'referral_source',
                            'Class']

            logging.info(f"{'='*20} READING BASE DATASET {'='*20} \n\n Walking Through All Dirs In [ {raw_data_dir_path} ] for all .data and .test files")
            
            # Traverse the directory structure recursively
            for root, dirs, files in os.walk(raw_data_dir_path):
                for file in files:
                    #print(files)
                    # Check if the file is a CSV file
                    if file.endswith(".data") or file.endswith(".test"):
                        file_path = os.path.join(root, file)
                        #print(file_path)
                        
                        # Read the CSV file into a pandas DataFrame
                        df = pd.read_csv(file_path, header=None)
                        df.columns = columns_list
                        print("Unique value per file",file_path,df['hypopituitary'].unique())
                        #print(file_path, df.columns)
                        csv_files.append(df)
                        # Do something with the DataFrame

            print("Number of csv files",len(csv_files))
            
            df_combined = pd.DataFrame()
            for i in range(len(csv_files)):
                #print(len(csv_files[i].columns))
                #df_name = f"df_{i}"
                df_next = csv_files[i]
                df_combined = pd.concat([df_combined, df_next], axis=0)

                print("Unique value per file", file_path, df_combined['hypopituitary'].unique())


            df_combined.columns = columns_list
            
            #print("Hypopituitory unique values before cleaning",df_combined['hypopituitary'].unique())
            ###############################################################################################

            df_combined.drop_duplicates(inplace=True)
            df_combined["Class"].replace(to_replace= r"[.|0-9]",value="", regex=True, inplace=True)
            df_combined = df_combined.drop(['TSH_measured','T3_measured','TT4_measured','T4U_measured','FTI_measured','TBG_measured'], axis=1)



            print("Missing Value('?') Count After Replaing Thm ith NaN Revealing:\n")
            for column in df_combined.columns:
                missing_count = df_combined[column][df_combined[column] == "?"].count()
                
                if missing_count != 0:
                    print(column,missing_count)
                    df_combined[column] = df_combined[column].replace("?", np.nan)
            print('====='*28)

            df_combined = df_combined.drop(["TBG"], axis=1)
            


            columns_float = ['age', 'TSH', 'T3', 'TT4','T4U', 'FTI']

            for column in columns_float:
                df_combined[column] = df_combined[column].astype(float)

            #print("Hypopituitory Uniqu valus",df_combined["hypopituitary"].unique())

            #processed_dataset_dir = os.path.join(self.base_dataset_path,"Processed_Dataset","Cleaned_Data")
            #os.makedirs(processed_dataset_dir, exist_ok=True)

            processed_data_dir = os.path.join(self.base_dataset_path,self.base_data_ingestion_config.processed_data_dir,"Cleaned_Data")
            os.makedirs(processed_data_dir, exist_ok=True)
            
            #processed_data_file_path = os.path.join(processed_dataset_dir,"df_combined_cleaned.csv")
            #df_combined.to_csv(processed_data_file_path,index=False)

            processed_data_file_path = os.path.join(processed_data_dir,"df_combined_cleaned.csv")
            df_combined.to_csv(processed_data_file_path,index=False)

            ########################################    MISSING vALUE IMPUATION    ##########################################################

            #df_combined['sex'] = SimpleImputer(missing_values=np.nan, strategy="most_frequent").fit_transform(df_combined[["sex"]].values)
            #df_combined['age'] = SimpleImputer(missing_values=np.nan, strategy="most_frequent").fit_transform(df_combined[["age"]].values)

            #df_combined['TSH'] = SimpleImputer(missing_values=np.nan, strategy="median").fit_transform(df_combined[["TSH"]].values)
            #df_combined['T3'] = SimpleImputer(missing_values=np.nan, strategy="median").fit_transform(df_combined[["T3"]].values)
            #df_combined['TT4'] = SimpleImputer(missing_values=np.nan, strategy="median").fit_transform(df_combined[["TT4"]].values)
            #df_combined['T4U'] = SimpleImputer(missing_values=np.nan, strategy="median").fit_transform(df_combined[["T4U"]].values)
            #df_combined['FTI'] = SimpleImputer(missing_values=np.nan, strategy="median").fit_transform(df_combined[["FTI"]].values)


            ####################### HANDLINNG CATEGOICAL VARIABLES ###########################
            #df_combined_plot = df_combined.copy()
            #
            #columns_list = df_combined.columns.to_list()
            #
            #for feature in columns_list:
            #    if len(df_combined[feature].unique()) <= 3:

            #        print(feature, df_combined[feature].unique())

            #        value1 = df_combined[feature].unique()[0]
            #        value2 = df_combined[feature].unique()[1]
            #        df_combined[feature] = df_combined[feature].map({f'{value1}':0, f'{value2}':1})
            #        
            #        print(feature, df_combined[feature].unique())

            #df_combined = pd.get_dummies(data  = df_combined, columns=['referral_source'], drop_first=True)
            #df_combined["Class_encoded"] = LabelEncoder().fit_transform(df_combined["Class"])


            ############################### OUTLIERS HANDLING ###############################

            #def outliers_fence(col):
            #    Q1 = df_combined[col].quantile(q=0.25)
            #    Q3 = df_combined[col].quantile(q=0.75)
            #    IQR = Q3 - Q1

            #    lower_fence = Q1 - 1.5*IQR
            #    upper_fence = Q3 + 1.5*IQR
            #    return lower_fence, upper_fence

            #lower_fence1, upper_fence1 = outliers_fence(col='TSH')
            #lower_fence2, upper_fence2 = outliers_fence(col='T3')
            #lower_fence3, upper_fence3 = outliers_fence(col='TT4')
            #lower_fence4, upper_fence4 = outliers_fence(col='T4U')
            #lower_fence5, upper_fence5 = outliers_fence(col='FTI')

            ## Winsorize the data just replace outliers with corresponding fence

            #df_combined['TSH'] = np.where(df_combined["TSH"] < lower_fence1, lower_fence1, df_combined["TSH"])
            #df_combined["TSH"] = np.where(df_combined["TSH"] > upper_fence1, upper_fence1, df_combined["TSH"])

            #df_combined['T3'] = np.where(df_combined["T3"] < lower_fence2, lower_fence2, df_combined["T3"])
            #df_combined["T3"] = np.where(df_combined["T3"] > upper_fence2, upper_fence2, df_combined["T3"])

            #df_combined['TT4'] = np.where(df_combined["TT4"] < lower_fence3, lower_fence3, df_combined["TT4"])
            #df_combined["TT4"] = np.where(df_combined["TT4"] > upper_fence3, upper_fence3, df_combined["TT4"])

            #df_combined['T4U'] = np.where(df_combined["T4U"] < lower_fence4, lower_fence4, df_combined["T4U"])
            #df_combined["T4U"] = np.where(df_combined["T4U"] > upper_fence4, upper_fence4, df_combined["T4U"])

            #df_combined['FTI'] = np.where(df_combined["FTI"] < lower_fence5, lower_fence5, df_combined["FTI"])
            #df_combined["FTI"] = np.where(df_combined["FTI"] > upper_fence5, upper_fence5, df_combined["FTI"])

            ###################################### MAJOR CLASS CREATION   ############################################################

            #df_combined_class_labels = df_combined.copy()
            #df_combined_class_labels["Class_label"] = df_combined_plot['Class']

            #df = df_combined_class_labels
            ## Define the major class conditions
            #conditions = [
            #    df['Class_label'].isin(['compensated hypothyroid', 'primary hypothyroid', 'secondary hypothyroid']),
            #    df['Class_label'].isin(['hyperthyroid', 'T toxic', 'secondary toxic']),
            #    df['Class_label'].isin(['replacement therapy', 'underreplacement', 'overreplacement']),
            #    df['Class_label'].isin(['goitre']),
            #    df['Class_label'].isin(['increased binding protein', 'decreased binding protein']),
            #    df['Class_label'].isin(['sick']),
            #    df['Class_label'].isin(['discordant'])
            #]

            ## Define the major class labels
            #class_labels = [
            #    'hypothyroid',
            #    'hyperthyroid',
            #    'replacement therapy',
            #    'goitre',
            #    'binding protein',
            #    'sick',
            #    'discordant'
            #]

            ## Add the major class column to the dataframe based on the conditions
            #df['major_class'] = np.select(conditions, class_labels, default='negative')
            #df.drop("Class_label", axis=1, inplace=True)
            #df_combined_grouped = df.copy()

            ##################################   RESAMPLING  #################################################
            #X = df_combined.drop("Class", axis=1)
            #y = df_combined["Class"]

            #categorical_features = ['sex','on_thyroxine','query_on_thyroxine','on_antithyroid_medication','sick','pregnant',
            #                        'thyroid_surgery','I131_treatment','query_hypothyroid','query_hyperthyroid','lithium',
            #                        'goitre','tumor','hypopituitary','psych']

            #continuous_features = df_combined.drop(categorical_features, axis=1)

            #categorical_features_indices = [df_combined.columns.get_loc(col) for col in categorical_features]

            ## Create an instance of SMOTENC oversampler

            #smote_nc = SMOTENC(categorical_features = categorical_features_indices, random_state=2023)


            ## Create an instance of RandomOverSampler
            #random_over_sampler = RandomOverSampler(random_state=2023)

            ## Create an instance of KMeansSMOTE
            #kmeans_smote = KMeansSMOTE(random_state=2023)

            #
            #X_resampled_random, y_resampled_random = random_over_sampler.fit_resample(X, y)

            #X_resampled_random = pd.DataFrame(data = X_resampled_random, columns = X.columns)
            #y_resampled_random = pd.DataFrame(y_resampled_random, columns= ["Class"])

            #df_resample_random = pd.concat([X_resampled_random,y_resampled_random], axis=1)

            #resample_data_dir = os.path.join(self.base_dataset_path,"Resampled_Dataset")
            #os.makedirs(resample_data_dir, exist_ok=True)
            #resample_data_file_path = os.path.join(resample_data_dir, "ResampleData_major.csv")

            #df_resample_random.to_csv(resample_data_file_path, index=False)
            #

            ##df_combined_grouped.to_csv("dataset/combined_processed_grouped.csv",index=False)


            ##return df_combined.head()

            return df_combined.head(10).to_html()
        
        except Exception as e:
            raise ThyroidException(e, sys)

    


    #def split_data(self):
    #    try:
    #        
    #        file_path = os.path.join(self.base_dataset_path,"Resampled_Dataset","ResampleData_major.csv")
    #        
    #        logging.info(f"Reading CSV file for Base dataset [{file_path}]")
    #        df = pd.read_csv(file_path)
    #        
    #        logging.info(f"Encoding Major_Class Categories")

    #        df["major_class_endcoded"] = LabelEncoder().fit_transform(df["major_class"])
    #        
    #        X = df.drop(['major_class', 'major_class_endcoded'],axis=1)
    #        y = df.drop(['major_class'],axis=1)['major_class_endcoded']
    #        
    #        logging.info(f"Spliting Dataset into train and test set")

    #        X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2, shuffle=True, stratify= y, random_state=2023 )
    #        
    #        start_train_set = None
    #        start_test_set = None


    #        start_train_set = pd.concat([X_train,y_train], axis=1)
    #        start_test_set = pd.concat([X_test,y_test], axis=1)

    #        print("done!")
    #        print(start_train_set.columns)
    #        

    #        train_file_dir = os.path.join(self.base_dataset_path, "train_set")
    #        test_file_dir = os.path.join(self.base_dataset_path, "test_set")
    #        
    #        print(train_file_dir)


    #        if start_train_set is not None:
    #    
    #            os.makedirs(train_file_dir, exist_ok=True)
    #            
    #            logging.info(f"Exporting training data to file:[{train_file_dir}]")
    #            
    #            train_file_path = os.path.join(train_file_dir, "train_set.csv")
    #            start_train_set.to_csv(train_file_path, index=False)
    #    
    #        if start_test_set is not None:
    #    
    #            os.makedirs(test_file_dir, exist_ok=True)

    #            logging.info(f"Exporting test data to file:[{test_file_dir}]")
    #            test_file_path = os.path.join(test_file_dir, "test_set.csv")
    #            
    #            start_test_set.to_csv(test_file_path, index=False)            
    #        return start_train_set.head().to_html()
    #    
    #    except Exception as e:                       
    #        raise ThyroidException(e, sys)
        
    


    def initiate_data_ingestion(self):
       try:
           #tgz_file_path = self.download_housing_data()
           #self.extract_tgz_file(tgz_file_path=tgz_file_path)
           
           #return self.split_data()
           return self.get_base_data()

       except Exception as e:
           raise ThyroidException(e, sys) from e
        
    
    def __del__(self):

        logging.info(f"{'='*20}Ingestion log completed {'='*20}\n\n")
    