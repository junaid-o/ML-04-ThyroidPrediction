from sklearn.metrics import ConfusionMatrixDisplay
from ThyroidPrediction.logger import logging
from ThyroidPrediction.exception import ThyroidException
from ThyroidPrediction.entity.config_entity import ModelEvaluationConfig
from ThyroidPrediction.entity.artifact_entity import ClassModelTrainerArtifact, DataIngestionArtifact, \
    DataValidationArtifact, ModelTrainerArtifact, ModelEvaluationArtifact, BaseDataTransformationArtifact
from ThyroidPrediction.constant import *
import numpy as np
import os
import sys
from ThyroidPrediction.util.util import write_yaml_file, read_yaml_file, load_object,load_data
from ThyroidPrediction.entity.model_factory import evaluate_classification_model #, evaluate_regression_model
import matplotlib.pyplot as plt


import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc


class ModelPerformance:

    #def __init__(self, model_evaluation_config: ModelEvaluationConfig, data_ingestion_artifact: DataIngestionArtifact, data_validation_artifact: DataValidationArtifact, model_trainer_artifact: ClassModelTrainerArtifact):
    def __init__(self, model_evaluation_config: ModelEvaluationConfig, data_transformation_artifact: BaseDataTransformationArtifact, data_validation_artifact: DataValidationArtifact, model_trainer_artifact: ClassModelTrainerArtifact):
        try:
            logging.info(f"{'>>' * 30}Model Performance log started.{'<<' * 30} ")
    
            self.model_evaluation_config = model_evaluation_config
            self.model_trainer_artifact = model_trainer_artifact
            #self.data_ingestion_artifact = data_ingestion_artifact
            self.data_transformation_artifact = data_transformation_artifact
            self.data_validation_artifact = data_validation_artifact

            head, _ = os.path.split(self.model_trainer_artifact.trained_model_file_path)
            head, _ = os.path.split(head)

            self.model_performance_dir  = os.path.join(head, "performance")
    
        except Exception as e:
            raise ThyroidException(e, sys) from e
        
    def get_trained_model_and_data(self):
        try:
            trained_model_file_path = self.model_trainer_artifact.trained_model_file_path


            print("========== model Performance v1: trained model file path =============")
            print(trained_model_file_path)
            print("============================================================="*2)

            #import pickle
            #trained_model_object2 = pickle.load(open(trained_model_file_path, "rb"))
            model = load_object(file_path=trained_model_file_path)
            trained_model_object =  model.trained_model_object

            train_file_path = self.data_transformation_artifact.transformed_resampled_train_file_path
            test_file_path = self.data_transformation_artifact.transformed_non_resampled_test_file_path


            print("========== model Performance v2: train and test file path =============")
            print(train_file_path)
            print(test_file_path)
            print("============================================================="*2)


            schema_file_path = self.data_validation_artifact.schema_file_path

            train_dataframe = load_data(file_path=train_file_path,
                                                           schema_file_path=schema_file_path,
                                                           )
            test_dataframe = load_data(file_path=test_file_path,
                                                          schema_file_path=schema_file_path,
                                                          )
            schema_content = read_yaml_file(file_path=schema_file_path)
            target_column_name = schema_content[TARGET_COLUMN_KEY]

            # target_column
            logging.info(f"Converting target column into numpy array.")

            print("========= Model Performance ==========="*3)
            print("Converting target column into numpy array.")
            print("======================================"*3)

            train_target_arr = np.array(train_dataframe[target_column_name])
            test_target_arr = np.array(test_dataframe[target_column_name])
            logging.info(f"Conversion completed target column into numpy array.")

            # dropping target column from the dataframe
            logging.info(f"Dropping target column from the dataframe.")


            print("========= Model Performance ==========="*3)
            print("Dropping target column from the dataframe.")
            print("======================================"*3)


            train_dataframe.drop(target_column_name, axis=1, inplace=True)
            test_dataframe.drop(target_column_name, axis=1, inplace=True)
            logging.info(f"Dropping target column from the dataframe completed.")
            return train_dataframe, test_dataframe, train_target_arr, test_target_arr, trained_model_object
        except Exception as e:
            raise ThyroidException(e, sys) from e
        
                
    def get_confusion_metrix(self):
        try:

            train_dataframe, test_dataframe, train_target_arr, test_target_arr, trained_model_object = self.get_trained_model_and_data()
            X_test =  test_dataframe
            y_test = test_target_arr

            #################################################
            fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(6, 6), squeeze=True)
            fig.suptitle('Confusion Metrix: Oversampled Data with Major Classes', fontsize=22, fontweight='bold', y=0.98)
            # Uncomment the axes line if increasing column in above line of code
            try:
                # flattening is required only when nrows or ncols is mor than one
                axes = axes.flatten()
            except:
                pass
            i = 0
            ###################################################

            #for name, clf in {'SVM':svm_model, "AdaBoost":AdaBoost_model, "GBDT":GBDT_model,"KNN":knn_model,"LogReg":lr_model,"RF":RF_model}.items():

            model = {"RF": trained_model_object}
            print(trained_model_object)
            
            class_names = ['binding protein', 'discordant', 'goitre', 'hyperthyroid', 'hypothyroid', 'negative', 'replacement therapy','sick']

            for name, clf in model.items():
                #print("\nFor ",name)
                
                try:
                    ax = axes[i]
                except:
                    ax= axes
                
                confusionMetrix = ConfusionMatrixDisplay.from_estimator(clf, X=X_test, y= y_test, display_labels= class_names, xticks_rotation='vertical', ax=ax, colorbar=False)
                ax.set_title(name)
                
                i +=1

            plt.tight_layout()    

            ####################    EXPORTING CONFUSION METRIX  #############
            #os.makedirs("Results/Results_Classification_resampled/ConfusionMetrix_resampled", exist_ok=True)
            #plt.savefig(f'Results/Results_Classification_resampled/ConfusionMetrix_resampled/ConfusionMetrix_resampled.svg',format='svg',dpi=600)

            os.makedirs(self.model_performance_dir,exist_ok=True)
            plt.savefig(os.path.join(self.model_performance_dir, "confusion_metrix.svg"), format="svg", bbox_inches='tight', dpi=300)

            #######################################################################
            #plt.show()

        except Exception as e:
            raise ThyroidException(e, sys) from e


    def get_roc_auc_curve(self):
        try:
            train_dataframe, X_test, train_target_arr, y_test, RF_model = self.get_trained_model_and_data()
            
            # Compute the predicted probabilities for each class
            y_score = RF_model.predict_proba(X_test)

            # Compute the ROC curve and ROC AUC for each class
            fpr = dict()
            tpr = dict()
            roc_auc = dict()
            for i in range(8):
                fpr[i], tpr[i], _ = roc_curve(y_test == i, y_score[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])

            # Plot the ROC curves for each class
            plt.figure()
            lw = 2
            colors = ['blue',"purple", 'orange',"cyan","gray",'green','red',"magenta"]
            for i, color in zip(range(8), colors):
                plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                        label='ROC curve of class {0} (AUC = {1:0.2f})'
                        ''.format(i, roc_auc[i]))
                
            plt.plot([0, 1], [0, 1], 'k--', lw=lw)
            plt.xlim([-0.05, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver operating characteristic for multiclass')
            plt.legend(loc="lower right")


            #######################################

            #os.makedirs(f'Results/Results_Classification_resampled/ROC_AUC_OVR_Curve_resampled', exist_ok=True)
            #plt.savefig(f'Results/Results_Classification_resampled/ROC_AUC_OVR_Curve_resampled/ROC_AUC_OVR_Curve_resampled.svg',format='svg',dpi=600)

            roc_auc_curve_file_path = os.path.join(self.model_performance_dir,"roc_auc.svg")
            plt.savefig(roc_auc_curve_file_path, format="svg", bbox_inches='tight', dpi=300)
            ##########################
            #plt.show()

        except Exception as e:
            raise ThyroidException(e, sys) from e
        

    def get_learning_curve_complexity_analysis(self):
        try:
            pass
        except Exception as e:
            raise ThyroidException(e, sys) from e
        
        
    def initiate_performance_evaluation(self):
        try:
            self.get_trained_model_and_data()
            self.get_confusion_metrix()
            return self.get_roc_auc_curve()
        except Exception as e:
            raise ThyroidException(e, sys) from e