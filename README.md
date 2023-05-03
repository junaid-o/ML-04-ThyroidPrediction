# <h1 style="color: red; text-align:center; font-weight: bold">Multiclass Classification of Diseased State of Thyroid</h1>


## **Background:**

Thyroid disease is a common cause of medical diagnosis and prediction, with an onset that is difficult to forecast in medical research. The thyroid gland is one of our body's most vital organs. Thyroid hormone releases are responsible for metabolic regulation. Hyperthyroidism and hypothyroidism are one of the two common diseases of the thyroid that releases thyroid hormones in regulating the rate of body's metabolism.


## **Problem Statement**
The main goal is to predict the estimated risk on a patient's chance of obtaining thyroid disease or not.

## **Dataset:**
<center>
<a href="https://archive.ics.uci.edu/ml/datasets/thyroid+disease"><button data-md-button>Dataset</button></a>
</center>


## **Deployment**

<center>
<a href="/"><button data-md-button>Deployment</button></a> 
</center>




## **Tools & Techniques**

*   `Data versioning` using time stamp
*   `Code versioning` using Git
*   `Modular coding` with separate files for data ingestion, transformation, validation, training, evaluation, performance monitoring, model pusher, model configuration etc
*   `CI / CD` using GitHub Actions
*   `S3 Bucket` for storage of dataset.
*   `Docker` file created
*   Custome `logger`
*   Custom `Exception Handler`
*   `Package building` using setuptools


## **Result**

*   Model trained on original data performed better than model trained on resampled data.
*   Best performer in both coditions is `RandomForestClassifier`
    *   Scores Achieved:
        

        | Metric                | Train | Test  |
        |-----------------------|-------|-------|
        | F1 weighted           | 0.732 | 0.732 |        
        | ROC AUC OVR Weighted  | 0.732 | 0.732 |
        | Balanced Accuracy     | 0.732 | 0.732 |
        | Log loss              | 0.83  | 0.832 |
        | Precission            | 0.83  | 0.832 |


## **Evaluation Metrices**
*   F1 weighted score
*   ROC AUC (One vs Rest) Weighted
*   Balanced Accuracy
*   Log loss
*   ConfusionMetrics
*   Learning Curve
*   Complexity and Scalability


## **Approach**

*   Data collection, cleaning, missing value handling, outlier handling, Data Profiling, exploration.

*   Feature selection, reducing number of class labels from 14 to 7 by grouping the labels based on medical conditions.

*   Random Over Sampling of data but ommitted due to better performance of orgianl data on test dataset

*   Tested Machine Learning algorithms, including `RandomForestClassifier`, `KNeighborsClassifier`, `AdaBoost` and `GradientBoostingClassifir`.

*   Once the training is completd, model is passed through evaluation phase where it has to pass through set of logical conditons. Only the models above the threshold value of evaluation metrics are consider as accepted model and pushed for integration with FlaskApp

```
f1_logic = (train_f1 >= 0.738) and abs(train_f1 - test_f1) <= 0.009
roc_auc_logic = (roc_auc_ovr_weighted_train >= 0.89) and abs(roc_auc_ovr_weighted_train - roc_auc_ovr_weighted_test) <= 0.02
model_accuracy_logic = (train_balanced_accuracy_score >= base_accuracy) and diff_test_train_acc <= 0.04
loss_logic = (loss_train <= 1.013) and abs(loss_train - loss_test) <= 0.04


if f1_logic and roc_auc_logic and model_accuracy_logic and loss_logic:
        -------
        -------
        ------
```

*   Profiling Report, EDA Report and Evaluation Report generation

## **API and Web UI**

*   API exposed via `Flask-Web-App`
*   Dashboard displays score cards for `F1_weighted`, `ROC_AUC_OVR_Weighted`, `Balanced Accuracy`, `Log_loss`
*   Web dashboard allow you:
    *   View all reports for the deployed model:
        *   Profiling Report
        *   EDA Report
        *   Model Performance Report
    
    *   View, modify model configuration and save changes
    *   View and download models accepted above a threshold value of evaluation metrics
    *   Trigger model training
    *   View Logs
    *   View all the artifacts
    *   View history of model training


## **Deployments**

*   AWS Beanstalk
*   Azure
*   Render


# **Installation**


## **Requirements**

*   Python 3.10.10
*   Scikit-learn
*   Seaborn
*   Matplotlib
*   Plotly
*   Pandas
*   Numpy
*   Imbalanced Learn
*   PyYAML
*   dill
*   six
*   Flask
*   gunicorn
*   natsort
*   Evidently
*   yData Profiling
*   boto3


## **Docker**

A Dockerfile is a text document that contains all the commands a user could call on the command line to assemble an image. Docker images can be used to create containers, which are isolated environments that run your application. This is useful because it ensures that your application runs in the same environment regardless of where it is being deployed.

To build and run this project using Docker, follow these steps:

1.  Install Docker on your machine if you haven't already.
2.  Open a terminal window and navigate to the project root directory.
3.  Build the Docker image by running the following command:

    ```
    docker build -t <image-name>:<version> <location-of-docker-file for curren directory just add dot (.)>

    ```
    or

    ```
    docker build -t <image-name>:<version> .
    
    ```

4.  To Check List of Docker Images
    ```
    docker images
    ```    

5.  Start a container using the following command, replacing <image-name> and <version> with the values you used in step 3:

    ```
    docker run -p <host-port>:<container-port> <image-name>:<version>

    ```

    or

    ```
    docker run -p 5000:5000 -e PORT=5000 <Image-ID>
    ```

6.  Open a web browser and go to `http://localhost:<host-port>` to see the application running.

7.  Check Running Containers in docker

    ```
    docker ps
    ```

8.  Stop Docker Container

    ```
    docker stop <container_id>
    ```    


# **Project Structure**


```
ML-04-ThyroidPrediction
├─ .dockerignore
├─ .git
|
├─ .github
│  └─ workflows
│     └─ main.yaml
├─ .gitignore
├─ .idea
|
├─ app.py
├─ config
│  ├─ config.yaml
│  ├─ model.yaml
│  └─ schema.yaml
|
├─ Dockerfile
├─ Docs
│  ├─ DeveloperNotes.md
│  └─ Thyroid Disease Detection.pdf
├─ LICENSE
├─ logs
├─ Notebook
├─ README.md
├─ requirements.txt
├─ saved_models
│  ├─ 20230501225927
│  │  ├─ model.pkl
│  │  └─ score
│  │     ├─ model_score.csv
│  │     └─ model_score.html
│  ├─ 20230502024020
│    ├─ model.pkl
│    └─ score
│       ├─ model_score.csv
│       └─ model_score.html
│  
├─ setup.py
├─ static
│  ├─ css
│  │  └─ style.css
│  └─ js
│     └─ script.js
├─ templates
│  ├─ bulk_prediction.html
│  ├─ drift_report.html
│  ├─ eda.html
│  ├─ experiment_history.html
│  ├─ files.html
│  ├─ header.html
│  ├─ index.html
│  ├─ log.html
│  ├─ log_files.html
│  ├─ PerformanceReport.html
│  ├─ predict.html
│  ├─ ProfileReport_1.html
│  ├─ ProfileReport_2.html
│  ├─ saved_models_files.html
│  ├─ train.html
│  └─ update_model.html
└─ ThyroidPrediction
   ├─ artifact
   │  ├─ base_data_ingestion
   │  │  ├─ 2023-05-01-17-08-31
   │  │  │  └─ cleaned_data
   │  │  │     ├─ processed_data
   │  │  │     │  ├─ Cleaned_transformed
   │  │  │     │  │  └─ df_transformed_major_class.csv
   │  │  │     │  └─ split_data
   │  │  │     │     ├─ test_set
   │  │  │     │     │  └─ test.csv
   │  │  │     │     └─ train_set
   │  │  │     │        └─ train.csv
   │  │  │     └─ raw_data_merged
   │  │  │        └─ df_combined.csv
   │  │  ├─ 2023-05-01-22-06-15
   │  │    └─ cleaned_data
   │  │       ├─ processed_data
   │  │       │  ├─ Cleaned_transformed
   │  │       │  │  └─ df_transformed_major_class.csv
   │  │       │  └─ split_data
   │  │       │     ├─ test_set
   │  │       │     │  └─ test.csv
   │  │       │     └─ train_set
   │  │       │        └─ train.csv
   │  │       └─ raw_data_merged
   │  │          └─ df_combined.csv
   │  │  
   │  ├─ data_validation
   │  │  ├─ 2023-05-01-17-08-31
   │  │  │  ├─ drift_report.html
   │  │  │  └─ drift_report.json
   │  │  ├─ 2023-05-01-22-06-15
   │  │      ├─ drift_report.html
   │  │      └─ drift_report.json
   |  |
   │  ├─ experiment
   │  │  └─ experiment.csv
   │  ├─ model_evaluation
   │  │  └─ model_evaluation.yaml
   │  ├─ model_trainer
   │  │  ├─ 2023-05-01-17-08-31
   │  │  │  ├─ score
   │  │  │  │  ├─ model_score.csv
   │  │  │  │  └─ model_score.html
   │  │  │  └─ trained_model
   │  │  │     └─ model.pkl
   │  │  ├─ 2023-05-01-22-40-48
   │  │     ├─ performance
   │  │     │  └─ PerformanceReport.html
   │  │     ├─ score
   │  │     │  ├─ model_score.csv
   │  │     │  └─ model_score.html
   │  │     └─ trained_model
   │  │        └─ model.pkl
   |  |
   │  ├─ Profiling
   │  │  ├─ 2023-05-01-22-03-28
   │  │  │  ├─ Part_1
   │  │  │  └─ Part_2
   │  │  ├─ 2023-05-01-22-06-15
   │  │     ├─ Part_1
   │  │     │  └─ ProfileReport_1.html
   │  │     └─ Part_2
   │  │        └─ ProfileReport_2.html
   |  |
   │  └─ transformed_data_dir
   │     ├─ 2023-05-01-17-08-31
   │     │  └─ resampled_data
   │     │     ├─ test
   │     │     │  └─ test_non_resample_major.csv
   │     │     └─ train
   │     │        └─ train_non_resample_major.csv
   │     ├─ 2023-05-01-22-06-15
   │       └─ resampled_data
   │          ├─ test
   │          │  └─ test_non_resample_major.csv
   │          └─ train
   │             └─ train_non_resample_major.csv
   | 
   ├─ component
   │  ├─ data_ingestion.py
   │  ├─ data_transformation.py
   │  ├─ data_validation.py
   │  ├─ model_evaluation.py
   │  ├─ model_performance.py
   │  ├─ model_pusher.py
   │  ├─ model_trainer.py
   │  └─ __init__.py
   ├─ config
   │  ├─ configuration.py
   │  └─ __init__.py
   ├─ constant
   │  └─ __init__.py
   ├─ dataset_base
   │  ├─ Processed_Dataset
   │  │  ├─ Cleaned_Data
   │  │  │  └─ df_combined_cleaned.csv
   │  │  ├─ Resampled_Dataset
   │  │  │  ├─ ResampleData_major.csv
   │  │  │  ├─ test_resampled
   │  │  │  │  └─ test_non_resample_major.csv
   │  │  │  └─ train_resampled
   │  │  │     └─ train_resample_major.csv
   │  │  └─ Transformed_Data
   │  │     └─ df_transformed_major_class.csv
   │  ├─ raw_data
   │  │  ├─ allbp
   │  │  │  ├─ allbp.data
   │  │  │  ├─ allbp.names
   │  │  │  └─ allbp.test
   │  │  ├─ allhyper
   │  │  │  ├─ allhyper.data
   │  │  │  ├─ allhyper.names
   │  │  │  └─ allhyper.test
   │  │  ├─ allhypo
   │  │  │  ├─ allhypo.data
   │  │  │  ├─ allhypo.names
   │  │  │  └─ allhypo.test
   │  │  ├─ allrep
   │  │  │  ├─ allrep.data
   │  │  │  ├─ allrep.names
   │  │  │  └─ allrep.test
   │  │  ├─ dis
   │  │  │  ├─ dis.data
   │  │  │  ├─ dis.names
   │  │  │  └─ dis.test
   │  │  └─ sick
   │  │     ├─ sick.data
   │  │     └─ sick.test
   │  ├─ ResampledData_major_class.csv
   │  ├─ test_set
   │  │  ├─ test_set
   │  │  └─ test_set.csv
   │  └─ train_set
   │     └─ train_set.csv
   ├─ entity
   │  ├─ artifact_entity.py
   │  ├─ config_entity.py
   │  ├─ experiment.py
   │  ├─ model_factory.py
   │  ├─ thyroid_predictor.py
   │  └─ __init__.py
   ├─ exception
   │  └─ __init__.py
   ├─ logger
   │  └─ __init__.py
   ├─ pipeline
   │  ├─ pipeline.py
   │  └─ __init__.py
   ├─ secrets
   │  └─ __init__.py
   ├─ util
   │  ├─ util.py
   │  └─ __init__.py
   └─ __init__.py

```