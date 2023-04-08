from flask import Flask


from ThyroidPrediction.logger import logging
from ThyroidPrediction.exception import ThyroidException
from ThyroidPrediction. component import data_ingestion, data_transformation
app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def index():
        
    logging.info("split data")
    
    #return data_ingestion.DataIngestion(data_ingestion_config=data_ingestion).initiate_data_ingestion()
    
    return data_transformation.DataTransformation().initiate_data_transformation()

if __name__ == "__main__":
    app.run(debug=True)