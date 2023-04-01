from flask import Flask
from ThyroidPrediction.logger import logging
from ThyroidPrediction.exception import ThyroidException
app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def index():
    
    logging.info("loggeer Tested")
    return "Starting Thyroid Project"

if __name__ == "__main__":
    app.run(debug=True)