import pandas as pd
import numpy as np
import logging
import os
from utils.all_utils import prepare_data,save_plot
from utils.model import Perceptron

log_dir = "logs"
os.makedirs(log_dir,exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    filename=os.path.join(log_dir,"running_logs.log"),
    format='[%(asctime)s : %(levelname)s: %(module)s]: %(message)s',
    filemode='a'
)

def main(data, eta, epochs, modelname, plotname):

    df = pd.DataFrame(data)
    logging.info(f"This is the raw dataset : \n{df}")

    X,y = prepare_data(df)
    logging.info(f"X is : \n{X} and y is : \n{y}")

    model = Perceptron(eta=eta, epochs=epochs)
    logging.info("Model created successfully !!!")

    model.fit(X,y)
    logging.info("Model fited successfully !!!")

    logging.info(f"Total Model Loss is : {model.total_loss()}")

    model.save(filename=modelname)
    logging.info("Model saved successfully !!!")

    save_plot(df, model=model, filename=plotname)
    logging.info("Prediction plot created and saved successfully !!!")


if __name__ == "__main__":

    XNOR = {
        "x1":[0,0,1,1],
        "x2":[0,1,0,1],
        "y":[0,1,0,1]
    }

    ETA = 0.01
    EPOCHS = 20
    MODELNAME = "XNOR.model"
    PLOTNAME = "xnor_result.png"

    try:

        logging.info("-----------------Training started-----------------")
        logging.info(f"Given parameter values ETA : {ETA} EPOCHS : {EPOCHS} MODELNAME : {MODELNAME} PLOTNAME : {PLOTNAME}")
        main(XNOR, eta=ETA, epochs=EPOCHS, modelname=MODELNAME, plotname=PLOTNAME)
        logging.info("-----------------Training completed---------------\n\n")

    except Exception as e:

        logging.exception(e)
        raise e

    