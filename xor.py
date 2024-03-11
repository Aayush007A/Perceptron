import pandas as pd
import numpy as np

from utils.all_utils import prepare_data,save_plot
from utils.model import Perceptron

def main(data, eta, epochs, modelname, plotname):

    df = pd.DataFrame(data)

    X,y = prepare_data(df)

    model = Perceptron(eta=eta, epochs=epochs)

    model.fit(X,y)

    print(f"Total Model Loss is : {model.total_loss()}")

    model.save(filename=modelname)

    save_plot(df, model=model, filename=plotname)


if __name__ == "__main__":

    XOR = {
        "x1":[0,0,1,1],
        "x2":[0,1,0,1],
        "y":[0,1,1,0]
    }
    main(XOR, eta=0.01, epochs=10, modelname="XOR.model", plotname="xor_result.png")
    