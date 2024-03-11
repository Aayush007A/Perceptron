import pandas as pd
import numpy as np
import logging
from utils.all_utils import prepare_data,save_plot
from utils.model import Perceptron

OR = {
    "x1":[0,0,1,1],
    "x2":[0,1,0,1],
    "y":[0,1,1,1]
}

df_OR = pd.DataFrame(OR)

X,y = prepare_data(df_OR)

ETA = 0.01
EPOCHS = 10

model_or = Perceptron(eta=ETA, epochs=EPOCHS)

model_or.fit(X,y)

print(f"Total Model Loss is : {model_or.total_loss()}")

model_or.save(filename="OR.model")

save_plot(df_OR, model=model_or, filename="or_result.png")