import pandas as pd
import numpy as np

from utils.all_utils import prepare_data,save_plot
from utils.model import Perceptron

AND = {
    "x1":[0,0,1,1],
    "x2":[0,1,0,1],
    "y":[0,0,0,1]
}

df_AND = pd.DataFrame(AND)

X,y = prepare_data(df_AND)

ETA = 0.01
EPOCHS = 10

model_and = Perceptron(eta=ETA, epochs=EPOCHS)

model_and.fit(X,y)

print(f"Total Model Loss is : {model_and.total_loss()}")

model_and.save(filename="AND.model")

save_plot(df_AND, model=model_and, filename="and_result.png")