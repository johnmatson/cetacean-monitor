from prepro_rt import Preprocess
from model_rt import Model
import numpy as np
import time

prp = Preprocess()
model = Model()

def run():
    prp.read()
    data = prp.process()
    return model.predict(data)

# execute run function every ~1 second
while(True):
    prediction = run()
    print(np.argmax(prediction, axis=1))
    time.sleep(1)
