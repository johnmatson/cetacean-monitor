from prepro_rt import Preprocess
from model_rt import Model

prp = Preprocess()
model = Model()

def run():
    prp.read()
    data = prp.process()
    return model.predict(data)

# execute run function every ~1 second
run()
