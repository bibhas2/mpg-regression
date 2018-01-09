# 1) Randomizes the data 
# 2) Separate the data in training and validation sets
# 3) Calculates mu and sigma for the training set

import numpy as np
import pickle

lines = []
mu = None
sigma = None

with open("./auto-mpg.data") as input_file:
    x = []
    y = []

    for line in input_file:
        parts = line.split()
        
        y.append(float(parts[0]))
        x.append([
            float(parts[1]),
            float(parts[2]),
            float(parts[3]),
            float(parts[4]),
            float(parts[5]),
            float(parts[6]) - 70 #Model year starts with 1970
        ])
        lines.append(line)
    
    mu = np.mean(x, axis=0)
    sigma = np.std(x, axis=0)

    # print mu
    # print sigma

    #Randomize the sequence
    np.random.shuffle(lines)

#Save the data
#Take 90% for training
with open("./training.data", "w") as training_file, open("./validation.data", "w") as validation_file, open("./params.pickle", "wb") as params_file:
    split_point = int(len(lines) * 0.9)

    for index, line in enumerate(lines):
        if index < split_point:
            training_file.write(line)
        else:
            validation_file.write(line)

    pickle.dump(mu, params_file)
    pickle.dump(sigma, params_file)



