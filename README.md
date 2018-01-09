## Linear Regression Using Tensorflow
Linear regression is perhaps the simplest form of machine learning. It is one of the best ways to get to know a ML framework like Tensorflow. 

In this project we do linear regression for vehicle MPG prediction. Data is obtained from [UCI](https://archive.ics.uci.edu/ml/datasets/Auto+MPG).

## Running the Project

First prepare the data set.

```
python prepare_data.py
```

Then run training.

```
python model.py --train
```

Then run validation.

```
python model.py --validate
```

## Analysis
Linera Regression may be simple but it takes some care and feeding to be accurate. The ``prepare_data.py`` script does some of those things.

### Feature Scaling
Linera regression heavily depends on feature scaling of the data. Here we use standardization that make each feature value mean to be 0. The ``prepare_data.py`` script calculates the mean and standard deviation for the entire data set and saves it in ``params.pickle`` file. These values are later loaded during training and validation to normalize the feature data (``x`` matrix).

Try commenting out the feature scaling code in the ``load_next_batch()`` method of ``MPGDataLoader``. Training will go horribly wrong.

**Note:** Feature values need to be scaled both before training and prediction (which includes validation) stages.

### Data Separation
We need to separate the master data set into traing and validation sets. For this to work we need to shuffle the data randomly. Otherwise by default the UCI data is cronologically ordered. The ``prepare_data.py`` script randomly shuffles the master data and uses 90% of it for training and 10% for validation.

### The Loss Function
Here we use this loss function:

Loss = ((Y_ - Y)<sup>2</sup>/m)

Where ``Y_`` is the prediction vector, ``Y`` is the ground truth and ``m`` is the number of samples in a training batch. This worked fine.

But for some reason using the ``tf.nn.l2_loss`` function worked very badly. This is essentially:

Loss = ((Y_ - Y)<sup>2</sup>/2)


