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
Linera Regression may be simple but it takes a lot of care and feeding to be accurate. The ``prepare_data.py`` does some of those things.

### Data Sequence Randomization
We do training in mini batches. For this to work we need to shuffle the data randomly. Otherwise by default the UCI data is cronologically ordered.