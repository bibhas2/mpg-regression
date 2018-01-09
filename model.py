import tensorflow as tf
import numpy as np
import sys
import pickle

def weight_variable(shape, name):
    initial = tf.truncated_normal(shape, stddev=0.001)
    return tf.Variable(initial, name=name)


def bias_variable(shape, name):
    return tf.Variable(tf.zeros(shape), name=name)

class LinearRegressionModel(object):
    def __init__(self, num_features):
        self.num_features = num_features
        self.build_graph()

    def build_graph(self):
        #We add an extra feature for bias. The feature value for
        #bias is always 1.
        self.X = tf.placeholder(tf.float32, [None, self.num_features])
        self.Y = tf.placeholder(tf.float32, [None, 1])
        self.b = weight_variable([1], "b")
        self.W = weight_variable([self.num_features, 1], 'W')

        self.Y_ = tf.add(tf.matmul(self.X, self.W), self.b)

        self.l2_loss = tf.reduce_mean(tf.square(self.Y_ - self.Y)) #tf.nn.l2_loss(tf.subtract(self.Y, self.Y_))
        self.training_step = tf.train.GradientDescentOptimizer(0.01).minimize(self.l2_loss)
        
        self.error_rate = tf.reduce_mean(tf.abs(self.Y_/self.Y - 1.0))

    def train_batch(self, session, train_x, train_y):
        feed_dict = {self.X: train_x, self.Y: train_y}
        
        return session.run(self.training_step, feed_dict)

    def run_validation(self, session, validation_x, validation_y):
        feed_dict = {self.X: validation_x, self.Y: validation_y}

        return session.run(self.error_rate, feed_dict)

    def save(self, session):
        saver = tf.train.Saver()
        saver.save(session, "./model.ckpt")

    def restore(self, session):
        saver = tf.train.Saver()
        saver.restore(session, "./model.ckpt")

class MPGDataLoader(object):
    def __init__(self, file_name):
        self.file_name = file_name
        self.data_file = None
        self.num_features = 6
        self.mu = None
        self.sigma = None

        with open("./params.pickle", "rb") as params_file:
            self.mu = pickle.load(params_file)
            self.sigma = pickle.load(params_file)

    def __enter__(self):
        self.data_file = open(self.file_name, "r")

        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.data_file != None:
            self.data_file.close()

    def load_next_batch(self, batch_size):
        #One extra feature for bias
        x = np.zeros([batch_size, self.num_features])
        y = np.zeros([batch_size, 1])

        for index in range(0, batch_size):
            line = self.data_file.readline()
            if line == "":
                self.data_file.seek(0)
                line = self.data_file.readline()
            
            parts = line.split()

            y[index, 0] = float(parts[0])

            x[index, 0] = float(parts[1])
            x[index, 1] = float(parts[2])
            x[index, 2] = float(parts[3])
            x[index, 3] = float(parts[4])
            x[index, 4] = float(parts[5])
            #Model year starts with 1970
            x[index, 5] = float(parts[6]) - 70
        
        x = (x - self.mu) / self.sigma

        return (y, x)

def train():
    with MPGDataLoader("./training.data") as loader:
        with tf.Session() as session:
            model = LinearRegressionModel(loader.num_features)

            init_op = tf.global_variables_initializer()
            session.run(init_op)

            for step in range(0, 600):
                y, x = loader.load_next_batch(150)

                model.train_batch(session, x, y)

                if step % 100 == 0:
                    error_rate = model.run_validation(session, x, y)
                    print "Step:", step, "Error:", (error_rate*100.0), "%"

            model.save(session)

def validate():
    with MPGDataLoader("./validation.data") as loader:
        with tf.Session() as session:
            model = LinearRegressionModel(loader.num_features)

            model.restore(session)

            for step in range(0, 5):
                y, x = loader.load_next_batch(50)

                error_rate = model.run_validation(session, x, y)
                print "Step:", step, "Error:", (error_rate*100.0), "%"

if len(sys.argv) == 1:
    print "Usage: [--train] [--validate]"
elif sys.argv[1] == "--train":
    train()
elif sys.argv[1] == "--validate":
    validate()
