"""
generate_data.py

Core script for generating training/test addition data. First, generates random pairs of numbers,
then steps through an execution trace, computing the exact order of subroutines that need to be
called.
"""
from tasks.addition.trace import Trace
import numpy as np
import pickle


def generate(prefix, num_examples, maximum=10000000000, debug_every=1000):
    """
    Generates addition data with the given string prefix (i.e. 'train', 'test') and the specified
    number of examples.

    :param prefix: String prefix for saving the file ('train', 'test')
    :param num_examples: Number of examples to generate.
    """
    data = []
    for i in range(num_examples):
        in1 = np.random.randint(maximum - 1)
        in2 = np.random.randint(maximum - in1)
        if i % debug_every == 0:
            trace = Trace(in1, in2, True).trace
        else:
            trace = Trace(in1, in2).trace
        data.append(( in1, in2, trace ))

    with open('tasks/addition/data/{}.pik'.format(prefix), 'wb') as f:
        pickle.dump(data, f)