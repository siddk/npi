"""
main.py
"""
import tensorflow as tf

from tasks.addition.env.generate_data import generate_addition
from tasks.addition.eval import evaluate_addition
from tasks.addition.train import train_addition


FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string("task", "addition", "Which NPI Task to run - [addition].")
tf.app.flags.DEFINE_boolean("generate", False, "Boolean whether to generate training/test data.")
tf.app.flags.DEFINE_integer("num_training", 1000, "Number of training examples to generate.")
tf.app.flags.DEFINE_integer("num_test", 100, "Number of test examples to generate.")

tf.app.flags.DEFINE_boolean("do_train", False, "Boolean whether to continue training model.")
tf.app.flags.DEFINE_boolean("do_eval", True, "Boolean whether to perform model evaluation.")
tf.app.flags.DEFINE_integer("num_epochs", 5, "Number of training epochs to perform.")


def main(_):
    if FLAGS.task == "addition":
        # Generate Data (if necessary)
        if FLAGS.generate:
            generate_addition('train', FLAGS.num_training)
            generate_addition('test', FLAGS.num_test)

        # Train Model (if necessary)
        if FLAGS.do_train:
            train_addition(FLAGS.num_epochs)

        # Evaluate Model
        if FLAGS.do_eval:
            evaluate_addition()


if __name__ == "__main__":
    tf.app.run()