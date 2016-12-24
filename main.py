"""
main.py
"""
import tensorflow as tf

from tasks.addition.env.generate_data import generate_addition


FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string("task", "addition", "Which NPI Task to run - [addition].")
tf.app.flags.DEFINE_boolean("generate", False, "Boolean whether to generate training/test data.")
tf.app.flags.DEFINE_integer("num_training", 500, "Number of training examples to generate.")
tf.app.flags.DEFINE_integer("num_test", 50, "Number of test examples to generate.")

tf.app.flags.DEFINE_boolean("do_train", True, "Boolean whether to continue training model.")


def main(_):
    if FLAGS.task == "addition":
        # Generate Data (if necessary)
        if FLAGS.generate:
            generate_addition('train', FLAGS.num_training)
            generate_addition('test', FLAGS.num_test)




if __name__ == "__main__":
    tf.app.run()