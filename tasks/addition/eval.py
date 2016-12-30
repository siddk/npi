"""
eval.py

Loads in an Addition NPI, and starts a REPL for interactive addition.
"""
from model.npi import NPI
from tasks.addition.addition import AdditionCore
from tasks.addition.env.config import CONFIG, get_args, PROGRAM_SET, ScratchPad
import numpy as np
import pickle
import tensorflow as tf

LOG_PATH = "tasks/addition/log/"
CKPT_PATH = "tasks/addition/log/model.ckpt"
TEST_PATH = "tasks/addition/data/test.pik"
MOVE_PID, WRITE_PID = 0, 1
W_PTRS = {0: "OUT", 1: "CARRY"}
PTRS = {0: "IN1_PTR", 1: "IN2_PTR", 2: "CARRY_PTR", 3: "OUT_PTR"}
R_L = {0: "LEFT", 1: "RIGHT"}


def evaluate_addition():
    """
    Load NPI Model from Checkpoint, and initialize REPL, for interactive carry-addition.
    """
    with tf.Session() as sess:
        # Load Data
        with open(TEST_PATH, 'r') as f:
            data = pickle.load(f)

        # Initialize Addition Core
        core = AdditionCore()

        # Initialize NPI Model
        npi = NPI(core, CONFIG, LOG_PATH)

        # Restore from Checkpoint
        saver = tf.train.Saver()
        saver.restore(sess, CKPT_PATH)

        # Run REPL
        repl(sess, npi, data)


def repl(session, npi, data):
    while True:
        inpt = raw_input('Enter Two Numbers, or Hit Enter for Random Pair: ')

        if inpt == "":
            x, y, _ = data[np.random.randint(len(data))]

        else:
            x, y = map(int, inpt.split())

        # Reset NPI States
        print ""
        npi.reset_state()

        # Setup Environment
        scratch = ScratchPad(x, y)
        prog_name, prog_id, arg, term = 'ADD', 2, [], False

        cont = 'c'
        while cont == 'c' or cont == 'C':
            # Print Step Output
            if prog_id == MOVE_PID:
                a0, a1 = PTRS.get(arg[0], "OOPS!"), R_L[arg[1]]
                a_str = "[%s, %s]" % (str(a0), str(a1))
            elif prog_id == WRITE_PID:
                a0, a1 = W_PTRS[arg[0]], arg[1]
                a_str = "[%s, %s]" % (str(a0), str(a1))
            else:
                a_str = "[]"

            print 'Step: %s, Arguments: %s, Terminate: %s' % (prog_name, a_str, str(term))
            print 'IN 1: %s, IN 2: %s, CARRY: %s, OUT: %s' % (scratch.in1_ptr[1],
                                                              scratch.in2_ptr[1],
                                                              scratch.carry_ptr[1],
                                                              scratch.out_ptr[1])

            # Update Environment if MOVE or WRITE
            if prog_id == MOVE_PID or prog_id == WRITE_PID:
                scratch.execute(prog_id, arg)

            # Print Environment
            scratch.pretty_print()

            # Get Environment, Argument Vectors
            env_in, arg_in, prog_in = [scratch.get_env()], [get_args(arg, arg_in=True)], [[prog_id]]
            t, n_p, n_args = session.run([npi.terminate, npi.program_distribution, npi.arguments],
                                         feed_dict={npi.env_in: env_in, npi.arg_in: arg_in,
                                                    npi.prg_in: prog_in})

            if np.argmax(t) == 1:
                print 'Step: %s, Arguments: %s, Terminate: %s' % (prog_name, a_str, str(True))
                print 'IN 1: %s, IN 2: %s, CARRY: %s, OUT: %s' % (scratch.in1_ptr[1],
                                                                  scratch.in2_ptr[1],
                                                                  scratch.carry_ptr[1],
                                                                  scratch.out_ptr[1])
                # Update Environment if MOVE or WRITE
                if prog_id == MOVE_PID or prog_id == WRITE_PID:
                    scratch.execute(prog_id, arg)

                # Print Environment
                scratch.pretty_print()

                output = int("".join(map(str, map(int, scratch[3]))))
                print "Model Output: %s + %s = %s" % (str(x), str(y), str(output))
                print "Correct Out : %s + %s = %s" % (str(x), str(y), str(x + y))
                print "Correct!" if output == (x + y) else "Incorrect!"

            else:
                prog_id = np.argmax(n_p)
                prog_name = PROGRAM_SET[prog_id][0]
                if prog_id == MOVE_PID or prog_id == WRITE_PID:
                    arg = [np.argmax(n_args[0]), np.argmax(n_args[1])]
                else:
                    arg = []
                term = False

            cont = raw_input('Continue? ')