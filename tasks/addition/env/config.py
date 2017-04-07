"""
config.py

Configuration Variables for the Addition NPI Task => Stores Scratch-Pad Dimensions, Vector/Program
Embedding Information, etc.
"""
import numpy as np
import sys
import time

CONFIG = {
    "ENVIRONMENT_ROW": 4,         # Input 1, Input 2, Carry, Output
    "ENVIRONMENT_COL": 10,        # 10-Digit Maximum for Addition Task
    "ENVIRONMENT_DEPTH": 10,      # Size of each element vector => One-Hot, Options: 0-9

    "ARGUMENT_NUM": 3,            # Maximum Number of Program Arguments
    "ARGUMENT_DEPTH": 11,         # Size of Argument Vector => One-Hot, Options 0-9, Default (10)
    "DEFAULT_ARG_VALUE": 10,      # Default Argument Value

    "PROGRAM_NUM": 6,             # Maximum Number of Subroutines
    "PROGRAM_KEY_SIZE": 5,        # Size of the Program Keys
    "PROGRAM_EMBEDDING_SIZE": 10  # Size of the Program Embeddings
}

PROGRAM_SET = [
    ("MOVE_PTR", 4, 2),       # Moves Pointer (4 options) either left or right (2 options)
    ("WRITE", 2, 10),         # Given Carry/Out Pointer (2 options) writes digit (10 options)
    ("ADD",),                 # Top-Level Add Program (calls children routines)
    ("ADD1",),                # Single-Digit (Column) Add Operation
    ("CARRY",),               # Carry Operation
    ("LSHIFT",)               # Shifts all Pointers Left (after Single-Digit Add)
]

PROGRAM_ID = {x[0]: i for i, x in enumerate(PROGRAM_SET)}


class ScratchPad():           # Addition Environment
    def __init__(self, in1, in2, rows=CONFIG["ENVIRONMENT_ROW"], cols=CONFIG["ENVIRONMENT_COL"]):
        # Setup Internal ScratchPad
        self.rows, self.cols = rows, cols
        self.scratchpad = np.zeros((self.rows, self.cols), dtype=np.int8)

        # Initialize ScratchPad In1, In2
        self.init_scratchpad(in1, in2)

        # Pointers initially all start at the right
        self.in1_ptr, self.in2_ptr, self.carry_ptr, self.out_ptr = self.ptrs = \
            [(x, -1) for x in range(4)]

    def init_scratchpad(self, in1, in2):
        """
        Initialize the scratchpad with the given input numbers (to be added together).
        """
        lst = [str(in1), str(in2)]
        for inpt in range(len(lst)):
            for i in range(1, len(lst[inpt]) + 1):
                self.scratchpad[inpt, -i] = int(lst[inpt][-i])

    def done(self):
        if self.in1_ptr[1] < -self.cols:
            return True
        else:
            lst = [x[1] for x in self.ptrs]
            if len(set(lst)) == 1:
                return sum(sum([self[x[0], :min(x[1] + 1, -1)] for x in self.ptrs])) == 0
            else:
                return False

    def add1(self):
        temp = self[self.in1_ptr] + self[self.in2_ptr] + self[self.carry_ptr]
        return temp % 10, temp / 10

    def write_carry(self, carry_val, debug=False):
        carry_row, carry_col = self.carry_ptr
        self[(carry_row, carry_col - 1)] = carry_val
        if debug:
            self.pretty_print()

    def write_out(self, value, debug=False):
        self[self.out_ptr] = value
        if debug:
            self.pretty_print()

    def lshift(self):
        self.in1_ptr, self.in2_ptr, self.carry_ptr, self.out_ptr = self.ptrs = \
            [(x, y - 1) for (x, y) in self.ptrs]

    def pretty_print(self):
        new_strs = ["".join(map(str, self[i])) for i in range(4)]
        line_length = len('Input 1:' + " " * 5 + new_strs[0])
        print 'Input 1:' + " " * 5 + new_strs[0]
        print 'Input 2:' + " " * 5 + new_strs[1]
        print 'Carry  :' + " " * 5 + new_strs[2]
        print '-' * line_length
        print 'Output :' + " " * 5 + new_strs[3]
        print ''
        time.sleep(.1)
        sys.stdout.flush()

    def get_env(self):
        env = np.zeros((CONFIG["ENVIRONMENT_ROW"], CONFIG["ENVIRONMENT_DEPTH"]), dtype=np.int32)
        if self.in1_ptr[1] < -CONFIG["ENVIRONMENT_COL"]:
            env[0][0] = 1
        else:
            env[0][self[self.in1_ptr]] = 1
        if self.in2_ptr[1] < -CONFIG["ENVIRONMENT_COL"]:
            env[1][0] = 1
        else:
            env[1][self[self.in2_ptr]] = 1
        if self.carry_ptr[1] < -CONFIG["ENVIRONMENT_COL"]:
            env[2][0] = 1
        else:
            env[2][self[self.carry_ptr]] = 1
        if self.out_ptr[1] < -CONFIG["ENVIRONMENT_COL"]:
            env[3][0] = 1
        else:
            env[3][self[self.out_ptr]] = 1
        return env.flatten()

    def execute(self, prog_id, args):
        if prog_id == 0:               # MOVE!
            ptr, lr = args
            lr = (lr * 2) - 1
            if ptr == 0:
                self.in1_ptr = (self.in1_ptr[0], self.in1_ptr[1] + lr)
            elif ptr == 1:
                self.in2_ptr = (self.in2_ptr[0], self.in2_ptr[1] + lr)
            elif ptr == 2:
                self.carry_ptr = (self.carry_ptr[0], self.carry_ptr[1] + lr)
            elif ptr == 3:
                self.out_ptr = (self.out_ptr[0], self.out_ptr[1] + lr)
            else:
                raise NotImplementedError
            self.ptrs = [self.in1_ptr, self.in2_ptr, self.carry_ptr, self.out_ptr]
        elif prog_id == 1:             # WRITE!
            ptr, val = args
            if ptr == 0:
                self[self.out_ptr] = val
            elif ptr == 1:
                self[self.carry_ptr] = val
            else:
                raise NotImplementedError

    def __getitem__(self, item):
        return self.scratchpad[item]

    def __setitem__(self, key, value):
        self.scratchpad[key] = value


class Arguments():             # Program Arguments
    def __init__(self, args, num_args=CONFIG["ARGUMENT_NUM"], arg_depth=CONFIG["ARGUMENT_DEPTH"]):
        self.args = args
        self.arg_vec = np.zeros((num_args, arg_depth), dtype=np.float32)


def get_args(args, arg_in=True):
    if arg_in:
        arg_vec = np.zeros((CONFIG["ARGUMENT_NUM"], CONFIG["ARGUMENT_DEPTH"]), dtype=np.int32)
    else:
        arg_vec = [np.zeros((CONFIG["ARGUMENT_DEPTH"]), dtype=np.int32) for _ in
                   range(CONFIG["ARGUMENT_NUM"])]
    if len(args) > 0:
        for i in range(CONFIG["ARGUMENT_NUM"]):
            if i >= len(args):
                arg_vec[i][CONFIG["DEFAULT_ARG_VALUE"]] = 1
            else:
                arg_vec[i][args[i]] = 1
    else:
        for i in range(CONFIG["ARGUMENT_NUM"]):
            arg_vec[i][CONFIG["DEFAULT_ARG_VALUE"]] = 1
    return arg_vec.flatten() if arg_in else arg_vec

