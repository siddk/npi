"""
config.py

Configuration Variables for the Addition NPI Task => Stores Scratch-Pad Dimensions, Vector/Program
Embedding Information, etc.
"""
import numpy as np
import sys
import time

CONFIG = {
    "ENVIRONMENT_ROW": 4,     # Input 1, Input 2, Carry, Output
    "ENVIRONMENT_COL": 10,    # 10-Digit Maximum for Addition Task
    "ENVIRONMENT_DEPTH": 10,  # Size of each element vector => One-Hot, Options: 0-9

    "ARGUMENT_NUM": 3,        # Maximum Number of Program Arguments
    "ARGUMENT_DEPTH": 10,     # Size of each argument vector => One-Hot, Options 0-9
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

    def __getitem__(self, item):
        return self.scratchpad[item]

    def __setitem__(self, key, value):
        self.scratchpad[key] = value


class Arguments():             # Program Arguments
    def __init__(self, args, num_args=CONFIG["ARGUMENT_NUM"], arg_depth=CONFIG["ARGUMENT_DEPTH"]):
        self.args = args
        self.arg_vec = np.zeros((num_args, arg_depth), dtype=np.float32)
