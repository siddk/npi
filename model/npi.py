"""
npi.py

Core model definition script for the Neural Programmer-Interpreter.
"""
import numpy as np
import tensorflow as tf
import tflearn


class NPI():
    def __init__(self, core, npi_core_dim=256, npi_core_layers=2):
        """
        Instantiate an NPI Model, with the necessary hyperparameters, including the task-specific
        core.

        :param core: Task-Specific Core, with fields representing the environment state vector,
                     the input placeholders, and the program embedding.
        :type core: AdditionCore
        """
        self.core, self.state_dim, self.program_dim = core, core.state_dim, core.program_dim
        self.bsz, self.npi_core_dim, self.npi_core_layers = core.bsz, npi_core_dim, npi_core_layers
        self.env_in, self.arg_in, self.prg_in = core.env_in, core.arg_in, core.prg_in
        self.state_encoding, self.program_embedding = core.state_encoding, core.program_embedding

        # Build NPI LSTM Core, hidden state
        self.reset_state()
        self.h = self.npi_core()

    def reset_state(self):
        """
        Zero NPI Core LSTM Hidden States. LSTM States are represented as a Tuple, consisting of the
        LSTM C State, and the LSTM H State.
        """
        zero_state = np.zeros((self.bsz, self.npi_core_dim))
        self.h_states = [(zero_state, zero_state) for _ in range(self.npi_core_layers)]

    def npi_core(self):
        """
        Build the NPI LSTM core, feeding the program embedding and state encoding to a multi-layered
        LSTM, returning the h-state of the final LSTM layer.
        """
        state_in = self.state_dim         # Shape: [bsz,


