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
        LSTM C State, and the LSTM H State (in that order: (c, h)).
        """
        zero_state = np.zeros((self.bsz, self.npi_core_dim))
        self.h_states = [(zero_state, zero_state) for _ in range(self.npi_core_layers)]

    def npi_core(self):
        """
        Build the NPI LSTM core, feeding the program embedding and state encoding to a multi-layered
        LSTM, returning the h-state of the final LSTM layer.

        References: Reed, de Freitas [2]
        """
        s_in = self.state_dim                                    # Shape: [bsz, state_dim]
        p_in = self.program_embedding                            # Shape: [bsz, 1, program_dim]

        # Reshape state_in
        s_in = tflearn.reshape(s_in, [None, 1, self.state_dim])  # Shape: [bsz, 1, state_dim]

        # Concatenate s_in, p_in
        c = tflearn.merge([s_in, p_in], 'concat', axis=2)        # Shape: [bsz, 1, state + prog]

        # Feed through Multi-Layer LSTM
        for i in range(len(self.npi_core_layers)):
            c, self.h_states[i] = tflearn.lstm(c, self.npi_core_dim, return_seq=True,
                                               initial_state=self.h_states[i], return_states=True)

        # Return Top-Most LSTM H-State
        return self.h_states[-1][1]                              # Shape: [bsz, npi_core_dim]