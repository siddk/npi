"""
npi.py

Core model definition script for the Neural Programmer-Interpreter.
"""
import numpy as np
import os
import tensorflow as tf
import tflearn


class NPI():
    def __init__(self, core, config, log_path, npi_core_dim=256, npi_core_layers=2, verbose=0):
        """
        Instantiate an NPI Model, with the necessary hyperparameters, including the task-specific
        core.

        :param core: Task-Specific Core, with fields representing the environment state vector,
                     the input placeholders, and the program embedding.
        :param config: Task-Specific Configuration Dictionary, with fields representing the
                       necessary parameters.
        :param log_path: Path to save network checkpoint and tensorboard log files.
        """
        self.core, self.state_dim, self.program_dim = core, core.state_dim, core.program_dim
        self.bsz, self.npi_core_dim, self.npi_core_layers = core.bsz, npi_core_dim, npi_core_layers
        self.env_in, self.arg_in, self.prg_in = core.env_in, core.arg_in, core.prg_in
        self.state_encoding, self.program_embedding = core.state_encoding, core.program_embedding
        self.num_args, self.arg_depth = config["ARGUMENT_NUM"], config["ARGUMENT_DEPTH"]
        self.num_progs, self.key_dim = config["PROGRAM_NUM"], config["PROGRAM_KEY_SIZE"]
        self.log_path, self.verbose = log_path, verbose

        # Build NPI LSTM Core, hidden state
        self.reset_state()
        self.h = self.npi_core()

        # Build Termination Network => Returns probability of terminating
        self.terminate = self.terminate_net()

        # Build Key Network => Generates probability distribution over programs
        self.program_distribution = self.key_net()

        # Build Argument Networks => Generates list of argument distributions
        self.arguments = self.argument_net()

        # Build Regressions
        self.t_reg, self.p_reg, self.a_regs = self.loss()

        # Compile Model
        outputs = tflearn.merge_outputs([self.t_reg, self.p_reg] + self.a_regs)
        self.network = tflearn.DNN(outputs, tensorboard_dir=self.log_path,
                                   tensorboard_verbose=self.verbose,
                                   checkpoint_path=os.path.join(self.log_path, 'model.ckpt'))

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

    def terminate_net(self):
        """
        Build the NPI Termination Network, that takes in the NPI Core Hidden State, and returns
        the probability of terminating program.

        References: Reed, de Freitas [3]
        """
        p_terminate = tflearn.fully_connected(self.h, 1, activation='sigmoid', regularizer='L2')
        return p_terminate                                      # Shape: [bsz, 1]

    def key_net(self):
        """
        Build the NPI Key Network, that takes in the NPI Core Hidden State, and returns a softmax
        distribution over possible next programs.

        References: Reed, de Freitas [3, 4]
        """
        # Get Key from Key Network
        hidden = tflearn.fully_connected(self.h, self.key_dim, activation='elu', regularizer='L2')
        key = tflearn.fully_connected(hidden, self.key_dim)    # Shape: [bsz, key_dim]

        # Perform dot product operation, then softmax over all options to generate distribution
        key = tflearn.reshape(key, [None, 1, self.key_dim])
        key = tf.tile(key, [1, self.num_progs, 1])             # Shape: [bsz, n_progs, key_dim]
        prog_sim = tf.mul(key, self.core.program_key)          # Shape: [bsz, n_progs, key_dim]
        prog_sim = tf.reduce_sum(prog_sim, [2])                # Shape: [bsz, n_progs]
        prog_dist = tflearn.activation(prog_sim, 'softmax')
        return prog_dist

    def argument_net(self):
        """
        Build the NPI Argument Networks (a separate net for each argument), each of which takes in
        the NPI Core Hidden State, and returns a softmax over the argument dimension.

        References: Reed, de Freitas [3]
        """
        args = []
        for i in range(self.num_args):
            arg = tflearn.fully_connected(self.h, self.arg_depth, activation='softmax',
                                          regularizer='L2', name='Argument_{}'.format(str(i)))
            args.append(arg)
        return args                                             # Shape: [bsz, arg_depth]

    def loss(self):
        """
        Build separate output regressions, with the necessary loss functions.
        """
        # Termination Regression
        t_reg = tflearn.regression(self.terminate, loss='binary_crossentropy', batch_size=self.bsz,
                                   name='Termination_Out')

        # Program Regression
        p_reg = tflearn.regression(self.program_distribution, batch_size=self.bsz,
                                   name='Program_Out')

        # Argument Regressions
        arg_regs = []
        for i in range(len(self.arguments)):
            arg_regs.append(tflearn.regression(self.arguments[i], batch_size=self.bsz,
                                               name='Argument{}_Out'.format(str(i))))

        return t_reg, p_reg, arg_regs