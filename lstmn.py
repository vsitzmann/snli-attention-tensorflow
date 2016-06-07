import tensorflow as tf

class LSTMNBase(object):
    '''Base class implementing the constructor and the generic soft attention reweighting.

    Args:
        num_units (int): Number of units in the hidden LSTMN layer
        scope (Optional[tf VariableScope]): Scope that all operations will be prefixed with.

    Attributes:
        num_units (int): Number of units in the hidden LSTMN layer
        scope (tf VariableScope): Scope that all operations will be prefixed with.
    '''
    def __init__(self, num_units, scope=None):
        self.num_units = num_units

        if scope:
            self.scope=scope
        else:
            self.scope=tf.variable_scope('LSTMN')

    def attention_reweighting(self, hidden_tape, memory_tape, inputs, prev_attn_state, scope):
        '''Soft attention over a hidden and memory tape.

        Args:
            hidden_tape (tf Tensor): hidden tape that soft attention should be calculated over.
                            Shape [batch_size, num_prev_states, self.num_units]
            memory_tape (tf Tensor): The memory tape that soft attention should be calculated over.
                            Shape [batch_size, num_prev_states, self.num_units]
            inputs (tf Tensor): An input vector that guides attention.
                            Shape [batch_size, unknown]
            prev_attn_state (tf Tensor): Previous result of attention-reweighting a hidden tape.
                            Shape [batch_size, self.num_units]
            scope (tf VariableScope): Scope that all operations should be prefixed with.

        Returns:
            memory_attn (tf Tensor): attention-reweighted previous memories. Shape [batch_size, self.num_units].
            hidden_attn (tf Tensor): attention-reweighted previous hidden states. Shape [batch_size, self.num_units].
        '''
        attn_length = hidden_tape.get_shape()[1].value
        num_units = hidden_tape.get_shape()[2].value
        attn_feat_size = num_units

        # 1x1 convolution to project each previous hidden state into depth att_feat_size
        hidden_tape_reshaped = tf.reshape(hidden_tape, [-1, attn_length, 1, num_units])
        hidden_feat_kernel = tf.get_variable("hProj", [1, 1, num_units, attn_feat_size])
        v = tf.get_variable("AttnV", [attn_feat_size])

        hidden_tape_feats = tf.nn.conv2d(hidden_tape_reshaped,
                                         hidden_feat_kernel,
                                         strides=[1, 1, 1, 1],
                                         padding="SAME",
                                         name=scope.name+'/conv_hidden_feats')
        input_feats = tf.contrib.layers.fully_connected(x=inputs,
                                                        num_output_units=attn_feat_size,
                                                        bias_init=tf.constant_initializer(0.0),
                                                        name=scope.name+'/fc_infeats')
        prev_att_state_feats = tf.contrib.layers.fully_connected(x=prev_attn_state,
                                                                 num_output_units=attn_feat_size,
                                                                 bias_init=tf.constant_initializer(0.0),
                                                                 name=scope.name+'/fc_prevattfeats')

        input_feats_reshaped = tf.reshape(input_feats, [-1, 1, 1, attn_feat_size])
        prev_att_feats_reshaped = tf.reshape(prev_att_state_feats, [-1, 1, 1, attn_feat_size])

        # Attention mask is a softmax of v^T * tanh(...).
        attn_feats_sum = prev_att_feats_reshaped + input_feats_reshaped + hidden_tape_feats
        s = tf.reduce_sum(v * tf.tanh(attn_feats_sum), [2, 3], name=scope.name + '/s')
        attn_weights = tf.nn.softmax(s, name=scope.name + '/softmax')

        # Now calculate the attention-weighted previous hidden and memory states.
        memory_tape_reshaped = tf.reshape(memory_tape, [-1, attn_length, 1, num_units])

        memory_attn_unsqueez = tf.reduce_sum(tf.reshape(attn_weights, [-1, attn_length, 1, 1]) * memory_tape_reshaped, [1, 2])
        hidden_attn_unsqueez = tf.reduce_sum(tf.reshape(attn_weights, [-1, attn_length, 1, 1]) * hidden_tape_reshaped, [1, 2])

        memory_attn = tf.reshape(memory_attn_unsqueez, [-1, num_units])
        hidden_attn = tf.reshape(hidden_attn_unsqueez, [-1, num_units])

        return memory_attn, hidden_attn

class LSTMN(LSTMNBase):
    '''LSTMN encoder with intra-attention.
    '''
    def zero_states(self, batch_size, dtype=tf.float32):
        """Return zero-filled state and tape tensors.
        Args:
            batch_size (int or unit Tensor): Batch size.
            dtype: the data type to use for the state.
        Returns:
            mem_tape_zeros, hid_tape_zeros: Zero-initialized hidden, state tapes of shape [batch_size, 1, self.num_units]
            state_zeros: Zero-initialized
        """
        zeros = tf.zeros(tf.pack([batch_size, self.num_units]), dtype=dtype)
        mem_tape_zeros = tf.reshape(zeros, [-1, 1, self.num_units])
        hid_tape_zeros = tf.reshape(zeros, [-1, 1, self.num_units])
        state_zeros = tf.reshape(zeros, [-1, 1, self.num_units])
        return mem_tape_zeros, hid_tape_zeros, state_zeros

    def __call__(self, inputs, prev_memory_tape, prev_hidden_tape, prev_att_state):
        '''One encoder time step: Calculate new hidden, memory state tapes and new attention reweighted hidden tape.

        Args:
            inputs (tf Tensor): timestep input (e.g. word embedding). Shape [batch_size, unknown].
            prev_memory_tape (tf Tensor): Previous memory tape. Shape [batch_size, num previous steps, self.num_units].
            prev_hidden_tape (tf Tensor): Previous hidden tape. Shape [batch_size, num previous steps, self.num_units].
            prev_att_state (tf Tensor): Previous attention-reweighted hidden state..
                                    Shape [batch_size, self.num_units]

        Returns:
            new_hidden_tape, new_memory_tape (tf Tensors):
                                    Hidden and memory tapes with new hidden and memory states appended.
                                    Shape [batch_size, num previous steps + 1, self.num_units].
            prev_h_attn (tf Tensor): Attention-reweighted hidden tape. Shape [batch_size, self.num_units]
        '''
        prev_c_attn, prev_h_attn = self.attention_reweighting(hidden_tape=prev_hidden_tape,
                                                              memory_tape=prev_memory_tape,
                                                              inputs=inputs,
                                                              scope=self.scope,
                                                              prev_attn_state=prev_att_state)

        lstm_input = tf.concat(1, [inputs, prev_h_attn])
        pre_lstm_activs = tf.contrib.layers.fully_connected(x=lstm_input,
                                                             num_output_units=4*self.num_units,
                                                            bias_init=tf.constant_initializer(0.0),
                                                            name=self.scope.name+'/fc_activs')

        i, f, o, proposed_c = tf.split(1, 4, pre_lstm_activs)

        new_c = (prev_c_attn * tf.sigmoid(f) + tf.sigmoid(i) * tf.tanh(proposed_c))
        new_h = tf.tanh(new_c) * tf.sigmoid(o)

        new_c = tf.reshape(new_c, [-1, 1, self.num_units])
        new_h = tf.reshape(new_h, [-1, 1, self.num_units])

        new_hidden_tape = tf.concat(1, [prev_hidden_tape, new_h])
        new_memory_tape = tf.concat(1, [prev_memory_tape, new_c])

        return new_hidden_tape, new_memory_tape, prev_h_attn


class DeepAttnFusionLSTMN(LSTMNBase):
    '''LSTMN decoder with intra-attention and inter-attention (deep attention fusion).
    '''
    def zero_states(self, batch_size, dtype=tf.float32):
        """Return zero-filled state and tape tensors.
        Args:
            batch_size (int or unit Tensor): Batch size.
            dtype: the data type to use for the state.
        Returns:
            mem_tape_zeros, hid_tape_zeros: Zero-initialized hidden, state tapes of shape [batch_size, 1, self.num_units]
            enc_state_zeros, dec_state_zeros: Zero-initialized encoder and decoder attention-reweighted states.
        """
        zeros = tf.zeros(tf.pack([batch_size, self.num_units]), dtype=dtype)
        mem_tape_zeros = tf.reshape(zeros, [-1, 1, self.num_units])
        hid_tape_zeros = tf.reshape(zeros, [-1, 1, self.num_units])
        enc_state_zeros = tf.reshape(zeros, [-1, 1, self.num_units])
        dec_state_zeros = tf.reshape(zeros, [-1, 1, self.num_units])
        return mem_tape_zeros, hid_tape_zeros, enc_state_zeros, dec_state_zeros

    def __call__(self,
                 inputs,
                 prev_memory_tape,
                 prev_hidden_tape,
                 prev_att_state,
                 enc_hidden_tape,
                 enc_memory_tape,
                 enc_prev_att_state):
        '''One LSTMN time step: Generate new hidden, memory state tapes and attention-reweighted hidden state.

        Args:
            inputs (tf Tensor): timestep input (e.g. word embedding). Shape [batch_size, unknown].
            prev_memory_tape (tf Tensor): Previous memory tape. Shape [batch_size, num previous steps, self.num_units].
            prev_hidden_tape (tf Tensor): Previous hidden tape. Shape [batch_size, num previous steps, self.num_units].
            prev_att_state (tf Tensor): Previous attention-reweighted hidden state of this decoder.
                                    Shape [batch_size, self.num_units]
            enc_hidden_tape (tf Tensor): Full hidden tape of encoder.
                                    Shape [batch_size, num encoder time steps, self.num_units]
            enc_memory_tape (tf Tensor): Full hidden tape of encoder.
                                    Shape [batch_size, num encoder time steps, self.num_units]
            enc_prev_att_state (tf Tensor): Previous decoder-reweighted hidden tape of encoder.
                                    Shape [batch_size, self.num_units]
        Returns:
            new_hidden_tape, new_memory_tape (tf Tensors):
                                    Hidden and memory tapes with new hidden and memory states appended.
                                    Shape [batch_size, num previous steps + 1, self.num_units].
            dec_hidd_attn, enc_hidd_attn (tf Tensors):
                                    Attention-reweighted encoder and decoder hidden tapes.
                                    Shape [batch_size, self.num_units].
        '''
        with tf.variable_scope('encoder_attn') as scope:
            enc_mem_attn, enc_hidd_attn = self.attention_reweighting(hidden_tape=enc_hidden_tape,
                                                                memory_tape=enc_memory_tape,
                                                                prev_attn_state=enc_prev_att_state,
                                                                inputs=inputs,
                                                                scope=scope)
            enc_prop_input = tf.concat(1, [inputs, enc_hidd_attn])
            enc_proposal_weights = tf.contrib.layers.fully_connected(x=enc_prop_input,
                                                                num_output_units=self.num_units,
                                                                bias_init=tf.constant_initializer(0.0),
                                                                name=scope.name+'/fc_activs')

        with tf.variable_scope('decoder_attn') as scope:
            dec_mem_attn, dec_hidd_attn = self.attention_reweighting(hidden_tape=prev_hidden_tape,
                                                                memory_tape=prev_memory_tape,
                                                                prev_attn_state=prev_att_state,
                                                                inputs=inputs,
                                                                scope=scope)

            lstm_input = tf.concat(1, [inputs, dec_hidd_attn])
            pre_lstm_activs = tf.contrib.layers.fully_connected(x=lstm_input,
                                                                num_output_units=4*self.num_units,
                                                                bias_init=tf.constant_initializer(0.0),
                                                                name=scope.name+'/fc_activs')

        i, f, o, proposed_c = tf.split(1, 4, pre_lstm_activs)

        new_c = (dec_mem_attn * tf.sigmoid(f) + tf.sigmoid(i) * tf.tanh(proposed_c) + tf.sigmoid(enc_proposal_weights)*enc_mem_attn)
        new_h = tf.tanh(new_c) * tf.sigmoid(o)

        new_c = tf.reshape(new_c, [-1, 1, self.num_units])
        new_h = tf.reshape(new_h, [-1, 1, self.num_units])

        new_hidden_tape = tf.concat(1, [prev_hidden_tape, new_h])
        new_memory_tape = tf.concat(1, [prev_memory_tape, new_c])

        return new_hidden_tape, new_memory_tape, dec_hidd_attn, enc_hidd_attn

