import tensorflow
tf = tensorflow.compat.v1

act_fn = tf.nn.relu 

class MessageNet(object):
    def __init__(self, d, nlayers, name, outputsize = -1, activation = None):
        self.d = d
        self.nlayers = nlayers
        self.name = name
        if activation is None:
            self.activation = act_fn
        else:
            self.activation = activation
        if outputsize > 0:
            self.outputsize = outputsize
        else:
            self.outputsize = d
        self.w = []
        self.b = []
        with tf.variable_scope(name) as scope:
            outsize = d
            for i in range(nlayers):
                with tf.variable_scope(str(i)) as scope:
                    if i == nlayers - 1:
                        outsize = self.outputsize
                    self.w.append(tf.get_variable(name = 'w', shape=[d,outsize], initializer=tensorflow.contrib.layers.xavier_initializer()))
                    self.b.append(tf.get_variable(name = 'b', shape=[outsize], initializer=tf.zeros_initializer()))



    def apply(self, input):
        with tf.variable_scope(self.name) as scope:
            state = input
            for i in range(self.nlayers):
                with tf.variable_scope(str(i)) as scope:
                    state = tf.matmul(state, self.w[i]) + self.b[i]
                    if i != self.nlayers - 1:
                        state = self.activation(state)
        return state
