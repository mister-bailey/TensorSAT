import tensorflow
tf = tensorflow.compat.v1
import numpy as np
import math
from messages import MessageNet

d = 64
nlayers = 3
nfinal = 3
nrounds = 40
act_fn = tf.nn.relu

# Learning rate:


#print(parse.load_file('C:/Users/miste/Dropbox/MapleSAT/Benchmarks/Beijing/2bitcomp_5.cnf'))

sess = tf.Session()

# Placeholders, i.e., input data and labels

nvars = tf.placeholder(tf.int32, shape=[], name='nvars')
nlits = tf.placeholder(tf.int32, shape=[], name='nlits')
nclauses = tf.placeholder(tf.int32, shape=[], name='nclauses')
nrounds_p = tf.placeholder(tf.int32, shape=[], name='nrounds')

# Literal-clause membership matrix
LC_mat = tf.sparse_placeholder(tf.float32, [None,None], name='LC_mat')
# Labelled solution: config later if needed
solution = None #tf.placeholder(tf.float32, shape=[None], name='solution')


# Model parameters:

L_init = tf.get_variable(name='L_init', initializer=tf.random_normal([1,d]))
C_init = tf.get_variable(name='C_init', initializer=tf.random_normal([1,d]))

LC_message = MessageNet(d, nlayers, 'LC_message')
CL_message = MessageNet(d, nlayers, 'CL_message')

L_update = tensorflow.contrib.rnn.LayerNormBasicLSTMCell(d, activation=act_fn)
C_update = tensorflow.contrib.rnn.LayerNormBasicLSTMCell(d, activation=act_fn)

L_logits = MessageNet(d, nfinal, 'L_logits', outputsize = 1)

# Recurrence loop

def loop_body(i, L_state, C_state):
    LC_msg = LC_message.apply(L_state.h)
    LC_msgs = tf.sparse_tensor_dense_matmul(LC_mat, LC_msg, adjoint_a=True)

    with tf.variable_scope('C_update') as scope:
        C_rands = tf.random_uniform([nclauses,1])
        _, C_state = C_update(inputs=tf.concat([LC_msgs, C_rands], axis=1), state=C_state)

    CL_msg = CL_message.apply(C_state.h)
    CL_msgs = tf.sparse_tensor_dense_matmul(LC_mat, CL_msg)

    with tf.variable_scope('L_update') as scope:
        L_rands = tf.random_uniform([nlits,1])
        _, L_state = L_update(inputs=tf.concat([CL_msgs, notLit(L_state.h), L_rands], axis=1), state=L_state)

    return i+1, L_state, C_state

def loop_cond(i, L_state, C_state):
    return tf.less(i, nrounds_p)

def notLit(lits):
    return lits[::-1,...]

# pass messages
def loop_messages(L_state_in, C_state_in):
    _, L_state_out, C_state_out = tf.while_loop(loop_cond, loop_body, [0, L_state_in, C_state_in])
    return L_state_out, C_state_out

with tf.name_scope('messages') as scope:
    L_h = tf.tile(tf.div(L_init, math.sqrt(d)), [nlits, 1], name='L_h_init')
    C_h = tf.tile(tf.div(C_init, math.sqrt(d)), [nclauses, 1], name='C_h_init')

    L_state_in = tf.nn.rnn_cell.LSTMStateTuple(h=L_h, c=tf.zeros([nlits, d]))
    C_state_in = tf.nn.rnn_cell.LSTMStateTuple(h=C_h, c=tf.zeros([nclauses, d]))

    L_state_out, C_state_out = loop_messages(L_state_in, C_state_in)
    L_h, C_h = L_state_out.h, C_state_out.h

# compute logits
def compute_logits(L):
    with tf.name_scope('compute_logits') as scope:
        logits = L_logits.apply(L)
        net_logits = tf.add(logits,-notLit(logits))
    return logits, net_logits

logits, net_logits = compute_logits(L_h)

def compute_result(net_logits):
    boolean_values = tf.math.greater(net_logits, 0, name='boolean_values')
    clause_sat = tf.math.greater(tf.sparse_tensor_dense_matmul(LC_mat, tf.cast(boolean_values,tf.float32), adjoint_a=True), 0, name='clause_sat')
    num_sat = tf.reduce_sum(tf.cast(clause_sat,tf.float32), axis=0, name='num_sat')
    sat_fraction = tf.math.divide(num_sat, tf.cast(nclauses,tf.float32), name='sat_fraction')
    return boolean_values, clause_sat, num_sat, sat_fraction

boolean_values, clause_sat, num_sat, sat_fraction = compute_result(net_logits)

def compute_SAT_cost(net_logits, LC_matrix):
    clauses = tf.sparse_tensor_dense_matmul(LC_matrix, tf.math.exp(net_logits), adjoint_a=True)
    return tf.reduce_mean(tf.math.reciprocal(clauses), axis=0, name='SAT_cost')

def compute_label_cost(net_logits, solution):
    costs = tf.nn.sigmoid_cross_entropy_with_logits(logits=net_logits[:,0], labels=solution) #tf.cast(model, tf.float32))
    return tf.reduce_mean(costs, name='label_cost')

def compute_l2_cost():
    with tf.name_scope('regularization') as scope:
        l2_cost = tf.zeros([])
        for var in tf.trainable_variables():
            l2_cost += tf.nn.l2_loss(var)
    return l2_cost


label_cost = None
SAT_cost = None
l2_cost = None

def use_SAT_cost(l2=0.0):
    global SAT_cost, cost, l2_cost
    SAT_cost = compute_SAT_cost(net_logits, LC_mat)
    if l2 > 0:
        l2_cost = compute_l2_cost()
        cost = SAT_cost + l2 * l2_cost
    else: cost = SAT_cost

def use_label_cost(l2=0.0):
    global label_cost, cost, l2_cost, solution
    solution = tf.placeholder(tf.float32, shape=[None], name='solution')
    label_cost = compute_label_cost(net_logits, solution)
    if l2 > 0:
        l2_cost = compute_l2_cost()
        cost = label_cost + l2 * l2_cost
    else:
        cost = label_cost
        print("Hi! cost = ", end='')
        print(cost)

def use_SAT_and_label_cost(s=1.0, l=1.0, l2=0.0):
    global cost, cost2, SAT_cost, label_cost, l2_cost, solution
    solution = tf.placeholder(tf.float32, shape=[None], name='solution')
    SAT_cost = compute_SAT_cost(net_logits, LC_mat)
    label_cost= compute_label_cost(net_logits, solution)
    if l2 > 0:
        l2_cost = compute_l2_cost()
        cost = s * SAT_cost + l * label_cost + l2 * l2_cost
    else: cost = s * SAT_cost + l * label_cost

def init_all_variables():
    tf.global_variables_initializer().run(session=sess)

def guarantee_init_variables():
    scope = tf.get_variable_scope()
    #print("scope.name = " + scope.name)
    #print("scope.initializer = " + str(scope.initializer))
    scope._reuse = tf.AUTO_REUSE
    #print("scope.reuse = " + str(scope.reuse))

    uninitialized_variables = list(tf.get_variable(name) for name in
                                   sess.run(tf.report_uninitialized_variables()))
    sess.run(tf.initialize_variables(uninitialized_variables))

def feed_dict(LC_matrix, n_rounds=None, labels=None):
    d = {}
    d[LC_mat] = LC_matrix
    d[nlits] = LC_matrix.dense_shape[0]
    d[nvars] = LC_matrix.dense_shape[0] / 2
    d[nclauses] = LC_matrix.dense_shape[1]
    global nrounds
    if n_rounds is not None:
        nrounds = int(n_rounds)
    d[nrounds_p] = nrounds
    if labels is not None:
        d[solution] = labels
    return d












