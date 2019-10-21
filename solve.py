import tensorflow.compat.v1 as tf
import numpy as np
import math
import TensorSAT as ts
import argparse
import os
import load

# parsing arguments
parser = argparse.ArgumentParser()
parser.add_argument('problem_file', action='store', type=str, nargs='?', help='DIMACS conjunctive normal form file to solve')
parser.add_argument('-r', '--rounds', action='store', type=int, default=100, help='Max number of recurrence rounds to run')
parser.add_argument('-rp', '--random', action='store', type=int, default=0, help='Solve random problem with ARG variables')
parser.add_argument('-md', '--model_dir', action='store', type=str, default='params', help='Directory with saved model parameters')

options = parser.parse_args()

# loading model
ts.init_all_variables()
checkpoint = tf.train.latest_checkpoint(options.model_dir)
print('Loading model from %s...' % checkpoint)
tf.train.Saver().restore(ts.sess, checkpoint)

# loading problem
if options.random == 0:
    print('Loading problem from %s...' % options.problem_file)
    LC_mat, _ = load.load_file(options.problem_file)
else:
    print('Generating random problem...')
    import PyRandSAT as rs
    LC_mat = rs.getProblem(options.random)

nvars = int(LC_mat.dense_shape[0] / 2)
nclauses = LC_mat.dense_shape[1]
print('%d variables, %d clauses, %d membership' % (nvars, nclauses, LC_mat.indices.shape[0]))



maxrounds = options.rounds
roundsbatch = 1
nruns = math.ceil(maxrounds / roundsbatch)


# initialize data before loops
L_state, C_state = ts.sess.run([ts.L_state_in, ts.C_state_in], feed_dict=ts.feed_dict(LC_mat))

# create message loop
L_h_in = tf.placeholder(tf.float32, [nvars * 2, ts.d])
L_c_in = tf.placeholder(tf.float32, [nvars * 2, ts.d])
C_h_in = tf.placeholder(tf.float32, [nclauses, ts.d])
C_c_in = tf.placeholder(tf.float32, [nclauses, ts.d])
L_state_in = tf.nn.rnn_cell.LSTMStateTuple(h=L_h_in, c=L_c_in)
C_state_in = tf.nn.rnn_cell.LSTMStateTuple(h=C_h_in, c=C_c_in)

L_state_out, C_state_out = ts.loop_messages(L_state_in, C_state_in)
#L_state_out = tf.identity(L_state_out, name='L_state_out')
#C_state_out = tf.identity(C_state_out, name='C_state_out')

# compute results up to this point
boolean_values, clause_sat, num_sat, sat_fraction = ts.compute_result(ts.compute_logits(L_state_out.h)[1])

def iter_feed_dict(LC_mat, L_in, C_in, nrounds=None):
    fd = ts.feed_dict(LC_mat, n_rounds = nrounds)
    fd[L_h_in] = L_in.h
    fd[L_c_in] = L_in.c
    fd[C_h_in] = C_in.h
    fd[C_c_in] = C_in.c
    return fd

for i in range(nruns):
    fd = iter_feed_dict(LC_mat, L_state, C_state, nrounds=roundsbatch)
    values, num, fraction, L_state, C_state = ts.sess.run([boolean_values, num_sat, sat_fraction,L_state_out,C_state_out],feed_dict=fd)
    print ('%4d rounds---accuracy: %3.3f %%  (%d UNSAT out of %d clauses) ' % ((i+1) * roundsbatch, 100 * fraction, nclauses - num, nclauses))
    if nclauses == num:
        break

