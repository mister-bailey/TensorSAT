import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # or any {'0', '1', '2', '3'}
import tensorflow.compat.v1 as tf
tf.logging.set_verbosity(tf.logging.ERROR)
import numpy as np
import TensorSAT
import time
import argparse
from math import sqrt

# defaults
batch_size=100
batches_per_epoch = 2
min_vars=10
var_vars=0
learning_rate = .000001
l2_weight=0.0 #0.00000001
SAT_weight = 10
test_every = 1
repeat_batch = 10

# parsing arguments
parser = argparse.ArgumentParser()
parser.add_argument('-td', '--train_dir', action='store', type=str, default='train', help='Directory with training/test problems')
parser.add_argument('-tf', '--train_from_files', action='store_true', help='Train from saved data, rather than dynamically generated (default)')
parser.add_argument('-bpe', '--batches_per_epoch', action='store', type=int, default=batches_per_epoch, help='Batches to run before incrementing epoch and computing stats')
parser.add_argument('-bs', '--batch_size', action='store', type=int, default=batch_size, help='Size (in variables) of random batches')
parser.add_argument('-mv', '--min_vars', action='store', type=int, default=min_vars, help='Size (in variables) of random subproblems')
parser.add_argument('-vv', '--var_vars', action='store', type=int, default=var_vars, help='Random variation in subproblem size')
parser.add_argument('-te', '--test_every', action='store', type=int, default=test_every, help='Run a standard test every N epochs (0 = no test)')
parser.add_argument('-ts', '--train_from_solution', action='store_true', help='Train from labelled solutions')
parser.add_argument('-wb', '--wait_for_batch', action='store_true', help='Wait for batch (instead of repeat) if new one not ready yet')
parser.add_argument('-rb', '--repeat_batch', action='store', type=int, default=repeat_batch, help='Reuse each batch for N training steps')
parser.add_argument('-sw', '--SAT_weight', action='store', type=float, default = SAT_weight, help="Weight to give SAT cost if training from labelled solutions")
parser.add_argument('-ns', '--no_save', action='store_true', help='Don\'t save model while training')
parser.add_argument('-new','--new_model', action='store_true', help='Initialize model randomly, rather than resume previous model')
parser.add_argument('-md', '--model_dir', action='store', type=str, default='params', help='Directory to save model parameters')
parser.add_argument('-r', '--rounds', action='store', type=int, default=40, help='Number of recurrence rounds to train on')
parser.add_argument('-e', '--epochs', action='store', type=int, default=100, help='Number of training epochs')
parser.add_argument('-lrd', '--learning_rate_decay', action='store', type=float, default=1.0, help='Decay of learning rate per epoch')
parser.add_argument('-lr', '--learning_rate', action='store', type=float, default=learning_rate, help='(Initial) learning rate')
parser.add_argument('-l2', '--l2_weight', action='store', type=float, default=l2_weight, help='L2 regularization coefficient')


options = parser.parse_args()

batches_per_epoch = options.batches_per_epoch
batch_size=options.batch_size
min_vars= options.min_vars
var_vars= options.var_vars
lr_start = options.learning_rate
l2_weight= options.l2_weight
test_every = options.test_every
repeat_batch = options.repeat_batch
n_epochs = options.epochs


saver = tf.train.Saver()
save_path = options.model_dir + '/snapshot'

def save():
    saver.save(TensorSAT.sess, save_path, global_step = epoch)

def restore(epoch):
    saver.restore(TensorSAT.sess, save_path + '-%d' % epoch)

if options.train_from_files:
    # loading files
    import load
    train_dir = options.train_dir
    print("Loading files from %s/ ..." % train_dir)
    loader = load.Loader(train_dir, batches_per_epoch)
    batches_per_epoch = loader.num_batches
    TensorSAT.use_SAT_cost(l2_weight)
    options.train_from_solution = False
else:
    # creating random training generator
    import PyRandSAT as rs
    print("Creating random problem generator...")
    bg = rs.BatchGenerator(batch_size,options.min_vars,var_vars=options.var_vars,model=options.train_from_solution)
    if options.train_from_solution:
        if options.SAT_weight != 0.0:
            d = 1.0 / sqrt(1.0 + options.SAT_weight * options.SAT_weight)
            sw = d * options.SAT_weight
            lw = d
            TensorSAT.use_SAT_and_label_cost(sw,lw,l2_weight)
        else: TensorSAT.use_label_cost(l2_weight)
    else:
        TensorSAT.use_SAT_cost(l2_weight)
        options.SAT_weight = 0
    clauses, membership = bg.getClauseMembership()
    print("First batch has %d clauses and %d membership" % (clauses, membership))

# building optimizer
epoch_var = tf.get_variable("epoch_var", shape=[], initializer=tf.zeros_initializer(), trainable=False)

decay_rate = options.learning_rate_decay # .99
grad_clip = 5

learning_rate = tf.train.exponential_decay(lr_start, epoch_var, batches_per_epoch, decay_rate, staircase=False)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

gradients, variables = zip(*optimizer.compute_gradients(TensorSAT.cost))
gradients, _ = tf.clip_by_global_norm(gradients, grad_clip)
apply_gradients = optimizer.apply_gradients(zip(gradients, variables), name='apply_gradients', global_step=epoch_var)

TensorSAT.init_all_variables()

if not options.new_model:
    checkpoint = tf.train.latest_checkpoint(options.model_dir)
    if checkpoint is not None:
        print('Loading existing model from %s...' % checkpoint)
        tf.get_variable_scope()._reuse = tf.AUTO_REUSE
        vs = [tf.get_variable(v[0]) for v in tf.train.list_variables(checkpoint)]
        tf.train.Saver(vs).restore(TensorSAT.sess, checkpoint)

#TensorSAT.guarantee_init_variables()


fcosts = [TensorSAT.label_cost, TensorSAT.SAT_cost, TensorSAT.l2_cost]
fcosts = [c for c in fcosts if c is not None]
if len(fcosts) == 1: fcosts = fcosts * 2
else: fcosts = fcosts[0:2]

batch_num = 0
LC_matrix = None
solution = None

def train_epoch(epoch):
    global LC_matrix
    global solution
    global batch_num
    start = time.process_time()
    
    costs = []
    costs2 = []
    sat_fractions = []

    for i in range(batches_per_epoch):
        if (batch_num % repeat_batch == 0 and (options.wait_for_batch or bg.batchReady())) or LC_matrix is None:
            if options.train_from_files: LC_matrix = loader.get_batch(batch_num)
            else: LC_matrix = bg.getNextBatch()
            if options.train_from_solution: LC_matrix, solution = LC_matrix
        fd = TensorSAT.feed_dict(LC_matrix, labels = solution)
        _, cost, cost2, sat_fraction = TensorSAT.sess.run([apply_gradients,*fcosts,TensorSAT.sat_fraction], feed_dict=fd)
        costs.append(cost)
        costs2.append(cost2)
        sat_fractions.append(sat_fraction)
        batch_num = batch_num + 1

    cost = sum(costs) / len(costs)
    cost2 = sum(costs2) / len(costs2)
    sat_fraction = sum(sat_fractions) / len(sat_fractions)
    return cost, cost2, sat_fraction, time.process_time() - start

print("Training %d epochs..." % n_epochs)
print("┌───────────────<Training>──────────────┐  ┌──────<Test>─────┐")
print("│Epoch| Time |  Cost  | Cost 2 |  Pct   │  │  Cost  |  Pct   │")
print("├─────┼──────┼────────┼────────┼────────┤  ├────────┼────────┤")


test_batch = test_sol =  None

for epoch in range(n_epochs):
    cost, cost2, sat_fraction, duration = train_epoch(epoch)
    print("│%4d │%5.2fs│ %6.4f │ %6.4f │%7.3f │  │" % (epoch, duration, cost, cost2, sat_fraction * 100), end='')
    if test_every > 0 and (epoch % test_every == 0):
        if test_batch is None:
            if options.train_from_files:
                fd = TensorSAT.feed_dict(loader.get_test())
            else:
                test_batch = bg.getNextBatch()
                if options.train_from_solution:
                    test_batch, test_sol = test_batch
                else: test_sol = None
        fd = TensorSAT.feed_dict(test_batch, labels=test_sol)
        cost, sat_fraction = TensorSAT.sess.run([TensorSAT.cost, TensorSAT.sat_fraction], feed_dict=fd)
        print(' %6.4f │%7.3f │' % (cost, sat_fraction * 100))

    else: print('        │        │')

    if (epoch % 5 == 0) or (epoch + 1 == n_epochs):
        if not options.no_save: save()

    


