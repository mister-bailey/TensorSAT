import numpy as np
import randSAT

# If we're being run as a main script, don't bother to import
# Tensorflow, and just return batches as raw data.
# If we're being imported for use elsewhere, import TF and
# package batches as tf.SparseTensorValue
if __name__ != "__main__":
    import tensorflow.compat.v1 as tf
    using_tf = True
else:
    using_tf = False

#print("Starting...")
#entries, nlits, nclauses = randSAT.get_problem(num_vars = 6, base_lits_pc = 3, var_lits_pc = 2, verbosity = 1)
#prob = tf.SparseTensorValue(entries, np.ones(entries.shape[0], dtype = np.int64), [nlits,nclauses])

def getProblem(num_vars, base_lits_pc = 3, var_lits_pc = 2, verbosity = 0):
    entries, nlits, nclauses = randSAT.get_problem(num_vars, base_lits_pc, var_lits_pc, verbosity = verbosity)
    return tf.SparseTensorValue(entries, np.ones(entries.shape[0], dtype = np.int64), [nlits,nclauses])

class BatchGenerator:
    waitingBatch = None

    def __init__(self, batch_size, min_vars, max_vars = 0, var_vars = 0, lits = 3, lits_var = 2.0, model = False, batches_ahead = 2, go = True):
        self.batch_size = batch_size # in number of variables
        self.min_vars = min_vars
        if max_vars < min_vars: max_vars = min_vars + var_vars
        self.max_vars = max_vars
        self.var_vars = var_vars
        self.lits = lits
        self.lits_var = lits_var
        self.batches_ahead = batches_ahead
        self.model = model
        self.capsule = randSAT.newBatchGenerator(batch_size, min_vars, max_vars, var_vars, lits, lits_var, model, batches_ahead, go)

    def getNextBatch(self):
        if self.waitingBatch is not None:
            wb = self.waitingBatch
            self.waitingBatch = None
            return wb
        if self.model:
            entries, nlits, nclauses, model = randSAT.getNextBatchAndModel(self.capsule)
            model = model.astype(np.float32)
            model = np.concatenate((model,-model[::-1] + 1))
            if using_tf:
                return tf.SparseTensorValue(entries, np.ones(entries.shape[0], dtype = np.int64), [nlits,nclauses]), model
            else:
                return (entries, [nlits,nclauses], model)
        else:
            entries, nlits, nclauses = randSAT.getNextBatch(self.capsule)
            if using_tf:
                return tf.SparseTensorValue(entries, np.ones(entries.shape[0], dtype = np.int64), [nlits,nclauses])
            else:
                return (entries, [nlits,nclauses])



    def batchReady(self):
        return self.waitingBatch is not None or randSAT.batchReady(self.capsule)
    
    def changeParameters(self, batch_size = -1, min_vars = -1, max_vars = -1, var_vars = -1, lits = -1, lits_var = -1, use_old = False):
        if batch_size > 0: self.batch_size = batch_size
        if min_vars > 0: self.min_vars = min_vars
        if max_vars > 0: self.max_vars = max_vars
        if var_vars > 0: self.var_vars = var_vars
        if lits > 0: self.lits = lits
        if lits_var > 0: self.lits_var = lits_var
        self.waitingBatch = None
        randSAT.changeParameters(self.capsule, batch_size, min_vars, max_vars, var_vars, lits, lits_var, use_old)

    def getClauseMembership(self):
        """Gets the number of clauses, and membership (number of clause membership relations)
        for a random batch. Most things that scale, like ideal learning rate,
        scale with this value."""
        if self.model:
            wb, _ = self.waitingBatch = self.getNextBatch()
        else:
            wb = self.waitingBatch = self.getNextBatch()
        return (wb.dense_shape[1] if using_tf else wb[1][1]), wb[0].shape[0]
        # Gets a new batch to sample, every time this is called,
        # but doesn't waist the batch, keeping it around for the
        # next getNextBatch()




if __name__ == "__main__":
    import os
    import sys
    def pause():
        if sys.platform.startswith('win'):
            os.system('pause')
        else:
            os.system('read -s -n 1 -p "Press any key to continue..."')
             
    pause()
    #print("Trying one problem...")
    #print(randSAT.get_problem(num_vars = 6, base_lits_pc = 3, var_lits_pc = 2, verbosity = 1))
    #os.system('pause')
    bg = BatchGenerator(50,50,50,0,model=True)
    pause()
    while True:
        print("Getting batch...")
        batch = bg.getNextBatch()
        pause()
    print("BatchReady() = %r" % bg.batchReady())
    print("Getting first batch...")
    batch1 = bg.getNextBatch()
    print(batch1)
    pause()
    print("BatchReady() = %r" % bg.batchReady())
    print("Getting second batch...")
    batch2 = bg.getNextBatch()
    print(batch2)
    pause()
    print("BatchReady() = %r" % bg.batchReady())
    print("Getting third batch...")
    batch3 = bg.getNextBatch()
    print(batch3)
    pause()

    
