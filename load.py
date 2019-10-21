import numpy as np
import tensorflow.compat.v1 as tf
import os
import re

def parse(s, sort=True): # s -- string
    """DIMACS parsing code"""
    nvars = 0
    nclauses = 0
    inds = [] #np.zeros([0,2],dtype=np.int)
    sol = None

    lines = s.split('\n')

    pComment = re.compile(r'c.*')
    pStats = re.compile(r'p\s*cnf\s*(\d*)\s*(\d*)')
    pSat = re.compile(r's\s*(\w*)')
    pVal = re.compile(r'v\s*(.*)')
    
    c = 0
    while len(lines) > 0:
        line = lines.pop(0)

        # Only deal with lines that aren't comments
        if pComment.match(line):
            continue
            
        m = pStats.match(line)
        if m:
            nvars = int(m[1])
            nclauses = int(m[2])
            continue

        if sol is None:
            m = pSat.match(line)
            if m and m[0] == 'SATISFIABLE':
                sol = np.zeros([nvars * 2 +1],type=np.float32)
                continue

        m = pVal.match(line)
        if m:
            nums = m[0].split(' ')
            for lit_str in nums:
                if lit_str != '' and int(lit_str) != 0:
                    sol[int(lit_str),c] = 1
            continue



        nums = line.rstrip('\n').split(' ')
        nonempty = False
        for lit_str in nums:
            if lit_str != '':
                try:
                    i = int(lit_str)
                except:
                    continue
                if i == 0:
                    continue
                if i < 0:
                    i += 2 * nvars
                else:
                    i -= 1
                inds.append([i,c])
                nonempty = True

        if nonempty:
            c = c + 1

    vals = np.ones([len(inds)], dtype=np.float32)
    cnf = tf.SparseTensorValue(indices = np.array(inds,dtype=np.int64), values = vals, dense_shape=[nvars * 2, nclauses])
    if sort:
        return batch_problems([(cnf,sol)])
    else:
        return cnf, sol

def batch_problems(problems):
    """Combines multiple problems into 1 big batch
        Since cnf is sparse, no wasted memory"""
    cnfs, sols = zip(*problems)
    nvars = int(sum([cnf.dense_shape[0] / 2 for cnf in cnfs]))
    nclauses = sum([cnf.dense_shape[1] for cnf in cnfs])

    if sols[0] is not None:
        sols = np.zeros((1), dtype=np.float32)
    else:
        sols = None

    vars_sofar = 0
    clauses_sofar = 0
    #inds = np.zeros([0,2], dtype=np.int64)
    inds0 = np.zeros([0], dtype=np.int64)
    inds1 = np.zeros([0], dtype=np.int64)

    for cnf, sol in problems:

        cnvars = int(cnf.dense_shape[0]) / 2 # number of vars in this problem
        cnclauses = cnf.dense_shape[1]        # number of clauses in this problem
        ind = np.array(cnf.indices, copy=False)               # index list from cnf sparse representation

        lit_nums = ind[:,0]

        # making signed indices:
        lit_nums[lit_nums >= cnvars] -= int(2 * cnvars + vars_sofar)
        lit_nums[lit_nums >= 0] += int(1 + vars_sofar)

        #ind = np.stack((lit_nums, ind[:,1] + clauses_sofar ), axis = 1) # new index list to concatenate
        #inds = np.concatenate((inds, ind), axis=0) # accumulated index list

        inds0 = np.concatenate((inds0, lit_nums), axis=0) # accumulated (signed) literal numbers
        inds1 = np.concatenate((inds1, ind[:,1] + int(clauses_sofar)), axis=0) # accumulated clause numbers
        
        vars_sofar += cnvars
        clauses_sofar += cnclauses

        if sols is not None:
            if sol is None:
                raise Exception("Some problems have solutions given, but others in same batch don't!")
            sols = np.concatenate((sol[-cnvars:0],sols,sol[1:cnvars+1]))

    #if sols is not None:
    #    assert sols.shape[0] == inds0.shape[0]

    # Making indices positive:
    inds0[inds0 > 0] -= 1
    inds0[inds0 < 0] += int(2 * nvars)

    inds = np.stack((inds0,inds1), axis=1)

    inds = inds[np.lexsort(inds[:,::-1].T),:]
    return tf.SparseTensorValue(indices=inds, values = np.ones(inds.shape[0], dtype=np.float32), dense_shape=[nvars * 2, nclauses]), sols

def load_file(location, sort=True):
    """Loads a CNF from a DIMACS file."""
    with open(location) as f:
        s = f.read()
    return parse(s, sort=sort)


class Loader(object):

    def __init__(self, location, batchsize=-1, batches_per_epoch=1, trainset=.7, testset=.3, loadnow=True):
        self.location = location
        filenames = [file for file in os.listdir(location) if '.cnf' in file]
        self.num_files = len(filenames)
        self.num_train = int(self.num_files * trainset)
        self.num_test = self.num_files - self.num_train
        self.train_files = filenames[:self.num_train]
        self.test_files = filenames[self.num_train:]
        self.test = None

        if batch_size > 0:
            self.batchsize = batchsize
        else:
            self.batchsize = self.num_train / batches_per_epoch
        self.batch_files = [self.train_files[x:x+self.batchsize] for x in range(0,self.num_train,self.batchsize)]
        self.num_batches = len(self.batch_files)
        self.batches = [None] * self.num_batches

        self.batchpos = 0

        if loadnow:
            self.load_batches()
            self.load_test()



    def load_batch(self, batchnum=-1):
        """Loads CNFs in the given directory, batches them together via batch_problems"""
        if batchnum < 0:
            batchnum = self.batchpos
        problems = []
        for f in self.batch_files[batchnum]:
            problems.append(load_file(self.location + '/' + f, sort=False))
        self.batches[batchnum] = batch_problems(problems)[0]
        return self.batches[batchnum]

    def load_batches(self, start=0,end=-1):
        end = self.num_batches if end < 0 else end
        for i in range(start,end):
            if self.batches[i] is None:
                self.load_batch(i)

    def load_test(self):
        tests = []
        for f in self.test_files:
            tests.append(load_file(self.location + '/' + f, sort=False))
        self.test = batch_problems(tests)[0]
        return self.test

    def get_test(self):
        if self.test is None:
            self.load_test()
        return self.test

    def get_batch(self, batch_num):
        i = batch_num % self.num_batches
        if self.batches[i] is None:
            self.load_batch(i)
        return self.batches[i]

    def get_batches(self):
        return (self.get_batch(i) for i in range(self.num_batches))



