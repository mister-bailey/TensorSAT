TensorSAT - neural net for solving SAT problems
Michael Bailey 2019

This is a machine learning model for taking SAT problems (either from a file or randomly generated) and finding satisfying solutions (if they exist). It draws upon prior work by Selsam et al ("NeuroSAT", https://arxiv.org/abs/1802.03685). Major differences in TensorSAT include a different objective function (in fact, though both networks share a recurrent message-passing model, this program is solving a different problem, strictly speaking), random variables within the network, and a C++ extension which live-feeds randomized training data to the Tensorflow training step. The C++ extension runs in a parallel thread, and uses my custom upgrade of the existing SAT-solver MiniSAT.

***Setup***

You must first run "python setup.py install" in the RandSAT subdirectory, to compile and install the random problem generator extension. "train.py" and "solve.py" are the main executables. See --help for command line options.

***Tensorflow model***

We describe a model which, for a single set of parameters, can be used on a SAT problem with any number of variables and clauses. For our purposes, a SAT problem is described by a sparse matrix, whose rows correspond to the literals (positive and negative versions of each variable) and whose columns correspond to numbered clauses. A matrix entry i,j has value 1 if literal i is a member of clause j, and 0 otherwise.

Each clause and literal has an associated Long Short-Term Memory (LSTM), which is initialized according to model parameters, and which will be updated recurrently over the course of R rounds.

Each round, each literal and each clause will compute a "message" from its public state using a shared neural net (one for all literals, and one for all clauses), then each literal receives and sums the messages from all of its containing clauses, and likewise each clause receives and sums the messages from all of its member literals. Furthermore, each literal passes its public state separately to its negation-literal (to signify the pair's special relationship).

Each literal and each clause then passes its incoming data, along with a random variable, through another shared neural net (one for all literals, one for all clauses) and uses the output to update its associated LSTM. This ends the round.

After the specified number of rounds, each literal computes a single logit from its state, using a shared neural network. The difference of logits between L and (not L) is used to assign a final truth value to L.

***Objective function***

We have two options for the objective function. In practice, neither objective function on its own trained well, and a weighted sum of the two was best.

The first function is a measurement of how close the assignment is to being a satisfying solution. It takes the final logits and the clause-membership matrix, and computes in a differentiable way the number of clauses which fail to contain any true literal. This has the advantage of begin equally happy with any correct solution. However, it doesn't give the training a lot of guidance.

The second function compares the final computed logits to a provided satisfying asignment, using the sigmoid cross-entropy. This has the advantage of giving the training a lot of specific guidance. However, a problem may have many satisfying solutions, and I worried that always directing the training toward specific solutions (determined who-knows-how) might confuse its attempts to learn a general algorithm. In response, I modified the random-problem-generator to generate problems with very few correct solutions---typically 1 or 2.

In contrast: NeuroSAT (the prior work by others) computed just a single logit per SAT problem, predicting whether the problem was satisfiable or not. (When trained on nice random data, satisfying assignments could often be found implicitly by inspecting the LSTM state). I worked only with satisfiable problems, and while in principle this limits my model (it will just spin its wheels if given an UNSAT problem), in practice it is equal: NeuroSAT was also unable to report that a problem was UNSAT (except when trained on certain special problem classes), and under the hood would just keep searching for a solution. This is because UNSAT proofs are generally quite big, on contrast with SAT proofs, which are just satisfying assignments.

***Training data and random problem generation***

There are libraries of SAT problems available, following various random distributions. However, I wanted to tune the precise problem distribution, and I wanted an extremely large quantity of problems. Therefore, I wrote a C++ Python extension to randomly generate satisfiable problems (and their solutions) according to adjustable parameters.

For a problem with N variables, clauses are added one at a time. To create a clause, we randomly choose K distinct variables (K is typically 3 + a decaying random amount), randomly assign them positive or negative, and try adding the resulting clause to the problem. An existing SAT-solver tells us if the problem is still SAT. If it's still SAT, we keep the new clause and repeat, but if it's now UNSAT, we discard the new clause and try again with another clause. At each step, we eliminate from our pool of variables those whose assignment we know is fixed by the problem (which the solver tells us), and generate subsequent clauses only from undetermined variables. Once our pool of unfixed variables is smaller than our minimum clause size (which is usually 3), we stop.

For the SAT-solver, I used my custom variant of MiniSAT which performs better than existing variants on random problems.

***Multithreading***

We wrote a batch generator, which generates batches of problems according to some random distribution of problem sizes, up to a given total number of variables per batch. This runs in a parallel thread, leaving the generated batches in memory for retrieval by python and Tensorflow, so that problem generation can happen concurrently with training and not impede the training at all (on a multi-core cpu). With the improved MiniSAT, on the hardware I used, batches are generated faster than the GPU can train them.

***Regularization***

Standard L2 regularization can be added to the objective function, but with every training batch totally unique, there is no risk of overtraining, and I typically don't do this.

***Training***

We can choose the size of the LSTM state vectors, and the depth of the various neural nets. The constraints on this are mostly down to available GPU memory. Furthermore, we can choose the number of recurrence rounds to train on. Sequential rounds are not parallelizable, so training time scales with the number of rounds. 40 rounds get good results and doesn't try my patience too much.

I trained different parameter sets for different problem sizes, though, eg., the model trained on problem size 10 performed pretty well on size 20 and vice versa.

I trained using an Adam optimizer. The relative weights of the two objective functions were adjusted to keep them both at the same order of magnitude.

***Prediction***

While the model is trained at a fixed number of rounds (40), once it's trained it can run for as many rounds as you like. The prediction script keeps iterating rounds until a satisfying solution is found (or until a max number of rounds). Indeed, many problems which weren't solved with 40 rounds were solved in 50 or 60. In fact, most problems from the training distribution were solved in less than 40 rounds.



