# Baum-Welch-Algorithm
Numpy Implementation of Baum-Welch (Forward-Backward) algorithm in Python.

This algorithm can run for any number of states and observations. The default example has two states (H&C) and three possible observations (emissions) namely 1, 2 and 3. Following are the matrices/variables that needs to be adjusted:

1. **Transition:** contains intial transition probabilies. The value at [0,0] is transition from H to H and value at [1,1] is transition                      from C to C. (keeping in view the default example)
2. **emission:**  contains intial emission probabilies. The value at [0,0] is emission of symbol 1 from H and the value at [0,1] is                         emission of symbol 2 from H. (keeping in view the default example)
3. **states_dic:** the dictionary that contains corresponding digits/indices for each state. The value starts from 0 and goes upto number                    of states for each state
4. **sequence_syms:** simply a dictionary of all possible observations with their corresponding indice. Indice start from 0 and increments                       with each observation
5. **sequence:** is a list of all possible observations
6. **test_sequence:** a string containing the sequence on which we want to train our matrices
7. **threshold value** that stops the algorithm can be adjusted in the very end of code
