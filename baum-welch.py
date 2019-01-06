import numpy as np


#generating initial probabilities

#transition probabilities
transition = np.array([[0.8,0.1],
                       [0.1,0.8]])
#Emission probabilities
emission = np.array([[0.1,0.2,0.7],
                     [0.7,0.2,0.1]])

#defining states and sequence symbols
states = ['H','C']
states_dic = {'H':0, 'C':1}
sequence_syms = {'1':0,'2':1,'3':2}
sequence = ['1','2','3']

#test sequence
test_sequence = '331122313'
test_sequence = [x for x in test_sequence]

#probabilities of going to end state
end_probs = [0.1, 0.1]
#probabilities of going from start state
start_probs = [0.5, 0.5]


#function to find forward probabilities
def forward_probs():
    # node values stored during forward algorithm
    node_values_fwd = np.zeros((len(states), len(test_sequence)))

    for i, sequence_val in enumerate(test_sequence):
        for j in range(len(states)):
            # if first sequence value then do this
            if (i == 0):
                node_values_fwd[j, i] = start_probs[j] * emission[j, sequence_syms[sequence_val]]
            # else perform this
            else:
                values = [node_values_fwd[k, i - 1] * emission[j, sequence_syms[sequence_val]] * transition[k, j] for k in
                          range(len(states))]
                node_values_fwd[j, i] = sum(values)

    #end state value
    end_state = np.multiply(node_values_fwd[:, -1], end_probs)
    end_state_val = sum(end_state)
    return node_values_fwd, end_state_val



#function to find backward probabilities
def backward_probs():
    # node values stored during forward algorithm
    node_values_bwd = np.zeros((len(states), len(test_sequence)))

    #for i, sequence_val in enumerate(test_sequence):
    for i in range(1,len(test_sequence)+1):
        for j in range(len(states)):
            # if first sequence value then do this
            if (-i == -1):
                node_values_bwd[j, -i] = end_probs[j]
            # else perform this
            else:
                values = [node_values_bwd[k, -i+1] * emission[k, sequence_syms[test_sequence[-i+1]]] * transition[j, k] for k in range(len(states))]
                node_values_bwd[j, -i] = sum(values)

    #start state value
    start_state = [node_values_bwd[m,0] * emission[m, sequence_syms[test_sequence[0]]] for m in range(len(states))]
    start_state = np.multiply(start_state, start_probs)
    start_state_val = sum(start_state)
    return node_values_bwd, start_state_val


#function to find si probabilities
def si_probs(forward, backward, forward_val):

    si_probabilities = np.zeros((len(states), len(test_sequence)-1, len(states)))

    for i in range(len(test_sequence)-1):
        for j in range(len(states)):
            for k in range(len(states)):
                si_probabilities[j,i,k] = ( forward[j,i] * backward[k,i+1] * transition[j,k] * emission[k,sequence_syms[test_sequence[i+1]]] ) \
                                                    / forward_val
    return si_probabilities

#function to find gamma probabilities
def gamma_probs(forward, backward, forward_val):

    gamma_probabilities = np.zeros((len(states), len(test_sequence)))

    for i in range(len(test_sequence)):
        for j in range(len(states)):
            #gamma_probabilities[j,i] = ( forward[j,i] * backward[j,i] * emission[j,sequence_syms[test_sequence[i]]] ) / forward_val
            gamma_probabilities[j, i] = (forward[j, i] * backward[j, i]) / forward_val

    return gamma_probabilities



#performing iterations until convergence

for iteration in range(2000):

    print('\nIteration No: ', iteration + 1)
    # print('\nTransition:\n ', transition)
    # print('\nEmission: \n', emission)

    #Calling probability functions to calculate all probabilities
    fwd_probs, fwd_val = forward_probs()
    bwd_probs, bwd_val = backward_probs()
    si_probabilities = si_probs(fwd_probs, bwd_probs, fwd_val)
    gamma_probabilities = gamma_probs(fwd_probs, bwd_probs, fwd_val)

    # print('Forward Probs:')
    # print(np.matrix(fwd_probs))
    #
    # print('Backward Probs:')
    # print(np.matrix(bwd_probs))
    #
    # print('Si Probs:')
    # print(si_probabilities)
    #
    # print('Gamma Probs:')
    # print(np.matrix(gamma_probabilities))

    #caclculating 'a' and 'b' matrices
    a = np.zeros((len(states), len(states)))
    b = np.zeros((len(states), len(sequence_syms)))

    #'a' matrix
    for j in range(len(states)):
        for i in range(len(states)):
            for t in range(len(test_sequence)-1):
                a[j,i] = a[j,i] + si_probabilities[j,t,i]

            denomenator_a = [si_probabilities[j, t_x, i_x] for t_x in range(len(test_sequence) - 1) for i_x in range(len(states))]
            denomenator_a = sum(denomenator_a)

            if (denomenator_a == 0):
                a[j,i] = 0
            else:
                a[j,i] = a[j,i]/denomenator_a

    #'b' matrix
    for j in range(len(states)): #states
        for i in range(len(sequence)): #seq
            indices = [idx for idx, val in enumerate(test_sequence) if val == sequence[i]]
            numerator_b = sum( gamma_probabilities[j,indices] )
            denomenator_b = sum( gamma_probabilities[j,:] )

            if (denomenator_b == 0):
                b[j,i] = 0
            else:
                b[j, i] = numerator_b / denomenator_b


    print('\nMatrix a:\n')
    print(np.matrix(a.round(decimals=4)))
    print('\nMatrix b:\n')
    print(np.matrix(b.round(decimals=4)))

    transition = a
    emission = b

    new_fwd_temp, new_fwd_temp_val = forward_probs()
    print('New forward probability: ', new_fwd_temp_val)
    diff =  np.abs(fwd_val - new_fwd_temp_val)
    print('Difference in forward probability: ', diff)

    if (diff < 0.0000001):
        break


c = 1