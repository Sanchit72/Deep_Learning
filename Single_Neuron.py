import numpy as np

# STEP 1 : Define input Feature   

inputs = np.array([2.0,3.0,4.0])

#STEP 2 : Define Weights  

weights = np.array([0.5,0.3,0.2])

#STEP 3 : Define Bias

bias = 1.0

#STEP 4 : Calculate Weighted Sum(Z)

weighted_sum = np.dot(inputs,weights)+bias

#STEP 5 : Activation Function(ReLU)


def relu(x):
    return max(0,x)

#STEP 6 : Final Output

output = relu(weighted_sum)

#STEP 7: Dispaly Result

print("Inputs :",inputs)
print("Weights :",weights)
print("Bias :",bias)
print("Weighted Sum(Z) :",weighted_sum)
print("Final Output :",output)