import math
import matplotlib.pyplot as plt
import numpy as np
# ReLU activation function

def Sigmoid(z):
    """ Sigmoid function: squashes values to(0,1)range. Used for probability output."""
    return 1/(1+math.exp(-z))

def ReLU(z):
    """ ReLU function: outputs max(0,z). Used in hidden layers of deep neural networks."""
    return max(0,z)

def Marvellous_neuron_forward(inputs,weights,bias,activation_function):
    print("\n==================NEURON ACTIVATION START===================")
    print("Inputs (x):",inputs)
    print("Weights(w):",weights)
    print("Bias(b): ",bias)
    
    z = sum(w * x for w,x in zip(weights,inputs)) + bias
    print("\n Step 1: Calculate weighted sum ")
    print("z = ",z)
    
    y_hat = activation_function(z)
    print("\n Step 2: Apply Activation Function")
    print("Activation Function:",activation_function.__name__)
    print("Output(y = activation(z)):",y_hat)
    print("\n==================NEURON ACTIVATION END===================\n")
    
    return z,y_hat

def plot_activation_functions():
    z_values = np.linspace(-10,10,200)
    sigmoid_values = 1/(1+np.exp(-z_values))
    relu_values = np.maximum(0,z_values)
    
    plt.figure(figsize=(8,5))
    plt.plot(z_values,sigmoid_values,label="Sigmoid",linewidth=2,color="blue")
    plt.plot(z_values,relu_values,label="ReLU",linewidth=2,color="orange")
    
    plt.axhline(y=0,color="black",linewidth=0.5)
    plt.axhline(y=1,color="black",linewidth=0.5)
    plt.axhline(y=0,color="gray",linestyle="--")
    
    plt.title("Activation Functions: Sigmoid vs ReLU",fontsize=16)
    plt.xlabel("Summation(z)",fontsize=14)
    plt.ylabel("Activation Output",fontsize=14)
    plt.grid(True,linestyle="--",alpha=0.6)
    plt.legend()
    plt.show()
    
def main():
    inputs = [1.0,2.0,3.0]
    weights = [0.6,0.4,-0.2]
    bias = 0.5
    
    print("Testing Sigmoid Activation Function:")
    Marvellous_neuron_forward(inputs,weights,bias,Sigmoid)
    
    print("\nTesting ReLU Activation Function:")
    Marvellous_neuron_forward(inputs,weights,bias,ReLU)
    
    plot_activation_functions()
    
if __name__ =="__main__":
    main()