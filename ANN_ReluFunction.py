#-------------------------------------------
# Program : Artificial Neuron with ReLU Activation
#Author : Sanchit Ashok Kale
#-------------------------------------------

import matplotlib.pyplot as plt
import numpy as np


#Step 1 : Activation Function(ReLU)
#ReLU = max(0,z)

def relu(z):
    return max(0,z)

# STEP 2 : Neuron Forward Pass Function

def Marvellous_neuron_forward(inputs,weights,bias):
    print("\n-------NEURON CALCULATION START--------\n")
    
    print("inputs(x) : ",inputs)
    print("weights (w) : ",weights)
    print("Bias(b) : ",bias)
    
#STEP 2.1 Weights sum calculation

    z = sum(w * x for w, x in zip(weights,inputs)) + bias
    print("\nStep 1 : Weighted Sum Calculation")
    print("z = w.x+b = ",z)

#STEP 2.2 Activation Function

    y_hat = relu(z)

    print("\nStep 2 : Activation Function Applied")
    print("Activation Function : ReLU")
    print("Output(y)=",y_hat)

    print("\n------ NEURON CALCULATION END --------\n")

    return z,y_hat

# STEP 3 : Plot ReLU Function

def plot_relu():
    
    z_values = np.linspace(-10,10,200)
    
    relu_values = np.maximum(0,z_values)
    
    plt.figure(figsize=(8,5))
    plt.plot(z_values,relu_values,label="ReLU Function",linewidth=2,color="green")
    plt.axhline(y=0,color="black",linewidth=0.5)
    plt.axvline(x=0,color="gray",linestyle="--")
    
    plt.title("ReLU Activation Function",fontsize=16)
    plt.xlabel("Input(z)",fontsize=14)
    plt.ylabel("Output",fontsize=14)
    
    plt.grid(True,linestyle="--",alpha=0.6) 
    plt.legend()
    
    plt.show()
    
#STEP 4: Main Function

def main():
    
    print("\n=========== NEURON DEMO ============")
    
    inputs = [1.0,2.0,3.0]
    
    weights = [0.6,0.4,-0.2]
    
    bias = 0.5
    
    z,y_hat = Marvellous_neuron_forward(inputs,weights,bias)
    
    plot_relu()

if __name__ == "__main__":
    main()


























