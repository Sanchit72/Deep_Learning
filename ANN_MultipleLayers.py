#--------------------------------------------------------
#Network Structure:
#Input Layer : 2 inputs
#Hidden Layer: neurons with ReLU activation
#Output Layer: 1 Neuron with Sigmoid activation
#
#--------------------------------------------------------

import math

#--------------------------------------------------------
#Function Name : Marvellous_ReLU
#Description: Applies ReLU activation function
#Formula : ReLU(x) = max(0,x)
#Use : Commonly used in hidden layers
#--------------------------------------------------------

def Marvellous_ReLU(values):
    return max(0,values)

#--------------------------------------------------------
#Function name : Marvellous Sigmoid
#Description : Applies Sigmoid sctivation function
#Formula : 1/(1+e(-x))
#Use : Commonly used in output layer for binary classification
#Output Range : 0 to 1
#--------------------------------------------------------
def Marvellous_Sigmoid(values):
    return 1/(1+math.exp(-values))
#--------------------------------------------------------
#Function name : Marvellous_calculate_weighte_sum
#Description : Calculate weight sum of input
#Formula : z = (x1*w1 + x2*w2+...+xn*wn)+b
#Parameter: inputs :List of input values,weights:List of weights
#bias : bias value
#Return : Weights sum
#--------------------------------------------------------

def Marvellous_calculate_weighted_sum(inputs,weights,bias):
    weighted_sum = sum(weight * input_value for weight,input_value in zip(weights,inputs))+bias
    return weighted_sum

#--------------------------------------------------------
#Function name : Marvellous_DispalyMultiplication
#Description : Display step-by-step multiplication of inputs and weights for one neuron
##Parameter: inputs :List of input values,weights:List of weights
#--------------------------------------------------------

def Marvellous_Display_Multiplication(inputs,weights):
    print("Step 1 : Multiply inputs by corresponding weights ")
    for index in range(len(inputs)):
        print(f"({weights[index]}*{inputs[index]}) = {weights[index]*inputs[index]:.3f}")
        
        
def Marvellous_process_hidden_layer(inputs,hidden_weights,hidden_biases):
    hidden_output = []
    print("\n==================HIDDEN LAYER===================")
    
    for neuron_index in range(len(hidden_weights)):
        print(f"Hidden Neuron {neuron_index + 1}:")
        
        current_weights = hidden_weights[neuron_index]
        current_bias = hidden_biases[neuron_index]
        
        Marvellous_Display_Multiplication(inputs,current_weights)
        
        z_value = Marvellous_calculate_weighted_sum(inputs,current_weights,current_bias)
        print(f"Step 2 : Add all multiplication result and biaa{current_bias}")
        print(f" z ={z_value:.3f}")
        
        activated_output = Marvellous_ReLU(z_value)
        print(f"Step 3 : Apply ReLU activation")
        print(f"ReLU({z_value:.3f}) = {activated_output:.3f}\n")
        
        hidden_output.append(activated_output)
        
    return hidden_output

def Marvellous_Process_Output_Layer(hidden_output,output_weights,output_bias):
    print("\n=============== OUTPUT LAYER =================")
    print("OutPut Neuron:")
    print("Step 1 : Multiply hidden layer output by output weights")
    
    for index in range(len(hidden_output)):
        print(f"({output_weights[index]} * {hidden_output[index]:.3f}) = "
              f"{output_weights[index] * hidden_output[index]:.3f}")
        
        z_output = Marvellous_calculate_weighted_sum(hidden_output,output_weights,output_bias)
        print(f"Step 2 : Add all multiplication result and bias{output_bias}")
        print(f"z ={z_output:.3f}")
        
        final_output = Marvellous_Sigmoid(z_output)
        print("Step 3 : Apply Sigmoid activation")
        print(f"Sigmoid({z_output:.3f}) = {final_output:.3f}")
        
        
    return z_output,final_output
    
def Marvellous_Display_Network_Summary(hidden_output,final_output):
    print("\n================= Final SUMMARY =================")
    print(f"Hidden Layer Output: {hidden_output}")
    print(f"Final Network Output: {final_output:.3f}")
    print(f"Confidence Percentage: {final_output*100:.2f}%")
    
    if final_output >= 0.5:
        print("Prediction: Class 1 (Positive)")
    else:
        print("Prediction: Class 0 (Negative)")
        
def Marvellous_ANN_Forward_Pass(inputs):
    print("=============== INPUT LAYER =================")
    print(f"Input Values x1: {inputs[0]}")
    print(f"Input Values x2: {inputs[1]}")
    
    hidden_weights = [[0.5,-0.2],[0.8,0.4]]
    
    hidden_biases = [0.1,-0.1]
    
    output_weights = [1.0,-1.5]
    output_bias = 0.2
    
    hidden_output = Marvellous_process_hidden_layer(inputs,hidden_weights,hidden_biases)
    
    z_output,final_output = Marvellous_Process_Output_Layer(hidden_output,output_weights,output_bias)
    
    Marvellous_Display_Network_Summary(hidden_output,final_output)
    
def main():
    inputs = [2.0,3.0]
    Marvellous_ANN_Forward_Pass(inputs)
    
if __name__ == "__main__":
    main()
