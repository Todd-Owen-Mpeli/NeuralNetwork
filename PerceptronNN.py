# Import Libraries
import numpy as np

#Sigmoid Normalizating Function 
def sigmoid(x):
    # "x" will return any value between 0 & 1
    return 1 / (1 + np.exp(-x))


#Sigmoid Derivative Function
def sigmoidDerivative(x):
    # "x" will return a derivative value
    return x * (1 - x)

# <!--===== TRAINING SECTION =====-->
trainingInputs = np.array([[0,0,1],
                          [1,1,1],
                          [1,0,1],
                          [0,1,1]]);

trainingOutputs= np.array([[0,1,1,0]]).T

# Generates Random numbers
np.random.seed(1)

# Weight Calculation
synapticWeights = 2 * np.random.random((3, 1)) - 1

#Printing Synaptic Weights for verification
print("\n Random Starting Synaptic Weights: ")
print(synapticWeights)

# <!--===== MAIN FOR-LOOP SECTION =====-->
for iteration in range(20000):

    inputLayer = trainingInputs

    output = sigmoid(np.dot(inputLayer,synapticWeights))

# <!--===== BACK PROPERGATION LOOP SECTION =====-->
# Error Weighted Derivative
error = trainingOutputs - output

adjustments = error * sigmoidDerivative(output)

synapticWeights += np.dot(inputLayer.T, adjustments)

#Printing Synaptic Weights
print("\n Synaptic Weights After Training: ")
print(synapticWeights)

print("\n Outputs After Training: ")
print(output)
