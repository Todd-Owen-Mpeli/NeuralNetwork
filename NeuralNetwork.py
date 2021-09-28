# Import Libraries
import numpy as np

# Neural Network Class
class NeuralNetwork():

    # Stores a variable
    def __init__(self):
        
        # Generates Random numbers
        np.random.seed(1)
        
        self.synapticWeights = 2 * np.random.random((3, 1)) - 1

    #Sigmoid Normalizating Function 
    def sigmoid(self, x):
        # "x" will return any value between 0 & 1
        return 1 / (1 + np.exp(-x))

    #Sigmoid Derivative Function
    def sigmoidDerivative(self, x):
        # "x" will return a derivative value
        return x * (1 - x)

    # <!--===== TRAINING SECTION =====-->
    
    # Training function>
    def  training(self, trainingInputs, trainingOutputs, trainingIterations):
        
        for iteration in range(trainingIterations):

            output = self.think(trainingInputs)
            error = trainingOutputs - output
            adjustments = np.dot(trainingInputs.T, error * self.sigmoidDerivative(output))
            self.synapticWeights += adjustments

    # 
    def think(self, inputs):

        inputs = inputs.astype(float)
        output = self.sigmoid(np.dot(inputs, self.synapticWeights))

        return output

# Initializing the Neural Network like a function
if __name__ == "__main__":

    NeuralNetwork = NeuralNetwork()

    #Printing Synaptic Weights
    print("\n Random Starting Synaptic Weights: ")
    print(NeuralNetwork.synapticWeights)

    # <!--===== TRAINING DATA =====-->
    trainingInputs = np.array([[0,0,1],
                              [1,1,1],
                              [1,0,1],
                              [0,1,1]]);

    trainingOutputs = np.array([[0,1,1,0]]).T

    NeuralNetwork.training(trainingInputs, trainingOutputs, 10000)

    #Printing Synaptic Weights
    print("\n Synaptic Weights After Training: ")
    print(NeuralNetwork.synapticWeights)

    # <!--===== USER SECTION =====-->
    # Creating a user section to provide possible inputs

    userA = str(input("\n Input 1: "))
    userB = str(input("\n Input 2: "))
    userC = str(input("\n Input 3: "))

    print("\n New Situation: Input data =  ", userA, userB, userC)
    print("\n Output data: ")
    print(NeuralNetwork.think(np.array([userA, userB, userC])))


