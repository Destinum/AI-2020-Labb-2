import numpy as np
import copy
import random

class NeuralNetwork():
    def __init__(self, StartingLocation, Destination, Map, WeightMatrix, Biases):
        self.StartingLocation = StartingLocation
        self.CurrentLocation = [StartingLocation[0], StartingLocation[1]]
        self.Destination = Destination
        self.Path = [(StartingLocation[0], StartingLocation[1])]
        self.Map = copy.deepcopy(Map)
        self.WeightMatrix = copy.deepcopy(WeightMatrix)
        self.Biases = copy.deepcopy(Biases)

    LengthOfPath = 0
    Outputs = []

    def ReturnFunction(self):
        ReturnValue = [self.DistanceToGoal(self.CurrentLocation[0], self.CurrentLocation[1]), self.LengthOfPath]
        print(ReturnValue)
        self.Outputs = ReturnValue
        return ReturnValue

    def Sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def Network(self, inputs):
        HiddenLayers = copy.deepcopy(self.Biases)

        for Neuron in range(len(self.Biases[0])):
            for Input in range(len(inputs)):
                HiddenLayers[0][Neuron] = inputs[Input] * self.WeightMatrix[0][Neuron][Input] + self.Biases[0][Neuron]

        NumberOfLayers = len(self.Biases)
        for Layer in range(1, NumberOfLayers):
            for Neuron in range(len(self.Biases[Layer])):
                for Weight in range(len(self.Biases[Layer - 1])):
                    HiddenLayers[Layer][Neuron] = HiddenLayers[Layer - 1][Neuron] * self.WeightMatrix[Layer][Neuron][Weight] + self.Biases[Layer][Neuron]

        #return([HiddenLayers[NumberOfLayers - 1][0], HiddenLayers[NumberOfLayers - 1][1]])
        return HiddenLayers[NumberOfLayers - 1]

    def DistanceToGoal(self, X, Y):
        xDistance = abs(self.Destination[0] - X)
        yDistance = abs(self.Destination[1] - Y)
        TotalDistance = abs(xDistance - yDistance) * 10
        if (xDistance > yDistance):
            TotalDistance += (xDistance - abs(xDistance - yDistance)) * 14
        else:
            TotalDistance += (yDistance - abs(xDistance - yDistance)) * 14
        return TotalDistance

    def Run(self):
        while True:
            #self.WeightMatrix, self.Biases = RandomWeightsAndBiases(2, 1, 1, 3)
            self.Map[self.CurrentLocation[1]][self.CurrentLocation[0]][1] = 0
            NextTile = self.Sigmoid(self.Network([self.DistanceToGoal(self.CurrentLocation[0], self.CurrentLocation[1]), self.LengthOfPath])[0])

            XMove = 0
            YMove = 0

            if (NextTile < 0.125):
                XMove = 1
            elif (0.125 <= NextTile and NextTile < 0.25):
                XMove = 1
                YMove = 1
            elif (0.25 <= NextTile and NextTile < 0.375):
                YMove = 1
            elif (0.375 <= NextTile and NextTile < 0.5):
                XMove = -1
                YMove = 1
            elif (0.5 <= NextTile and NextTile < 0.625):
                XMove = -1
            elif (0.625 <= NextTile and NextTile < 0.75):
                XMove = -1
                YMove = -1
            elif (0.75 <= NextTile and NextTile < 0.875):
                YMove = -1
            else:
                XMove = 1
                YMove = -1

            if (self.Map[self.CurrentLocation[1] + YMove][self.CurrentLocation[0] + XMove][1] == 0):
                return self.ReturnFunction()

            elif (0 not in (XMove, YMove)):
                if (0 in (self.Map[self.CurrentLocation[1]][self.CurrentLocation[0] + XMove][1], self.Map[self.CurrentLocation[1] + YMove][self.CurrentLocation[0]][1])):
                    return self.ReturnFunction()

            
            self.CurrentLocation[0] += XMove
            self.CurrentLocation[1] += YMove
            if (0 in (XMove, YMove)):
                self.LengthOfPath += 10
            else:
                self.LengthOfPath += 14
            self.Path.append((self.CurrentLocation[0], self.CurrentLocation[1]))

            if (self.CurrentLocation == self.Destination):
                return self.ReturnFunction()

def RandomWeightsAndBiases(NumberOfInputs, NumberOfOutputs, NumberOfHiddenLayers, NeuronsInLayer):  #(2, 1, 1, 3)
    Weights = [[]]
    Biases = [[]]

    for i in range(NeuronsInLayer):
        Biases[0].append(np.random.randn())
        Weights[0].append([])
        for f in range(NumberOfInputs):
            Weights[0][i].append(np.random.randn())

    for i in range(1, NumberOfHiddenLayers):
        Biases.append([])
        Weights.append([])
        for Neuron in range(NeuronsInLayer):
            Biases[i].append(np.random.randn())
            Weights[i].append([])
            for f in range(NeuronsInLayer):
                Weights[i][Neuron].append(np.random.randn())

    BiasesLastIndex = len(Biases)
    WeightsLastIndex = len(Weights)
    Biases.append([])
    Weights.append([])
    for Output in range(NumberOfOutputs):
        Biases[BiasesLastIndex].append(np.random.randn())
        Weights[WeightsLastIndex].append([])
        for f in range(NeuronsInLayer):
            Weights[WeightsLastIndex][Output].append(np.random.randn())

    return Weights, Biases

def AddToNeuralNetworkList(NewNetwork, ListOfNetworks):
    Minimum = 0
    Maximum = len(ListOfNetworks) - 1
    Index = int(len(ListOfNetworks) / 2)
    if (Maximum > 0):
        while True:
            if NewNetwork.Outputs[0] < ListOfNetworks[Index].Outputs[0]:
                Maximum = Index
                Index = int(Index - (Index - Minimum) / 2)
            elif NewNetwork.Outputs[0] > ListOfNetworks[Index].Outputs[0]:
                Minimum = Index
                Index = int(Index + (Maximum - Index) / 2)
            else:
                if (NewNetwork.LengthOfPath > ListOfNetworks[Index].LengthOfPath):
                    while Index < len(ListOfNetworks) and NewNetwork.Outputs[0] == ListOfNetworks[Index].Outputs[0] and NewNetwork.LengthOfPath > ListOfNetworks[Index].LengthOfPath:
                        Index += 1
                elif (NewNetwork.LengthOfPath < ListOfNetworks[Index].LengthOfPath):
                    while Index > 0 and NewNetwork.Outputs[0] == ListOfNetworks[Index].Outputs[0] and NewNetwork.LengthOfPath < ListOfNetworks[Index].LengthOfPath:
                        Index -= 1
                break

            if (Index in (Minimum, Maximum)):
                if (NewNetwork.Outputs[0] > ListOfNetworks[Maximum].Outputs[0] or (NewNetwork.Outputs[0] == ListOfNetworks[Maximum].Outputs[0] and NewNetwork.LengthOfPath > ListOfNetworks[Maximum].LengthOfPath)):
                    Index = Maximum + 1
                elif (NewNetwork.Outputs[0] > ListOfNetworks[Minimum].Outputs[0] or (NewNetwork.Outputs[0] == ListOfNetworks[Minimum].Outputs[0] and NewNetwork.LengthOfPath > ListOfNetworks[Minimum].LengthOfPath)):
                    Index = Maximum
                else:
                    Index = Minimum               
                break

    elif (Maximum == 0):
        if (NewNetwork.Outputs[0] > ListOfNetworks[Minimum].Outputs[0] or (NewNetwork.Outputs[0] == ListOfNetworks[Minimum].Outputs[0] and NewNetwork.LengthOfPath > ListOfNetworks[Minimum].LengthOfPath)):
            Index = 1    

    ListOfNetworks.insert(Index, NewNetwork)

def BreedNetworks(NetworkList, MidpointIndex):
    Weights = []
    Biases = []
    Networks = [NetworkList[random.randrange(0, len(NetworkList), 1)], NetworkList[random.randrange(0, len(NetworkList), 1)]]
    
    for Layer in range(len(Networks[0].WeightMatrix)):
        Weights.append([])
        for Neuron in range(len(Networks[0].WeightMatrix[Layer])):
            Weights[Layer].append([])
            for Weight in range(len(Networks[0].WeightMatrix[Layer][Neuron])):
                RandomValue = random.randrange(0, MidpointIndex + 1, 1)
                if (RandomValue < MidpointIndex):
                    Weights[Layer][Neuron].append(Networks[0].WeightMatrix[Layer][Neuron][Weight])
                elif (RandomValue > MidpointIndex):
                    Weights[Layer][Neuron].append(Networks[1].WeightMatrix[Layer][Neuron][Weight])
                else:
                    Weights[Layer][Neuron].append(np.random.randn())

    for Layer in range(len(Networks[0].Biases)):
        Biases.append([])
        for Neuron in range(len(Networks[0].Biases[Layer])):
            RandomValue = random.randrange(0, 101, 1)
            if (RandomValue < 50):
                Biases[Layer].append(Networks[0].Biases[Layer][Neuron])
            elif (RandomValue > 50):
                Biases[Layer].append(Networks[1].Biases[Layer][Neuron])
            else:
                Biases[Layer].append(np.random.randn())
    
    return Weights, Biases

def SortPopulation(Population):
    TempList = []
    for NewNetwork in Population:
        Minimum = 0
        Maximum = len(TempList) - 1
        Index = int(len(TempList) / 2)
        if (Maximum > 0):
            while True:
                if NewNetwork.Outputs[0] < TempList[Index].Outputs[0]:
                    Maximum = Index
                    Index = int(Index - (Index - Minimum) / 2)
                elif NewNetwork.Outputs[0] > TempList[Index].Outputs[0]:
                    Minimum = Index
                    Index = int(Index + (Maximum - Index) / 2)
                else:
                    if (NewNetwork.LengthOfPath > TempList[Index].LengthOfPath):
                        while Index < len(TempList) and NewNetwork.Outputs[0] == TempList[Index].Outputs[0] and NewNetwork.LengthOfPath > TempList[Index].LengthOfPath:
                            Index += 1
                    elif (NewNetwork.LengthOfPath < TempList[Index].LengthOfPath):
                        while Index > 0 and NewNetwork.Outputs[0] == TempList[Index].Outputs[0] and NewNetwork.LengthOfPath < TempList[Index].LengthOfPath:
                            Index -= 1
                    break

                if (Index in (Minimum, Maximum)):
                    if (NewNetwork.Outputs[0] > TempList[Maximum].Outputs[0] or (NewNetwork.Outputs[0] == TempList[Maximum].Outputs[0] and NewNetwork.LengthOfPath > TempList[Maximum].LengthOfPath)):
                        Index = Maximum + 1
                    elif (NewNetwork.Outputs[0] > TempList[Minimum].Outputs[0] or (NewNetwork.Outputs[0] == TempList[Minimum].Outputs[0] and NewNetwork.LengthOfPath > TempList[Minimum].LengthOfPath)):
                        Index = Maximum
                    else:
                        Index = Minimum               
                    break

        elif (Maximum == 0):
            if (NewNetwork.Outputs[0] > TempList[Minimum].Outputs[0] or (NewNetwork.Outputs[0] == TempList[Minimum].Outputs[0] and NewNetwork.LengthOfPath > TempList[Minimum].LengthOfPath)):
                Index = 1    

        TempList.insert(Index, NewNetwork)

    #Population = TempList
    return TempList