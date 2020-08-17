import numpy as np
import copy
import random
import os.path
import pickle

NeuralNetworkTiles = []

SizeOfPopulation = 200
BestAgentSelection = 12
WorstAgentSelection = 3
MutationRate = 0.01
MutationRange = 10.0

class NeuralNetwork():
    def __init__(self, StartingLocation, Destination, Map, WeightMatrix, Biases):
        self.StartingLocation = StartingLocation
        self.CurrentLocation = [StartingLocation[0], StartingLocation[1]]
        self.Destination = Destination
        self.Path = [(StartingLocation[0], StartingLocation[1])]
        self.Map = copy.deepcopy(Map)
        self.WeightMatrix = copy.deepcopy(WeightMatrix)
        self.Biases = copy.deepcopy(Biases)
        #self.MaxPathLength = self.DistanceToGoal(self.StartingLocation[0], self.StartingLocation[1]) * 10                  #Enable if Backtracking is Impossible

    LengthOfPath = 0
    Outputs = []
    Fitness = 0

    def ReturnFunction(self):
        ReturnValue = [self.DistanceToGoal(self.CurrentLocation[0], self.CurrentLocation[1]), self.LengthOfPath]
        self.Fitness = (1 - self.Sigmoid(ReturnValue[0] * 0.01)) * 50 - self.Sigmoid(ReturnValue[1] * 0.01)
        #print(ReturnValue, "Fitness:", self.Fitness)
        self.Outputs = ReturnValue
        return ReturnValue

    def Sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def Network(self, inputs):
        HiddenLayers = copy.deepcopy(self.Biases)


        for Neuron in range(len(self.Biases[0])):
            for Input in range(len(inputs)):
                HiddenLayers[0][Neuron] += inputs[Input] * self.WeightMatrix[0][Neuron][Input] #+ self.Biases[0][Neuron]
            HiddenLayers[0][Neuron] = self.Sigmoid(HiddenLayers[0][Neuron])


        NumberOfLayers = len(self.Biases)
        for Layer in range(1, NumberOfLayers):
            for Neuron in range(len(self.Biases[Layer])):
                for Weight in range(len(self.Biases[Layer - 1])):
                    HiddenLayers[Layer][Neuron] += HiddenLayers[Layer - 1][Weight] * self.WeightMatrix[Layer][Neuron][Weight]
                    #HiddenLayers[Layer][Neuron] += HiddenLayers[Layer - 1][Neuron] * self.WeightMatrix[Layer][Neuron][Weight] #+ self.Biases[Layer][Neuron]
                HiddenLayers[Layer][Neuron] = self.Sigmoid(HiddenLayers[Layer][Neuron])

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
            self.Map[self.CurrentLocation[1]][self.CurrentLocation[0]][1] = 0               #Make Backtracking Impossible
            TheDistance = self.DistanceToGoal(self.CurrentLocation[0], self.CurrentLocation[1])
            #Fisk = self.Sigmoid(TheDistance)
            #Inputs = [self.Sigmoid(self.CurrentLocation[0]),
                      #self.Sigmoid(self.CurrentLocation[1]),
            Inputs = [self.Sigmoid(TheDistance),
                      self.Map[self.CurrentLocation[1] - 1][self.CurrentLocation[0] - 1][1],
                      self.Map[self.CurrentLocation[1]][self.CurrentLocation[0] - 1][1],
                      self.Map[self.CurrentLocation[1] + 1][self.CurrentLocation[0] - 1][1],
                      self.Map[self.CurrentLocation[1] - 1][self.CurrentLocation[0]][1],
                      self.Map[self.CurrentLocation[1] + 1][self.CurrentLocation[0]][1],
                      self.Map[self.CurrentLocation[1] - 1][self.CurrentLocation[0] + 1][1],
                      self.Map[self.CurrentLocation[1]][self.CurrentLocation[0] + 1][1],
                      self.Map[self.CurrentLocation[1] + 1][self.CurrentLocation[0] + 1][1]]
            #NextTile = self.Sigmoid(self.Network(Inputs)[0])
            #NextTile = self.Network(Inputs)[0]

            NetworkOutput = self.Network(Inputs)
            NextTile = 0
            for i in range(len(NetworkOutput)):
                if NetworkOutput[i] > NetworkOutput[NextTile]:
                    NextTile = i

            XMove = 0
            YMove = 0

            if (NextTile == 0):
                XMove = 1
            elif (NextTile == 1):
                XMove = 1
                YMove = 1
            elif (NextTile == 2):
                YMove = 1
            elif (NextTile == 3):
                XMove = -1
                YMove = 1
            elif (NextTile == 4):
                XMove = -1
            elif (NextTile == 5):
                XMove = -1
                YMove = -1
            elif (NextTile == 6):
                YMove = -1
            else:
                XMove = 1
                YMove = -1

            """
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
            """

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

            #if (self.CurrentLocation == self.Destination or self.LengthOfPath > self.MaxPathLength):           #Enable if Backtracking is Impossible
            if (self.CurrentLocation == self.Destination):
                return self.ReturnFunction()

################################################################################################################################################################################################################################################

def RandomWeightsAndBiases(NumberOfInputs, NumberOfOutputs, NumberOfHiddenLayers, NeuronsInLayer):  #(9, 8, 2, 4)
    Weights = [[]]
    Biases = [[]]

#random.uniform(-1.0, 1.0)

    for i in range(NeuronsInLayer):
        #Biases[0].append(np.random.randn())
        Biases[0].append(random.uniform(-1.0, 1.0))
        Weights[0].append([])
        for f in range(NumberOfInputs):
            #Weights[0][i].append(np.random.randn())
            Weights[0][i].append(random.uniform(-1.0, 1.0))

    for i in range(1, NumberOfHiddenLayers):
        Biases.append([])
        Weights.append([])
        for Neuron in range(NeuronsInLayer):
            #Biases[i].append(np.random.randn())
            Biases[i].append(random.uniform(-1.0, 1.0))
            Weights[i].append([])
            for f in range(NeuronsInLayer):
                #Weights[i][Neuron].append(np.random.randn())
                Weights[i][Neuron].append(random.uniform(-1.0, 1.0))

    BiasesLastIndex = len(Biases)
    WeightsLastIndex = len(Weights)
    Biases.append([])
    Weights.append([])
    for Output in range(NumberOfOutputs):
        #Biases[BiasesLastIndex].append(np.random.randn())
        Biases[BiasesLastIndex].append(random.uniform(-1.0, 1.0))
        Weights[WeightsLastIndex].append([])
        for f in range(NeuronsInLayer):
            #Weights[WeightsLastIndex][Output].append(np.random.randn())
            Weights[WeightsLastIndex][Output].append(random.uniform(-1.0, 1.0))

    return Weights, Biases

################################################################################################################################################################################################################################################

def InitialPopulation(SizeOfPopulation, MapNumber, StartCoordinates, GoalCoordinates):
    TempList = []

    if os.path.exists("Population" + MapNumber + ".txt"):
        with open("Population" + MapNumber + ".txt", "rb") as PopulationFile:
            TheFirstPopulation = pickle.load(PopulationFile)
        for IndividualNetwork in TheFirstPopulation:
            NN = NeuralNetwork(StartCoordinates, GoalCoordinates, NeuralNetworkTiles, IndividualNetwork.WeightMatrix, IndividualNetwork.Biases)
            TempList.append(NN)

    else:
        for i in range(SizeOfPopulation):
            WeightMatrix, Biases = RandomWeightsAndBiases(9, 8, 2, 8)       #(NumberOfInputs, NumberOfOutputs, NumberOfHiddenLayers, NeuronsInLayer)
            NN = NeuralNetwork(StartCoordinates, GoalCoordinates, NeuralNetworkTiles, WeightMatrix, Biases)
            TempList.append(NN)

    return TempList

################################################################################################################################################################################################################################################


def SortPopulation(Population):
    TempList = []
    for NewNetwork in Population:
        Minimum = 0
        Maximum = len(TempList) - 1
        Index = int(len(TempList) / 2)
        if (Maximum > 0):
            while True:
                if NewNetwork.Fitness > TempList[Index].Fitness:
                    Maximum = Index
                    Index = int(Index - (Index - Minimum) / 2)
                elif NewNetwork.Fitness < TempList[Index].Fitness:
                    Minimum = Index
                    Index = int(Index + (Maximum - Index) / 2)
                else:
                    break


                if (Index in (Minimum, Maximum)):
                    if (NewNetwork.Fitness < TempList[Maximum].Fitness):
                        Index = Maximum + 1
                    elif (NewNetwork.Fitness < TempList[Minimum].Fitness):
                        Index = Maximum
                    else:
                        Index = Minimum               
                    break

        elif (Maximum == 0):
            if (NewNetwork.Fitness < TempList[Minimum].Fitness):
                Index = 1    

        TempList.insert(Index, NewNetwork)

    #Population = TempList
    return TempList


################################################################################################################################################################################################################################################


def PickBestPopulation(Population, GenePool):
    TempList = []
    Chosen = 0

    for i in range(len(Population)):
        if (i == 0 or Population[i].Fitness != TempList[Chosen - 1].Fitness):
            TempList.append(Population[i])
            Chosen += 1
            for n in range(int(Population[i].Fitness)):
                GenePool.append(Population[i])
            if (Chosen >= BestAgentSelection):
                break

    for i in range(WorstAgentSelection):
        for n in range(int(Population[i].Fitness)):
            GenePool.append(Population[len(Population) - 1 - i])

    return TempList


################################################################################################################################################################################################################################################

def Crossover(Population, GenePool, StartCoordinates, GoalCoordinates):

    for i in range(int((SizeOfPopulation - BestAgentSelection) / 2)):
        Parent1 = GenePool[random.randrange(0, len(GenePool), 1)]
        Parent2 = GenePool[random.randrange(0, len(GenePool), 1)]
        while (Parent1 == Parent2):
            Parent2 = GenePool[random.randrange(0, len(GenePool), 1)]

        Weights1 = []
        Biases1 = []
        Weights2 = []
        Biases2 = []

        for Layer in range(len(Parent1.WeightMatrix)):
            Weights1.append([])
            Weights2.append([])

            for Neuron in range(len(Parent1.WeightMatrix[Layer])):
                Weights1[Layer].append([])
                Weights2[Layer].append([])

                for WeightValue in range(len(Parent1.WeightMatrix[Layer][Neuron])):
                    if (random.randrange(0, 2, 1) == 0):
                        Weights1[Layer][Neuron].append(Parent1.WeightMatrix[Layer][Neuron][WeightValue])
                        Weights2[Layer][Neuron].append(Parent2.WeightMatrix[Layer][Neuron][WeightValue])
                    else:
                        Weights1[Layer][Neuron].append(Parent2.WeightMatrix[Layer][Neuron][WeightValue])
                        Weights2[Layer][Neuron].append(Parent1.WeightMatrix[Layer][Neuron][WeightValue])

        for Layer in range(len(Parent1.Biases)):
            Biases1.append([])
            Biases2.append([])

            for Neuron in range(len(Parent1.Biases[Layer])):
                if (random.randrange(0, 2, 1) == 0):
                    Biases1[Layer].append(Parent1.Biases[Layer][Neuron])
                    Biases2[Layer].append(Parent2.Biases[Layer][Neuron])
                else:
                    Biases1[Layer].append(Parent2.Biases[Layer][Neuron])
                    Biases2[Layer].append(Parent1.Biases[Layer][Neuron])
    
        NN = NeuralNetwork(StartCoordinates, GoalCoordinates, NeuralNetworkTiles, Weights1, Biases1)
        Population.append(NN)
        NN = NeuralNetwork(StartCoordinates, GoalCoordinates, NeuralNetworkTiles, Weights2, Biases2)
        Population.append(NN)


################################################################################################################################################################################################################################################

def Mutate(Population):
    for Child in Population:
        for Layer in range(len(Child.WeightMatrix)):
            for Neuron in range(len(Child.WeightMatrix[Layer])):
                for WeightValue in range(len(Child.WeightMatrix[Layer][Neuron])):
                    if (random.uniform(0.0, 1.0) < MutationRate):
                        Child.WeightMatrix[Layer][Neuron][WeightValue] += random.uniform(-MutationRange, MutationRange)

        for Layer in range(len(Child.Biases)):
            for Neuron in range(len(Child.Biases[Layer])):
                if (random.uniform(0.0, 1.0) < MutationRate):
                    Child.Biases[Layer][Neuron] += random.uniform(-MutationRange, MutationRange)



"""
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

"""








"""

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

"""