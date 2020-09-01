import pygame
import time
from Neural_Network import*

MapNumber = input("Select which Map to run (1, 2 or 3): ")

while MapNumber not in ("1", "2", "3"):
    MapNumber = input("Not a valid input. Select which Map to run (1, 2 or 3): ")

DisplayWidth = 1000
DisplayHeight = 800
Displacement = [[int(DisplayWidth / 4), int(DisplayHeight / 4)],
                [int(DisplayWidth / 4 * 3), int(DisplayHeight / 4)],
                [int(DisplayWidth / 4), int(DisplayHeight / 4 * 3)],
                [int(DisplayWidth / 4 * 3), int(DisplayHeight / 4 * 3)]]

pygame.init()
Window = pygame.display.set_mode((DisplayWidth, DisplayHeight))
pygame.display.set_caption("AI Labb 2, Map 1")

#MapNumber = str(2)
Map = open("Map" + MapNumber + ".txt", "r")
lines = Map.readlines()
Map.close()

Tiles = []

StartCoordinates = [0, 0]
GoalCoordinates = [0, 0]

TileSize = 10
SmallTile = int(TileSize / 2)

DisplacementX = int((len(lines[0]) - 1) * TileSize / 2)
DisplacementY = int(len(lines) * TileSize / 2)

Window.fill((255, 255, 255))

NeuralNetworkGeneration = 1

########################################################################################################################

class AStar:
    def __init__(self, StartCoordinates):
        self.OpenList = []
        self.ClosedList = []
        self.CurrentCoordinates = StartCoordinates

        TotalDistance = self.DistanceToGoal(StartCoordinates[0], StartCoordinates[1])
        Tiles[StartCoordinates[1]][StartCoordinates[0]] = ["S", 0, TotalDistance, TotalDistance, "NULL", "Unknown"]
        self.ClosedList.append([StartCoordinates[0], StartCoordinates[1]])
        self.PathLength = 0

    def DistanceToGoal(self, X, Y):
        xDistance = abs(GoalCoordinates[0] - X)
        yDistance = abs(GoalCoordinates[1] - Y)
        TotalDistance = abs(xDistance - yDistance) * 10
        if (xDistance > yDistance):
            TotalDistance += (xDistance - abs(xDistance - yDistance)) * 14
        else:
            TotalDistance += (yDistance - abs(xDistance - yDistance)) * 14
        return TotalDistance


    def CheckTile(self, BaseX, BaseY):
        X = BaseX + self.CurrentCoordinates[0]
        Y = BaseY + self.CurrentCoordinates[1]

        for Closed in self.ClosedList:
            if (Closed == [X, Y]):
                return False

        if (Tiles[Y][X][0] != "X"):
            DistanceToStart = Tiles[self.CurrentCoordinates[1]][self.CurrentCoordinates[0]][1] + 10

            if (BaseX != 0 and BaseY != 0):
                if (Tiles[Y][self.CurrentCoordinates[0]][0] != "X" and Tiles[self.CurrentCoordinates[1]][X][0] != "X"):
                    DistanceToStart += 4
                else:
                    return False

            for Open in self.OpenList:
                if (Open == [X, Y]):
                    if (DistanceToStart < Tiles[Y][X][1]):
                        Tiles[Y][X][1] = DistanceToStart
                        Tiles[Y][X][3] = Tiles[Y][X][1] + Tiles[Y][X][2]
                        Tiles[Y][X][4] = self.CurrentCoordinates
                    return True

            Tiles[Y][X][1] = DistanceToStart
            Tiles[Y][X][2] = self.DistanceToGoal(X, Y)
            Tiles[Y][X][3] = Tiles[Y][X][1] + Tiles[Y][X][2]
            Tiles[Y][X][4] = self.CurrentCoordinates
            self.OpenList.append([X, Y])
            return True

        return False

    def Run(self):
        Path = []
        while (Tiles[self.CurrentCoordinates[1]][self.CurrentCoordinates[0]][0] != "G"):
            X = -1
            Y = -1

            while X < 2:
                while Y < 2:         
                    self.CheckTile(X, Y)
                    Y += 1
                X += 1
                Y = -1

            ShortestTotal = float('inf')
            for Open in self.OpenList:
                if (Tiles[Open[1]][Open[0]][3] < ShortestTotal or (Tiles[Open[1]][Open[0]][3] == ShortestTotal and Tiles[Open[1]][Open[0]][2] < Tiles[self.CurrentCoordinates[1]][self.CurrentCoordinates[0]][2])):
                    self.CurrentCoordinates = Open
                    ShortestTotal = Tiles[Open[1]][Open[0]][3]

            pygame.draw.rect(Window, (200, 0, 255), (int(Displacement[0][0] - DisplacementX + TileSize * self.CurrentCoordinates[0] + SmallTile / 2), int(Displacement[0][1] - DisplacementY + TileSize * self.CurrentCoordinates[1] + SmallTile / 2), SmallTile, SmallTile))
            self.OpenList.remove(self.CurrentCoordinates)
            self.ClosedList.append(self.CurrentCoordinates)


        while (self.CurrentCoordinates != StartCoordinates):
            Path.insert(0, self.CurrentCoordinates)
            pygame.draw.rect(Window, (255, 0, 0), (int(Displacement[0][0] - DisplacementX + TileSize * self.CurrentCoordinates[0] + SmallTile / 2), int(Displacement[0][1] - DisplacementY + TileSize * self.CurrentCoordinates[1] + SmallTile / 2), SmallTile, SmallTile))           
            if (self.CurrentCoordinates[0] != Tiles[self.CurrentCoordinates[1]][self.CurrentCoordinates[0]][4][0] and self.CurrentCoordinates[1] != Tiles[self.CurrentCoordinates[1]][self.CurrentCoordinates[0]][4][1]):
                self.PathLength += 14
            else:
                self.PathLength += 10 
            self.CurrentCoordinates = Tiles[self.CurrentCoordinates[1]][self.CurrentCoordinates[0]][4]
        
        Path.insert(0, self.CurrentCoordinates)
        pygame.draw.rect(Window, (255, 0, 0), (int(Displacement[0][0] - DisplacementX + TileSize * self.CurrentCoordinates[0] + SmallTile / 2), int(Displacement[0][1] - DisplacementY + TileSize * self.CurrentCoordinates[1] + SmallTile / 2), SmallTile, SmallTile))
        
        return Path


########################################################################################################################

class BreadthFirst:
    def __init__(self, StartCoordinates):
        self.CurrentList = []
        self.TempList = []
        self.CurrentCoordinates = StartCoordinates
        self.CurrentList.append(self.CurrentCoordinates)
        Tiles[self.CurrentCoordinates[1]][self.CurrentCoordinates[0]][5] = "Visited"
        self.PathLength = 0


    def CheckTile(self, BaseX, BaseY):
        X = BaseX + self.CurrentCoordinates[0]
        Y = BaseY + self.CurrentCoordinates[1]

        if (Tiles[Y][X][0] == "G"):
            Tiles[Y][X][4] = self.CurrentCoordinates
            self.CurrentCoordinates = [X, Y]
            return True

        elif (Tiles[Y][X][0] != "X" and Tiles[Y][X][5] != "Visited"):
            
            if (BaseX != 0 and BaseY != 0):
                if (Tiles[Y][self.CurrentCoordinates[0]][0] == "X" or Tiles[self.CurrentCoordinates[1]][X][0] == "X"):
                    return False
                self.TempList.append([X, Y])
            else:
                self.TempList.insert(0, [X, Y])

            Tiles[Y][X][4] = self.CurrentCoordinates
            Tiles[Y][X][5] = "Visited"
            pygame.draw.rect(Window, (200, 0, 255), (int(Displacement[1][0] - DisplacementX + TileSize * X + SmallTile / 2), int(Displacement[1][1] - DisplacementY + TileSize * Y + SmallTile / 2), SmallTile, SmallTile))


            return False

        return False


    def Run(self):
        while (Tiles[self.CurrentCoordinates[1]][self.CurrentCoordinates[0]][0] != "G"):
            Breaker = False

            for Tile in self.CurrentList:
                self.CurrentCoordinates = Tile
                X = -1
                Y = -1

                while X < 2:
                    while Y < 2:         
                        if (self.CheckTile(X, Y)):
                            Y = 1
                            X = 1
                            Breaker = True
                        Y += 1
                    X += 1
                    Y = -1

                if (Breaker):
                    break

            self.CurrentList = self.TempList
            self.TempList = []


        while (self.CurrentCoordinates != StartCoordinates):
            pygame.draw.rect(Window, (255, 0, 0), (int(Displacement[1][0] - DisplacementX + TileSize * self.CurrentCoordinates[0] + SmallTile / 2), int(Displacement[1][1] - DisplacementY + TileSize * self.CurrentCoordinates[1] + SmallTile / 2), SmallTile, SmallTile))
            if (self.CurrentCoordinates[0] != Tiles[self.CurrentCoordinates[1]][self.CurrentCoordinates[0]][4][0] and self.CurrentCoordinates[1] != Tiles[self.CurrentCoordinates[1]][self.CurrentCoordinates[0]][4][1]):
                self.PathLength += 14
            else:
                self.PathLength += 10 
            self.CurrentCoordinates = Tiles[self.CurrentCoordinates[1]][self.CurrentCoordinates[0]][4]

        pygame.draw.rect(Window, (255, 0, 0), (int(Displacement[1][0] - DisplacementX + TileSize * self.CurrentCoordinates[0] + SmallTile / 2), int(Displacement[1][1] - DisplacementY + TileSize * self.CurrentCoordinates[1] + SmallTile / 2), SmallTile, SmallTile))


########################################################################################################################

class DepthFirst:
    def __init__(self, StartCoordinates):
        self.CurrentCoordinates = StartCoordinates
        Tiles[self.CurrentCoordinates[1]][self.CurrentCoordinates[0]][5] = "Visited2"
        self.PathLength = 0


    def CheckTile(self, BaseX, BaseY):      

        if (Tiles[BaseY][BaseX][0] == "G"):
            self.CurrentCoordinates = [BaseX, BaseY]
            return True

        X = -1
        Y = -1

        while X < 2:
            while Y < 2:                      
                if (Tiles[BaseY + Y][BaseX + X][0] != "X" and Tiles[BaseY + Y][BaseX + X][5] != "Visited2"):
                    if (X != 0 and Y != 0):
                        if (Tiles[BaseY + Y][BaseX][0] == "X" or Tiles[BaseY][BaseX + X][0] == "X"):
                            Y += 1
                            continue

                    Tiles[BaseY + Y][BaseX + X][5] = "Visited2"
                    pygame.draw.rect(Window, (200, 0, 255), (int(Displacement[2][0] - DisplacementX + TileSize * (BaseX + X) + SmallTile / 2), int(Displacement[2][1] - DisplacementY + TileSize * (BaseY + Y) + SmallTile / 2), SmallTile, SmallTile))


                    if (self.CheckTile(BaseX + X, BaseY + Y)):
                        Tiles[BaseY + Y][BaseX + X][4] = [BaseX, BaseY]
                        return True
                Y += 1
            X += 1
            Y = -1

        return False



    def Run(self):
        
        if (self.CheckTile(self.CurrentCoordinates[0], self.CurrentCoordinates[1]) == False):
            return False

        while (self.CurrentCoordinates != StartCoordinates):
            pygame.draw.rect(Window, (255, 0, 0), (int(Displacement[2][0] - DisplacementX + TileSize * self.CurrentCoordinates[0] + SmallTile / 2), int(Displacement[2][1] - DisplacementY + TileSize * self.CurrentCoordinates[1] + SmallTile / 2), SmallTile, SmallTile))
            if (self.CurrentCoordinates[0] != Tiles[self.CurrentCoordinates[1]][self.CurrentCoordinates[0]][4][0] and self.CurrentCoordinates[1] != Tiles[self.CurrentCoordinates[1]][self.CurrentCoordinates[0]][4][1]):
                self.PathLength += 14
            else:
                self.PathLength += 10          
            self.CurrentCoordinates = Tiles[self.CurrentCoordinates[1]][self.CurrentCoordinates[0]][4]

        pygame.draw.rect(Window, (255, 0, 0), (int(Displacement[2][0] - DisplacementX + TileSize * self.CurrentCoordinates[0] + SmallTile / 2), int(Displacement[2][1] - DisplacementY + TileSize * self.CurrentCoordinates[1] + SmallTile / 2), SmallTile, SmallTile))


########################################################################################################################


def text_object(text, font):
    textSurface = font.render(text, True, (0, 0, 0))
    return textSurface, textSurface.get_rect()

Labyrinth = pygame.Surface(((len(lines[0]) - 1) * TileSize, len(lines) * TileSize))

for index, line in enumerate(lines):
    Tiles.append([])
    NeuralNetworkTiles.append([])

    for i, letter in enumerate(line):
        Tiles[index].append([letter, float('inf'), float('inf'), float('inf'), "NULL", "Unknown"])
        
        if (letter == "X"):
            NeuralNetworkTiles[index].append([letter, 0])
        else:
            NeuralNetworkTiles[index].append([letter, 1])

        if (letter == "0"):
            pygame.draw.rect(Labyrinth, (255, 255, 255), ((TileSize * i), (TileSize * index), TileSize - 1, TileSize - 1))
        elif (letter == "S"):
            pygame.draw.rect(Labyrinth, (0, 0, 255), ((TileSize * i), (TileSize * index), TileSize, TileSize))
            StartCoordinates = [i, index]
        elif (letter == "G"):
            pygame.draw.rect(Labyrinth, (0, 255, 0), ((TileSize * i), (TileSize * index), TileSize, TileSize))
            GoalCoordinates = [i, index]

for Location in Displacement:
    DrawArea = Labyrinth.get_rect().move(Location[0] - DisplacementX, Location[1] - DisplacementY)
    Window.blit(Labyrinth, DrawArea)

largeText = pygame.font.Font('freesansbold.ttf', 20)
TextSurf, TextRect = text_object("A*", largeText)
TextRect.midtop = (Displacement[0][0], Displacement[0][1] + DisplacementY + 4)
Window.blit(TextSurf, TextRect)

TextSurf, TextRect = text_object("Breadth First", largeText)
TextRect.midtop = (Displacement[1][0], Displacement[1][1] + DisplacementY + 4)
Window.blit(TextSurf, TextRect)

TextSurf, TextRect = text_object("Depth First", largeText)
TextRect.midtop = (Displacement[2][0], Displacement[2][1] + DisplacementY + 4)
Window.blit(TextSurf, TextRect)

TextSurf, TextRect = text_object("Neural Network", largeText)
TextRect.midtop = (Displacement[3][0], Displacement[3][1] + DisplacementY + 4)
Window.blit(TextSurf, TextRect)

TextSurf, TextRect = text_object("Generation: " + str(NeuralNetworkGeneration), largeText)
TextRect.midtop = (Displacement[3][0], Displacement[3][1] + DisplacementY + 20)
Window.blit(TextSurf, TextRect)


TheAStar = AStar(StartCoordinates)
LastTime = time.process_time()
ThePath = TheAStar.Run()
CurrentTime = time.process_time()
print("A* Runtime: ", CurrentTime - LastTime)

TheBreadthFirst = BreadthFirst(StartCoordinates)
LastTime = time.process_time()
TheBreadthFirst.Run()
CurrentTime = time.process_time()
print("Breadth First Runtime: ", CurrentTime - LastTime)

TheDepthFirst = DepthFirst(StartCoordinates)
LastTime = time.process_time()
TheDepthFirst.Run()
CurrentTime = time.process_time()
print("Depth First Runtime: ", CurrentTime - LastTime)


Population = InitialPopulation(SizeOfPopulation, MapNumber, StartCoordinates, GoalCoordinates)
LastPopulation = Population
GenePool = []

LastTime = time.process_time()
Population[0].Run()
CurrentTime = time.process_time()
print("Neural Network Runtime: ", CurrentTime - LastTime)


DrawArea = Labyrinth.get_rect().move(Displacement[3][0] - DisplacementX, Displacement[3][1] - DisplacementY)
Window.blit(Labyrinth, DrawArea)
for Tile in Population[0].Path:
    pygame.draw.rect(Window, (255, 0, 0), (int(Displacement[3][0] - DisplacementX + TileSize * Tile[0] + SmallTile / 2), int(Displacement[3][1] - DisplacementY + TileSize * Tile[1] + SmallTile / 2), SmallTile, SmallTile))
pygame.display.update()

CurrentIndex = 0
Paused = False
Training = False
Running = True
while Running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            Running = False
            break
        
        elif event.type == pygame.KEYDOWN:
            if (event.key == pygame.K_ESCAPE):
                Running = False
                break
            elif (event.key == pygame.K_SPACE):
                Paused = not Paused
            elif (event.key == pygame.K_t):
                Training = not Training
                if(Training == False):
                    CurrentIndex = 0
                    DrawArea = Labyrinth.get_rect().move(Displacement[3][0] - DisplacementX, Displacement[3][1] - DisplacementY)
                    Window.blit(Labyrinth, DrawArea)
                    for Tile in LastPopulation[0].Path:
                        pygame.draw.rect(Window, (255, 0, 0), (int(Displacement[3][0] - DisplacementX + TileSize * Tile[0] + SmallTile / 2), int(Displacement[3][1] - DisplacementY + TileSize * Tile[1] + SmallTile / 2), SmallTile, SmallTile))
                    pygame.display.update()

            elif (event.key == pygame.K_s):
                with open("Population" + MapNumber + ".txt", "wb") as PopulationFile:
                    pickle.dump(LastPopulation, PopulationFile)



    if Paused == False and Training == True:
        if (CurrentIndex < len(Population)):
            Population[CurrentIndex].Run()
            
            DrawArea = Labyrinth.get_rect().move(Displacement[3][0] - DisplacementX, Displacement[3][1] - DisplacementY)
            Window.blit(Labyrinth, DrawArea)
            for Tile in Population[CurrentIndex].Path:
                pygame.draw.rect(Window, (255, 0, 0), (int(Displacement[3][0] - DisplacementX + TileSize * Tile[0] + SmallTile / 2), int(Displacement[3][1] - DisplacementY + TileSize * Tile[1] + SmallTile / 2), SmallTile, SmallTile))
            pygame.display.update()

            CurrentIndex += 1
        
        else:
            Population = SortPopulation(Population)
            LastPopulation = Population
            GenePool.clear()
            BestPopulation = PickBestPopulation(Population, GenePool)
            NewPopulation = []
            Crossover(NewPopulation, GenePool, StartCoordinates, GoalCoordinates)
            Mutate(NewPopulation)
            for Network in BestPopulation:
                NewPopulation.append(Network)
            Population = NewPopulation
            CurrentIndex = 0

            NeuralNetworkGeneration += 1
            pygame.draw.rect(Window, (255, 255, 255), (Displacement[3][0] - DisplacementX - 20, Displacement[3][1] + DisplacementY + 20, 500, 20))
            TextSurf, TextRect = text_object("Generation: " + str(NeuralNetworkGeneration), largeText)
            TextRect.midtop = (Displacement[3][0], Displacement[3][1] + DisplacementY + 20)
            Window.blit(TextSurf, TextRect)