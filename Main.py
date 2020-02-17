import pygame
import time
from Heuristic_Astar_3 import*

DisplayWidth = 1000
DisplayHeight = 800
Displacement = [[DisplayWidth / 4, DisplayHeight / 4],
                [DisplayWidth / 4 * 3, DisplayHeight / 4],
                [DisplayWidth / 4, DisplayHeight / 4 * 3],
                [DisplayWidth / 4 * 3, DisplayHeight / 4 * 3]]

pygame.init()
Window = pygame.display.set_mode((DisplayWidth, DisplayHeight))
pygame.display.set_caption("AI Labb 2, Map 1")

#Map = open("Map1.txt", "r")
#Map = open("Map2.txt", "r")
Map = open("Map3.txt", "r")
lines = Map.readlines()
Map.close()

Tiles = []

TileSize = 10
SmallTile = TileSize / 2

DisplacementX = (len(lines[0]) - 1) * TileSize / 2
DisplacementY = len(lines) * TileSize / 2

Window.fill((255, 255, 255))

StartCoordinates = [0, 0]
GoalCoordinates = [0, 0]

########################################################################################################################

class AStar:
    def __init__(self, StartCoordinates):
        self.OpenList = []
        self.ClosedList = []
        self.CurrentCoordinates = StartCoordinates

        TotalDistance = self.DistanceToGoal(StartCoordinates[0], StartCoordinates[1])
        Tiles[StartCoordinates[1]][StartCoordinates[0]] = ["S", 0, TotalDistance, TotalDistance, "NULL", "Unknown"]
        self.ClosedList.append([StartCoordinates[0], StartCoordinates[1]])

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

            pygame.draw.rect(Window, (200, 0, 255), ((Displacement[0][0] - DisplacementX + TileSize * self.CurrentCoordinates[0] + SmallTile / 2), (Displacement[0][1] - DisplacementY + TileSize * self.CurrentCoordinates[1] + SmallTile / 2), SmallTile, SmallTile))
            self.OpenList.remove(self.CurrentCoordinates)
            self.ClosedList.append(self.CurrentCoordinates)

            
            #pygame.draw.rect(Window, (255, 255, 255), ((Location[0] - DisplacementX + TileSize * i), (Location[1] - DisplacementY + TileSize * index), TileSize - 1, TileSize - 1))
            #DisplacementX = (len(lines[0]) - 1) * TileSize / 2
            #DisplacementY = len(lines) * TileSize / 2



        while (self.CurrentCoordinates != StartCoordinates):
            pygame.draw.rect(Window, (255, 0, 0), ((Displacement[0][0] - DisplacementX + TileSize * self.CurrentCoordinates[0] + SmallTile / 2), (Displacement[0][1] - DisplacementY + TileSize * self.CurrentCoordinates[1] + SmallTile / 2), SmallTile, SmallTile))
            self.CurrentCoordinates = Tiles[self.CurrentCoordinates[1]][self.CurrentCoordinates[0]][4]
        pygame.draw.rect(Window, (255, 0, 0), ((Displacement[0][0] - DisplacementX + TileSize * self.CurrentCoordinates[0] + SmallTile / 2), (Displacement[0][1] - DisplacementY + TileSize * self.CurrentCoordinates[1] + SmallTile / 2), SmallTile, SmallTile))


########################################################################################################################

class BreadthFirst:
    def __init__(self, StartCoordinates):
        self.CurrentList = []
        self.TempList = []
        self.CurrentCoordinates = StartCoordinates
        self.CurrentList.append(self.CurrentCoordinates)
        Tiles[self.CurrentCoordinates[1]][self.CurrentCoordinates[0]][5] = "Visited"


    def CheckTile(self, BaseX, BaseY):
        X = BaseX + self.CurrentCoordinates[0]
        Y = BaseY + self.CurrentCoordinates[1]

        if (Tiles[Y][X][0] == "G"):
            Tiles[Y][X][4] = self.CurrentCoordinates
            self.CurrentCoordinates = [X, Y]
            return True

        elif (Tiles[Y][X][0] != "X" and Tiles[Y][X][5] != "Visited"):
            
            if (BaseX != 0 and BaseY != 0):
                #return False
                if (Tiles[Y][self.CurrentCoordinates[0]][0] == "X" or Tiles[self.CurrentCoordinates[1]][X][0] == "X"):
                    return False
                self.TempList.append([X, Y])
            else:
                self.TempList.insert(0, [X, Y])

            Tiles[Y][X][4] = self.CurrentCoordinates
            Tiles[Y][X][5] = "Visited"
            pygame.draw.rect(Window, (200, 0, 255), ((Displacement[1][0] - DisplacementX + TileSize * X + SmallTile / 2), (Displacement[1][1] - DisplacementY + TileSize * Y + SmallTile / 2), SmallTile, SmallTile))


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
            pygame.draw.rect(Window, (255, 0, 0), ((Displacement[1][0] - DisplacementX + TileSize * self.CurrentCoordinates[0] + SmallTile / 2), (Displacement[1][1] - DisplacementY + TileSize * self.CurrentCoordinates[1] + SmallTile / 2), SmallTile, SmallTile))
            self.CurrentCoordinates = Tiles[self.CurrentCoordinates[1]][self.CurrentCoordinates[0]][4]
        pygame.draw.rect(Window, (255, 0, 0), ((Displacement[1][0] - DisplacementX + TileSize * self.CurrentCoordinates[0] + SmallTile / 2), (Displacement[1][1] - DisplacementY + TileSize * self.CurrentCoordinates[1] + SmallTile / 2), SmallTile, SmallTile))


########################################################################################################################

class DepthFirst:
    def __init__(self, StartCoordinates):
        self.CurrentCoordinates = StartCoordinates
        Tiles[self.CurrentCoordinates[1]][self.CurrentCoordinates[0]][5] = "Visited2"


    def CheckTile(self, BaseX, BaseY):      
        
        #X = BaseX + self.CurrentCoordinates[0]
        #Y = BaseY + self.CurrentCoordinates[1]

        if (Tiles[BaseY][BaseX][0] == "G"):
            #Tiles[BaseY][BaseX][4] = self.CurrentCoordinates
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
                    pygame.draw.rect(Window, (200, 0, 255), ((Displacement[2][0] - DisplacementX + TileSize * (BaseX + X) + SmallTile / 2), (Displacement[2][1] - DisplacementY + TileSize * (BaseY + Y) + SmallTile / 2), SmallTile, SmallTile))


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
            pygame.draw.rect(Window, (255, 0, 0), ((Displacement[2][0] - DisplacementX + TileSize * self.CurrentCoordinates[0] + SmallTile / 2), (Displacement[2][1] - DisplacementY + TileSize * self.CurrentCoordinates[1] + SmallTile / 2), SmallTile, SmallTile))
            self.CurrentCoordinates = Tiles[self.CurrentCoordinates[1]][self.CurrentCoordinates[0]][4]
        pygame.draw.rect(Window, (255, 0, 0), ((Displacement[2][0] - DisplacementX + TileSize * self.CurrentCoordinates[0] + SmallTile / 2), (Displacement[2][1] - DisplacementY + TileSize * self.CurrentCoordinates[1] + SmallTile / 2), SmallTile, SmallTile))


########################################################################################################################

class NeuralNetwork:
    def __init__(self):
        self.InputLayerSize = 2
        self.OutputLayerSize = 1
        self.HiddenLayerSize = 3







########################################################################################################################


def text_object(text, font):
    textSurface = font.render(text, True, (0, 0, 0))
    return textSurface, textSurface.get_rect()


for Location in Displacement:
    pygame.draw.rect(Window, (0, 0, 0), (Location[0] - DisplacementX, Location[1] - DisplacementY, (len(lines[0]) - 1) * TileSize, len(lines) * TileSize))

for index, line in enumerate(lines):
    Tiles.append([])

    for i, letter in enumerate(line):
        Tiles[index].append([letter, float('inf'), float('inf'), float('inf'), "NULL", "Unknown"])
        if (letter == "0"):
            for Location in Displacement:
                pygame.draw.rect(Window, (255, 255, 255), ((Location[0] - DisplacementX + TileSize * i), (Location[1] - DisplacementY + TileSize * index), TileSize - 1, TileSize - 1))
        elif (letter == "S"):
            StartCoordinates = [i, index]
            for Location in Displacement:
                pygame.draw.rect(Window, (0, 0, 255), ((Location[0] - DisplacementX + TileSize * i), (Location[1] - DisplacementY + TileSize * index), TileSize, TileSize))
        elif (letter == "G"):
            GoalCoordinates = [i, index]
            for Location in Displacement:
                pygame.draw.rect(Window, (0, 255, 0), ((Location[0] - DisplacementX + TileSize * i), (Location[1] - DisplacementY + TileSize * index), TileSize, TileSize))



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


#StartCoordinates = [22, 4]


TheAStar = AStar(StartCoordinates)
TheAStar.Run()

TheBreadthFirst = BreadthFirst(StartCoordinates)
TheBreadthFirst.Run()

TheDepthFirst = DepthFirst(StartCoordinates)
TheDepthFirst.Run()

pygame.display.update()

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


    #Window.fill((255, 255, 255))
    #pygame.draw.rect(Window, (76, 35, 123), (20, 20, 10, 10))
    #pygame.display.update()