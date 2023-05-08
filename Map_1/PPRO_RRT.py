#Brendan Neal and Adam Lobo
#ENPM661 Project 5 -- Implementation of PPRO RRT Algorithm in 2D

##======================Importing Libraries=============================##
import cv2 as cv
import numpy as np
import timeit
import math
from matplotlib import pyplot as plt
import random
##=====================================================Map Setup=============================================================##
#Sets up arena with obstacles
def setup(s, r):

    global arena
    
    #Colors
    white = (255, 255, 255)
    gray = (177, 177, 177)
    darkGray = (104, 104, 104)
    
    #Draw Radial Clearance
    for x in range(0, 500):

        for y in range(0, 400):
        
            if checkClearance(x, y, s, r):
                arena[399 - y, x] = darkGray
    
    #Draw Obstacle Borders
    for x in range(0, 500):

        for y in range(0, 400):
        
            if checkBorder(x, y, s):
                arena[399 - y, x] = gray
    
    #Draw Obstacles
    for x in range(0, 500):

        for y in range(0, 400):
        
            if checkObstacle(x, y):
                arena[399 - y, x] = white
                
#Checks to see if a point is within an obstacle
def checkObstacle(x, y):
    
    #Left Squares
    if x >= 50 and x < 100:
        
        if (y >= 50 and y < 100) or (y >= 300 and y < 350):
            return True
    
    #Right Squares
    if x >= 400 and x < 450:
        
        if (y >= 50 and y < 100) or (y >= 300 and y < 350):
            return True
        
    #Left M Rectangle
    if x >= 175 and x < 185:
        
        if y >= 125 and y < 275:
            return True
    
    #Right M Rectangle
    if x >= 315 and x < 325:
        
        if y >= 125 and y < 275:
            return True
        
    #Left M Parallelogram
    if x >= 185 and x < 250:
        
        if (y >= (27/13) * x - (3370/13)) and (y < (27/13) * x - (3175/13)):
            return True
        
    #Right M Parallelogram
    if x >= 250 and x < 315:
        
        if (y >= (-27/13) * x + (10130/13)) and (y < (-27/13) * x + (10325/13)):
            return True
    
    #Circle
    #if (x - 400) * (x - 400) + (y - 110) * (y - 110) <= 50*50:
        #return True
        
    return False
  
#Checks to see if a point is within the border of an obstacle
def checkBorder(x, y, s):
    
    slopeHeight = int(round(s/math.cos(math.radians(64.29))))
    
    #Left Squares
    if x >= 50 - s and x < 100 + s:
        
        if (y >= 50 - s and y < 100 + s) or (y >= 300 - s and y < 350 + s):
            return True
    
    #Right Squares
    if x >= 400 - s and x < 450 + s:
        
        if (y >= 50 - s and y < 100 + s) or (y >= 300 - s and y < 350 + s):
            return True
        
    #Left M Rectangle
    if x >= 175 - s and x < 185 + s:
        
        if y >= 125 - s and y < 275 + s:
            return True
    
    #Right M Rectangle
    if x >= 315 - s and x < 325 + s:
        
        if y >= 125 - s and y < 275 + s:
            return True
        
    #Left M Parallelogram
    if x >= 185 and x < 250:
        
        if (y >= (27/13) * x - (3370/13) - slopeHeight) and (y < (27/13) * x - (3175/13) + slopeHeight) and (y >= 125 - s) and (y <= 275 + s):
            return True
        
    #Right M Parallelogram
    if x >= 250 and x < 315:
        
        if (y >= (-27/13) * x + (10130/13) - slopeHeight) and (y < (-27/13) * x + (10325/13) + slopeHeight) and (y >= 125 - s) and (y <= 275 + s):
            return True

#Checks to see if a point is within radial clearance of a border
def checkClearance(x, y, s, r):
    
    rr = r - 1
    
    if rr == 0:
        return False
    
    slopeHeight = int(round((s + rr)/math.cos(math.radians(64.29))))
    
    #Left Squares
    if x >= 50 - s - rr and x < 100 + s + rr:
        
        if (y >= 50 - s - rr and y < 100 + s + rr) or (y >= 300 - s - rr and y < 350 + s + rr):
            return True
    
    #Right Squares
    if x >= 400 - s - rr and x < 450 + s + rr:
        
        if (y >= 50 - s - rr and y < 100 + s + rr) or (y >= 300 - s - rr and y < 350 + s + rr):
            return True
        
    #Left M Rectangle
    if x >= 175 - s - rr and x < 185 + s + rr:
        
        if y >= 125 - s - rr and y < 275 + s + rr:
            return True
    
    #Right M Rectangle
    if x >= 315 - s - rr and x < 325 + s + rr:
        
        if y >= 125 - s - rr and y < 275 + s + rr:
            return True
        
    #Left M Parallelogram
    if x >= 185 and x < 250:
        
        if (y >= (27/13) * x - (3370/13) - slopeHeight) and (y < (27/13) * x - (3175/13) + slopeHeight) and (y >= 125 - s - rr) and (y <= 275 + s + rr):
            return True
        
    #Right M Parallelogram
    if x >= 250 and x < 315:
        
        if (y >= (-27/13) * x + (10130/13) - slopeHeight) and (y < (-27/13) * x + (10325/13) + slopeHeight) and (y >= 125 - s - rr) and (y <= 275 + s + rr):
            return True

#Checks to see if a point is valid (by checking obstacle, border, and clearance, as well as making sure the point is within arena bounds)
def checkValid(x, y, s, r):
    
    if checkObstacle(x, 399 - y):
        return False
    
    if checkBorder(x, 399 - y, s):
        return False
    
    if checkClearance(x, 399 - y, s, r):
        return False
    
    if (x < 0 or x >= 500 or y < 0 or y >= 400):
        return False
    
    return True


##=====================================Function and Class Definitions====================================================##
class Node:
    def __init__(self, state, parent):
        self.state = state
        self.parent = parent

    def ReturnState(self):
        return self.state
    
    def ReturnParent(self):
        return self.parent
    
    def ReturnParentState(self): #Returns the Parent Node's State
        if self.ReturnParent() is None:
            return None
        return self.ReturnParent().ReturnState()


def GenerateRandomPoint():
    rand_x = random.randint(1,499)
    rand_y = random.randint(1,399)
    Point = [rand_x, rand_y]

    return Point

## Generates random points based on Circle of Bias
def GenerateRandomPoints(CentX, CentY, R, RobotRadius, DesClearance):
    RandPoints = []
    while len(RandPoints) < 5:
        angle = random.uniform(0, math.pi)
        X = CentX + R*math.cos(angle)
        Y = CentY + R*math.sin(angle)
        if checkValid(X, Y, RobotRadius, DesClearance):
            RandPoints.append([int(X),int(Y)])
        else:
            continue

    return RandPoints

# def GenerateRandomPoints(CentX, CentY, R, RobotRadius, DesClearance):
#     RandPoints = []
#     while len(RandPoints) < 4:
#         angle = random.uniform(0, math.pi)
#         X = CentX + R*math.cos(angle)
#         Y = CentY + R*math.sin(angle)
#         RandPoints.append([int(X),int(Y)])


#     return RandPoints


## Finds nearest point on the tree to a randomly generated point
def FindNearestTreePoint(NodeList, RandomPoint):
    min_dist = math.inf
    for idx, node in enumerate(NodeList):
        NodeState = node.ReturnState()
        dist = np.sqrt((RandomPoint[0]-NodeState[0])**2 + (RandomPoint[1]-NodeState[1])**2)
        if dist < min_dist:
            min_dist = dist
            closest_node_idx = idx
            closest_node = node

    return closest_node_idx, closest_node

#Finds the nearest action that leads you to the random point.
def FindNearestState(NewStateList, RandomPoint):
    min_dist = math.inf
    for idx, state in enumerate(NewStateList):
        dist = np.sqrt((RandomPoint[0]-state[0])**2 + (RandomPoint[1]-state[1])**2)
        if dist < min_dist:
            min_dist = dist
            closest_state_idx = idx
            closest_state = state

    return closest_state, closest_state_idx

#Finds the nearest tree point to the goal.
def FindNearest2Goal(Tree, GoalPoint):
    min_dist = math.inf
    for idx, state in enumerate(Tree):
        dist = np.sqrt((GoalPoint[0]-state[0])**2 + (GoalPoint[1]-state[1])**2)
        if dist < min_dist:
            min_dist = dist
            closest_state_2_goal_idx = idx
            closest_state_2_goal = state

    return closest_state_2_goal, closest_state_2_goal_idx

#Finds the nearest tree point to the goal.
def FindNearestTree2Goal(Tree, GoalPoint):
    min_dist = math.inf
    for idx, node in enumerate(Tree):
        state = node.ReturnState()
        dist = np.sqrt((GoalPoint[0]-state[0])**2 + (GoalPoint[1]-state[1])**2)
        if dist < min_dist:
            min_dist = dist
            closest_state_2_goal_idx = idx
            closest_state_2_goal = state

    return closest_state_2_goal, closest_state_2_goal_idx

#Returns Euclidian Distance in ancestor calculations
def EuclidDist(State1,GoalState):
    dist = np.sqrt((State1[0]-GoalState[1])**2 + (State1[1] - GoalState[1])**2)
    return dist


##---------------------------------Defining my Action Set -----------------------------------------##
''' Though this project is NOT action based planning, I need to limit the graph generation based
off of differential drive constraints. Thus, I reuse this from the previous project.'''

def ReturnPossibleStates(CurrentNodeState, Wheel_RPMS, RobotRadius, ObsClearance, WheelRad, WheelDist):
    RPM1 = Wheel_RPMS[0]
    RPM2 = Wheel_RPMS[1]
    ActionSet = [[RPM1, RPM1], [RPM2,RPM2],[RPM1, RPM2], [RPM2, RPM1], [0,RPM1], [RPM1,0], [0,RPM2], [RPM2,0]] #Differential Drive Action Set
    NewNodeStates = [] #Init List of States

    for action in ActionSet: #For each differential drive action
        NewNodeState, Cost = CalcMoveWithCost(CurrentNodeState, action, RobotRadius, ObsClearance, WheelRad, WheelDist) #Calculate the state and cost
        if NewNodeState is not None:
            NewNodeStates.append([NewNodeState, Cost, action]) #Append Chile Node States
    return NewNodeStates

##---------------------------------Defining my Cost and NewNodeState Function--------------------------------------##
''' Though this project is NOT action based planning, I need to limit the graph generation based
off of differential drive constraints. Thus, I reuse this from the previous project.'''

def CalcMoveWithCost(CurrentNodeState, WheelAction, RobotRadius, ObsClearance, WheelRad, WheelDist):
    t = 0 
    dt = 0.1 
    Curr_Node_X = CurrentNodeState[0] #Grab Current Node X
    Curr_Node_Y = CurrentNodeState[1] #Grad Current Node Y
    Curr_Node_Theta = np.deg2rad(CurrentNodeState[2]) #Grab Current Node Theta, convert to radians.

    MoveCost = 0.0 #Init Cost

    New_Node_X = Curr_Node_X #Set New Node Start Point X
    New_Node_Y = Curr_Node_Y #Set New Node Start Point Y
    New_Node_Theta = Curr_Node_Theta #Set New Node Start Point Theta

    ##----------------Euler Integration to Generate Curvature----------------##
    while t < 1:
        t += dt
        ChangeX = 0.5*WheelRad*(WheelAction[0]+WheelAction[1])*np.cos(New_Node_Theta)*dt
        ChangeY = 0.5*WheelRad*(WheelAction[0]+WheelAction[1])*np.sin(New_Node_Theta)*dt
        ChangeTheta = (WheelRad/WheelDist)*(WheelAction[0]-WheelAction[1])*dt
        
        New_Node_X += ChangeX
        New_Node_Y += ChangeY
        New_Node_Theta += ChangeTheta

        MoveCost += np.sqrt((ChangeX)**2 + (ChangeY)**2)

        ##-----------Why CheckValid is inside the loop---------------##
        '''Inside the loop because if we only checked final, the intermediate steps would sometimes be in the obstacle space.'''
        if checkValid(New_Node_X, New_Node_Y, ObsClearance, RobotRadius) == False:
            return None, None
        
    New_Node_Theta = int(np.rad2deg(New_Node_Theta)) #Convert back to Degrees

    ##-----Wrap to -360-360-----##
    if New_Node_Theta >= 360:
        New_Node_Theta = New_Node_Theta - 360
    if New_Node_Theta < -360:
        New_Node_Theta = New_Node_Theta + 360

    return [New_Node_X, New_Node_Y, New_Node_Theta], MoveCost


## Generate Based off of DiffDrive Curve
def GenerateBranch(Closest_Node, RandomPoint, Closest_Idx, Wheel_RPMS, RobotRadius, ObsClearance, WheelRad, WheelDist):
    ActionStateInfo = ReturnPossibleStates(Closest_Node.ReturnState(), Wheel_RPMS, RobotRadius, ObsClearance, WheelRad, WheelDist)
    ActionStateList = [sub_array[0] for sub_array in ActionStateInfo]
    #print("\nPossible Action States:\n", ActionStateList)
    if ActionStateList:
        Closest_State, Closest_Idx = FindNearestState(ActionStateList, RandomPoint)
        Full_Info = ActionStateInfo[Closest_Idx]
        return Closest_State, Full_Info
    else:
        return False, False

'''For Curves'''
#Relatively the same function as the cost and state function, but with modifications to just plot.
#Plots Curve from Parent to New State
def PlotBranch(ParentNodeState, WheelAction, WheelRad, WheelDist, Color, RobotRadius, ObsClearance):
    t = 0
    dt = 0.1
    Curr_Node_X = ParentNodeState[0]
    Curr_Node_Y = ParentNodeState[1]
    Curr_Node_Theta = np.deg2rad(ParentNodeState[2])

    New_Node_X = Curr_Node_X
    New_Node_Y = Curr_Node_Y
    New_Node_Theta = Curr_Node_Theta

    while t < 1.5:
        t += dt
        X_Start = New_Node_X
        Y_Start = New_Node_Y
        ChangeX = 0.5*WheelRad*(WheelAction[0]+WheelAction[1])*np.cos(New_Node_Theta)*dt
        ChangeY = 0.5*WheelRad*(WheelAction[0]+WheelAction[1])*np.sin(New_Node_Theta)*dt
        ChangeTheta = (WheelRad/WheelDist)*(WheelAction[0]-WheelAction[1])*dt

        New_Node_X += ChangeX
        New_Node_Y += ChangeY
        New_Node_Theta += ChangeTheta
        if checkValid(New_Node_X, New_Node_Y, ObsClearance, RobotRadius) == True:
            plt.plot([X_Start, New_Node_X], [Y_Start, New_Node_Y], color = Color, linewidth = 0.75)


# Defining my compare to goal fxn.
def CompareToGoal(Current_Node_Position, Goal_Node_Position, ErrorThreshold):
    Dist2Goal = (Goal_Node_Position[0] - Current_Node_Position[0])**2 + (Goal_Node_Position[1] - Current_Node_Position[1])**2 #Euclidian Distance
    if Dist2Goal < ErrorThreshold**2: #Error less than threshold PLUS the angle has to be equal
        return True
    else:
        return False







##------------------------------Color Points on the Workspace-----------------------##
def WSColoring(Workspace, Location, Color):
    x,_,_ = Workspace.shape #Get Shape of Workspace
    translation_x = Location[1] #Where in X
    translation_y = Location[0] #Where in Y
    Workspace[translation_x,translation_y,:] = Color #Change the Color to a set Color
    return Workspace  

##------------------------Defining my GetInitialState Function-----------------------##
def GetInitialState():
    print("Enter Initial Node X and Y, and Theta, separated by spaces: ")
    Init_State=[int(x) for x in input().split()]
    return Init_State

##------------------------Defining my GetGoalState Function--------------------------##
def GetGoalState():
    print("Enter Goal Node X and Y, separated by spaces: ")
    Goal_State=[int(x) for x in input().split()]
    return  Goal_State
##-------------------------Defining my Get Robot Radius Function---------------------##
def GetClearance():
    print("Enter Desired Clearance From Obstacles.")
    Clearance=int(input())
    return  Clearance

##--------------------------Defining my GetWheelRPMS Function------------------------##
def GetWheelRPM():
    print("Enter Wheel RPMS, 2 Unique, Separated By Spaces")
    WheelRPMS = [int(x) for x in input().split()]
    return  WheelRPMS



##======================================================"Main Function"========================================================##

##-------Getting Parameters from Burger TurtleBot Dimensions-------##

WheelRadius = 3.8 #cm
RobotRadius = 17.8 #cm
WheelDistance = 35.4 #cm
MaxIterations = 1000

##----------------------Arena Setup-------------------##

arena = np.zeros((400, 500, 3), dtype = "uint8")
InitState = GetInitialState()
GoalState =GetGoalState()
DesClearance = GetClearance()
WheelRPMS = GetWheelRPM()

#-----Check Valid Initial State-------##
if not checkValid(InitState[0], InitState[1], RobotRadius, DesClearance):
    print("Your initial state is inside an obstacle or outside the workspace. Please retry.")
    exit()

##----Check Valid Goal State----------##
if not checkValid(GoalState[0], GoalState[1], RobotRadius, DesClearance):
    print("Your goal state is inside an obstacle or outside the workspace. Please retry.")
    exit()

setup(DesClearance, RobotRadius) #Arena Setup


WSColoring(arena, InitState, (0,255,0)) #Plot Initial State
WSColoring(arena, GoalState, (0,255,0)) #Plot Goal State

plt.imshow(arena, origin='lower') #Show Initial Arena Setup
plt.show()

#Initialization
Check_Goal = False
Init_Node_Temp = Node(InitState, None)
Init_Node = Node(InitState, Init_Node_Temp)

#Init Lists for Plotting
Wheel_CMD_List = []
Explored_Tree = []
random_point_list = []
Explored_Tree.append(Init_Node)
Wheel_CMD_List.append([0,0])
random_point_list.append([0,0])

iteration = 0

##---------------------Thresholds---------------------##
ChangeX = 0.5*WheelRadius*(2*np.max(WheelRPMS))*np.cos(45)
ChangeY = 0.5*WheelRadius*(2*np.max(WheelRPMS))*np.sin(45)
PPRO_Radius = np.sqrt(ChangeX**2 + ChangeY**2)
ErrorThresh = PPRO_Radius


#Start Algorithm
starttime = timeit.default_timer() #Start the Timer when serch starts
print("PPRO RRT Starting!!!!")
Nearest2Goal = Init_Node

while iteration < MaxIterations:
    print("The Nearest State to Goal is:", Nearest2Goal.ReturnState())
    RandomPoints = GenerateRandomPoints(Nearest2Goal.ReturnState()[0], Nearest2Goal.ReturnState()[1], PPRO_Radius, RobotRadius, DesClearance) #Generate Random Points based on Circle of Bias
    Nearest_Rando_2_Goal, Rando_IDX = FindNearest2Goal(RandomPoints, GoalState) #Find nearest random point to goal
    Tree_IDX, NearestTreeNode = FindNearestTreePoint(Explored_Tree, Nearest_Rando_2_Goal) #Find nearest tree point to random 
    Closest_Action_State, Info = GenerateBranch(Nearest2Goal, Nearest_Rando_2_Goal, Tree_IDX, WheelRPMS, RobotRadius, DesClearance, WheelRadius, WheelDistance) #Generate Action that takes you closest to random point

    if Closest_Action_State: #If a valid action exists
        NewBranch = Node(Closest_Action_State, Explored_Tree[Tree_IDX]) #Generate a New Branch
        Explored_Tree.append(NewBranch) #Add for Plotting
        Wheel_CMD_List.append(Info[2])
        random_point_list.append(Nearest_Rando_2_Goal)

        Nearest2GoalSTATE, _ = FindNearestTree2Goal(Explored_Tree, GoalState) #Find New Nearest2Goal Branch Point
        Ancestor1 = Nearest2Goal.ReturnParent()
        Ancestor2 = Nearest2Goal.ReturnParent()
        Ancestor1_Dist = EuclidDist(Ancestor1.ReturnState(), GoalState) #Calculate Ancestor Information
        Ancestor2_Dist = EuclidDist(Ancestor2.ReturnState(), GoalState)
        Nearest2Goal = Node(Nearest2GoalSTATE, Explored_Tree[Tree_IDX]) #Update "Nearest Node to Goal"

        #Compare to Goal
        Check_Goal = CompareToGoal(Nearest2Goal.ReturnState(), GoalState, ErrorThresh)
        if Check_Goal:
            print("Goal Reached!")
            break
    
        iteration += 1
        print(iteration)

    else:
        iteration +=1
        print(iteration)
        continue


stoptime = timeit.default_timer() #Stop the Timer, as Searching is complete.
print("That took", stoptime - starttime, "seconds to complete")



##--------------Visualize--------------##
print("Visualization Starting!")
plt.plot(InitState[0], InitState[1], 'go', markersize = 0.5) #plot init state
plt.imshow(arena, origin = 'lower')

for idx, node in enumerate(Explored_Tree): #Plots the search area
    curr_node_state = node.ReturnState()
    parent_node_state = node.ReturnParentState()
    plt.plot(random_point_list[idx][0], random_point_list[idx][1], 'y+')
    plt.pause(0.5)


    PlotBranch(node.ReturnParentState(), Wheel_CMD_List[idx], WheelRadius, WheelDistance, 'g', RobotRadius, DesClearance)
    plt.pause(0.5)
    
plt.show()
plt.close()


