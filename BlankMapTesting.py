#Brendan Neal and Adam Lobo
#ENPM661 Project 5 -- Implementation of PPRO RRT Algorithm in 2D

##======================Importing Libraries=============================##
import cv2 as cv
import numpy as np
import timeit
import sys
import math
from matplotlib import pyplot as plt
import random

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

def FindNearestState(NewStateList, RandomPoint):
    min_dist = math.inf
    for idx, state in enumerate(NewStateList):
        dist = np.sqrt((RandomPoint[0]-state[0])**2 + (RandomPoint[1]-state[1])**2)
        if dist < min_dist:
            min_dist = dist
            closest_state_idx = idx
            closest_state = state

    return closest_state, closest_state_idx

##---------------------------------Defining my Action Set -----------------------------------------##
''' Though this project is not necessarily action based planning, I need to limit the graph generation based
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
''' Though this project is not necessarily action based planning, I need to limit the graph generation based
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
    print("\nPossible Action States:\n", ActionStateList)
    Closest_State, Closest_Idx = FindNearestState(ActionStateList, RandomPoint)
    Full_Info = ActionStateInfo[Closest_Idx]

    return Closest_State, Full_Info

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

    while t < 1:
        t += dt
        X_Start = New_Node_X
        Y_Start = New_Node_Y
        ChangeX = 0.5*WheelRad*(WheelAction[0]+WheelAction[1])*np.cos(New_Node_Theta)*dt
        ChangeY = 0.5*WheelRad*(WheelAction[0]+WheelAction[1])*np.sin(New_Node_Theta)*dt
        ChangeTheta = (WheelRad/WheelDist)*(WheelAction[0]-WheelAction[1])*dt

        New_Node_X += ChangeX
        New_Node_Y += ChangeY
        New_Node_Theta += ChangeTheta
        plt.plot([X_Start, New_Node_X], [Y_Start, New_Node_Y], color = Color, linewidth = 0.75)


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

##---------------------Thresholds---------------------##
ErrorThresh = 3

##----------------------Arena Setup-------------------##

arena = np.zeros((400, 500, 3), dtype = "uint8")
InitState = GetInitialState()
GoalState =GetGoalState()
DesClearance = GetClearance()
WheelRPMS = GetWheelRPM()


WSColoring(arena, InitState, (0,255,0)) #Plot Initial State
WSColoring(arena, GoalState, (0,255,0)) #Plot Goal State

plt.imshow(arena, origin='lower') #Show Initial Arena Setup
plt.show()


Check_Goal = False
Init_Node_Temp = Node(InitState, None)
Init_Node = Node(InitState, Init_Node_Temp)

Wheel_CMD_List = []
Explored_Tree = []
random_point_list = []

Explored_Tree.append(Init_Node)
Wheel_CMD_List.append([0,0])
random_point_list.append([0,0])

iteration = 0

while not Check_Goal:
    Rand_Point = GenerateRandomPoint()
    random_point_list.append(Rand_Point)
    WSColoring(arena, Rand_Point, (255,0,0)) #Plot Initial State
    Tree_Idx, NearestTreeNode = FindNearestTreePoint(Explored_Tree, Rand_Point)
    print("\nRandom Point:", Rand_Point, "Nearest Tree State:", NearestTreeNode.ReturnState())
    Closest_Action_State, Info = GenerateBranch(NearestTreeNode, Rand_Point, Tree_Idx, WheelRPMS, RobotRadius, DesClearance, WheelRadius, WheelDistance)
    NewBranch = Node(Closest_Action_State, Explored_Tree[Tree_Idx])
    print("\nNewest Branch State:", NewBranch.ReturnState())
    PlotBranch(NewBranch.ReturnParentState(), Info[2], WheelRadius, WheelDistance, 'g', RobotRadius, DesClearance)
    Explored_Tree.append(NewBranch)
    Wheel_CMD_List.append(Info[2])
    Check_Goal = CompareToGoal(NewBranch.ReturnState(), GoalState, ErrorThresh)
    iteration += 1

    if iteration > 50:
        print("Iteration Limit Reached!")
        break

print(len(random_point_list), len(Explored_Tree))

plt.imshow(arena, origin= 'lower')
plt.show()

print("Visualization Starting!")
plt.plot(InitState[0], InitState[1], 'go', markersize = 0.5) #plot init state
plt.imshow(arena, origin = 'lower')

for idx, node in enumerate(Explored_Tree): #Plots the search area
    curr_node_state = node.ReturnState()
    parent_node_state = node.ReturnParentState()
    plt.plot(random_point_list[idx][0], random_point_list[idx][1], 'y+')
    plt.pause(1)


    PlotBranch(node.ReturnParentState(), Wheel_CMD_List[idx], WheelRadius, WheelDistance, 'g', RobotRadius, DesClearance)
    plt.pause(1)
    
plt.show()
plt.close()


