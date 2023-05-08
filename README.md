# ENPM661_Project_5
UMD ENPM661 Project 5: Implementation of PPRO RRT

# Student Information
Brendan Neal:

Directory ID: bneal12.

UID: 119471128.

Adam Lobo:

Directory ID: alobo.

UID: 115806078.

# Project Information
Implement a cutting edge path planning technique from literature published later than 2018. The path planning we chose is called PPRO RRT. This technique is implemented with a custom made obstacle space as well as in a different 3-D obstacle space. We are testing this technique's speed and cost versus the A* algorithm. The paper can be found here:

https://drive.google.com/file/d/1dyK7FZaS1Vq_c-HZibgc8_yRqxTh3m1-/view?usp=sharing

Python Version: 3

# GitHub Repository Links

Brendan: https://github.com/brendanneal12/ENPM661_Project_5

Adam: https://github.com/AdazInventionz/ENPM-661-Project-5

# Libraries Used
opencv, numpy, timeit, from matplotlib: pyplot, math, random

# Files:
Map1/PPRO_RRT.py
Map2/PPRO_RRT.py
Blank/PPRO_RRT.py

# Important Notes Please Read:
1. If you have not read the report yet, I highly suggest you do, as the findings from this project state that more research and work has to be done to make this path planning algorithm work with a differential drive robot.


# How to Run Code
1. Download Files
2. Prompted by the terminal, enter the initial state X, Y, and Theta with spaces separating them. Example: 50 100 0
3. Prompted by the terminal, enter the goal state X and Y, with spaces separating them. Example: 500 100
4. Prompted by the terminal, enter your desired clearance from obstacles. Example: 5
5. Prompted by the terminal, enter 2 unique wheel RPMS, separated by spaces. Example: 12 10
6. If your initial or goal state is inside an obstacle or outside of the workspace, you will be prompted to restart.
7. Observe the obstacle space setup. Obstacles are white, the entered desired clearance is light gray, and the additional robot radius is dark gray.
8. Close the window to begin PPRO RRT
9. While the search is running, the closest node to goal is printed to terminal.
10. Once the goal is reached, the time to complete search and iteration number will be printed to the terminal and visualization begins.
11. Once the visualization is complete, close the window to end the program.










