from rubik.cube import Cube
import numpy as np
#import tensorflow as tf

SOLVED_STATE = "OOOOOOOOOYYYWWWGGGBBBYYYWWWGGGBBBYYYWWWGGGBBBRRRRRRRRR"
actions = ('L', 'Li', 'R', 'Ri', 'F', 'Fi', 'B', 'Bi', 'U', 'Ui', 'D', 'Di')
'''
starting from solved state, randomly move 100 times to get 100 different states and the depth associated. the depth will be turned 
into a weight to the loss later during the training.

for each state in the set, do breadth-1 search and get the 12 different sub states.

inference nn to get the policy and value of the parent node
inference all 12 sub nodes to get each children value and children's policy

loss =  
batch_loss = sum(weight * loss)
'''
def data_generator():
    c = Cube(SOLVED_STATE)
    print_cube(c)
    for x in np.random.randint(0,12,100):
        getattr(c, actions[x])() #scramble cube
        print(f"action={actions[x]}")
        print_cube(c)
        str = c.flat_str()
        print("12 children")
        for a in actions:
            c_working = Cube(str)
            getattr(c_working, a)()
            print_cube(c_working)
        print("--------------------")


class bcolors:
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    WHITE = '\033[37m'
    MAGENTA = '\033[35m'
    BLACK = '\033[30m'

def terminal_str(s):
    s = s.replace("R", bcolors.RED+"R")
    s = s.replace("G", bcolors.GREEN+"G")
    s = s.replace("B", bcolors.BLUE+"B")
    s = s.replace("W", bcolors.BLACK+"W")
    s = s.replace("O", bcolors.MAGENTA+"O")
    s = s.replace("Y", bcolors.YELLOW+"Y")
    return s

def print_cube(c):
    s = c.flat_str()

    index = 0
    for i in range(3):
        print(" "*3 + " " + terminal_str(s[index:index+3]))
        index += 3

    for i in range(3):
        for j in range(4):
            print(terminal_str(s[index:index+3] + " "), end='')
            index += 3
        print('')

    for i in range(3):
        print(" "*3 + " " + terminal_str(s[index:index+3]))
        index += 3
    print('')

if __name__ == "__main__":
    #print(f"{bcolors.RED}Warning: {bcolors.MAGENTA}No {bcolors.YELLOW}active {bcolors.GREEN}frommets {bcolors.BLUE}remain. {bcolors.BLACK}Continue?")
    data_generator()
    #c = Cube(SOLVED_STATE)
    #print_cube(c)
    #c.L()
    #print_cube(c)
