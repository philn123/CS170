from queue import PriorityQueue
import copy, time, math


# Create this class, recommended by instructions
# Use this class to do trace of program
class Node:
    def __init__(self, puzzle, parent, g, h, action):
        self.puzzle = puzzle
        self.parent = parent
        self.g = g
        self.h = h
        self.action = action

    def get_p(self):
        return self.puzzle

    def get_g(self):
        return self.g

    def get_h(self):
        return self.h

    def get_parent(self):
        return self.parent

    def get_f(self):
        return self.g + self.h

    def get_action(self):
        return self.action


class Problem:
    def __init__(self, init, goal):
        self.init = init
        self.goal = goal

    def get_init(self):
        return self.init

    def get_goal(self):
        return self.goal


def start_8_puzzle():
    print("Welcome to 862050241 8 puzzle solver")
    goal = [[1, 2, 3], [4, 5, 6], [7, 8, 0]]
    # Loop to make sure the user enters a valid choice for puzzle picking
    while True:
        user_input = input("Type 1 to use a default puzzle, or 2 to enter your own puzzle\n")
        if user_input == '1':
            print("You have selected default puzzle.")
            puzzle = select_puzzles()
            break
        elif user_input == '2':
            puzzle = custom_puzzle()
            break
        else:
            print("Please enter a valid input")

    user_alg = select_alg()
    problem = Problem(puzzle, goal)
    solve(problem, user_alg)


# returns the puzzle selected
def select_puzzles():
    while True:
        print("Select difficulty of puzzle going from 1-6 easiest to impossible")
        print("1) Trivial")
        print("2) Very Easy")
        print("3) Easy")
        print("4) Doable")
        print("5) Oh boy")
        print("6) Impossible")
        user_select = input()
        print()

        if user_select == '1':
            return [[1, 2, 3], [4, 5, 6], [7, 8, 0]]
        elif user_select == '2':
            return [[1, 2, 3], [4, 5, 6], [7, 0, 8]]
        elif user_select == '3':
            return [[1, 2, 0], [4, 5, 3], [7, 8, 6]]
        elif user_select == '4':
            return [[0, 1, 2], [4, 5, 3], [7, 8, 6]]
        elif user_select == '5':
            return [[8, 7, 1], [6, 0, 2], [5, 4, 3]]
        elif user_select == '6':
            return [[1, 2, 3], [4, 5, 6], [8, 7, 0]]
        else:
            print("Please enter valid number\n")


# returns number selected
def select_alg():
    while True:
        print("Enter your choice of algorithm")
        print("1) Uniform Cost Search")
        print("2) A* with the Misplaced Tile heuristic")
        print("3) A* with the Euclidean distance heuristic")
        user_select_alg = input()
        print()

        if user_select_alg < '1' or user_select_alg > '3':
            print("Select valid algorithm")
        else:
            return int(user_select_alg)


# this is for entering a custom puzzle from user
# https://www.geeksforgeeks.org/taking-multiple-inputs-from-user-in-python/ - method for taking input
def custom_puzzle():
    puzzle = []
    print("Enter your puzzle, use a zero to represent the blank")
    for x in range(eight_puzzle_size):
        if x == 0:
            a, b, c = [int(y) for y in input("Enter the first row, use space or tabs between numbers  ").split()]
            puzzle.append([a, b, c])
        elif x == 1:
            a, b, c = [int(y) for y in input("Enter the second row, use space or tabs between numbers  ").split()]
            puzzle.append([a, b, c])
        elif x == 2:
            a, b, c = [int(y) for y in input("Enter the third row, use space or tabs between numbers  ").split()]
            puzzle.append([a, b, c])
    return puzzle


def solve(problem, alg_choice):

    puzzle = problem.get_init()
    goal = problem.get_goal()
    # priorityQueue doesn't have sort stability, use entry count as tiebreaker
    entry_count = 0

    initial = Node(puzzle, None, 0, 0, None)
    frontier = PriorityQueue()
    frontier.put((initial.get_f(), entry_count, initial))
    explored = set()  # use a set because its access is faster

    max_nodes_in_queue = 1
    num_of_nodes_expanded = 0
    start = time.time()
    while True:
        if frontier.empty():
            print("ERROR")
            break

        node = frontier.get()  # remove node from frontier

        # checking for goal state after removing from frontier
        # if we check goal when about to explore because we know its most optimal due to queue
        # if we check during node creation, it might be suboptimal
        if node[2].get_p() == goal:
            # return the trace here
            end = time.time()
            print_trace(node, goal, max_nodes_in_queue, num_of_nodes_expanded)
            print_time(start, end, alg_choice)
            break

        # add to explored list
        explored.add(node)

        # return the children Nodes that are found with the node expansion
        # an array of Node objects
        children = node_expansion(node, goal, alg_choice)
        num_of_nodes_expanded += 1
        # make sure these nodes aren't in frontier or explored set
        for child in children:
            entry_count += 1
            if in_frontier(frontier, child) or in_explored(explored, child):
                continue
            frontier.put((child.get_f(), entry_count, child))
            max_nodes_in_queue = max(frontier.qsize(), max_nodes_in_queue)


def node_expansion(current_node, goal, alg_choice):
    # this will contain all the new children Nodes
    children = []

    new_g = current_node[2].get_g()

    current_board = current_node[2].get_p()

    # initializing variables for the pos of zero
    # result stored will be like row x col
    zero_x = None
    zero_y = None

    for i in range(len(current_board)):
        for j in range(len(current_board[i])):
            if current_board[i][j] == 0:
                zero_x = i
                zero_y = j

    # check possible moves that zero can be at this current state
    # check up,right,down,left

    if zero_x > 0:  # up
        # need to swap values, plus deep copy to not modify original
        new_board = copy.deepcopy(current_board)
        new_board[zero_x][zero_y] = new_board[zero_x-1][zero_y]
        new_board[zero_x-1][zero_y] = 0

        # calculate heuristic
        new_h = calculate_heuristic(new_board, goal, alg_choice)

        # create child node, make sure to iterate g
        child = Node(new_board, current_node, new_g+1, new_h, "up")
        children.append(child)

    if zero_y < 2:  # right
        # need to swap values, plus deep copy to not modify original
        new_board = copy.deepcopy(current_board)
        new_board[zero_x][zero_y] = new_board[zero_x][zero_y+1]
        new_board[zero_x][zero_y+1] = 0

        # calculate heuristic
        new_h = calculate_heuristic(new_board, goal, alg_choice)

        # create child node, make sure to iterate g
        child = Node(new_board, current_node, new_g+1, new_h, "right")
        children.append(child)

    if zero_x < 2:  # down
        # need to swap values, plus deep copy to not modify original
        new_board = copy.deepcopy(current_board)
        new_board[zero_x][zero_y] = new_board[zero_x + 1][zero_y]
        new_board[zero_x + 1][zero_y] = 0

        # calculate heuristic
        new_h = calculate_heuristic(new_board, goal, alg_choice)

        # create child node, make sure to iterate g
        child = Node(new_board, current_node, new_g + 1, new_h, "down")
        children.append(child)

    if zero_y > 0:  # left
        # need to swap values, plus deep copy to not modify original
        new_board = copy.deepcopy(current_board)
        new_board[zero_x][zero_y] = new_board[zero_x][zero_y - 1]
        new_board[zero_x][zero_y - 1] = 0

        # calculate heuristic
        new_h = calculate_heuristic(new_board, goal, alg_choice)

        # create child node, make sure to iterate g
        child = Node(new_board, current_node, new_g + 1, new_h, "left")
        children.append(child)

    return children


# board - current board state
# goal - what the goal state looks light
# algorithm - numbers: 1 is uniform, 2 is misplaced, 3 is euclidean
def calculate_heuristic(board, goal, algorithm):
    heuristic_number = 0
    if algorithm == 1:
        heuristic_number = 0
    elif algorithm == 2:
        for i in range(eight_puzzle_size):
            for j in range(eight_puzzle_size):
                if board[i][j] != goal[i][j] and board[i][j] != 0:
                    heuristic_number += 1
    # euclidean - checks how finished the board state is
    elif algorithm == 3:
        for i in range(eight_puzzle_size):
            for j in range(eight_puzzle_size):
                val = board[i][j]  # position of num on current board
                if val != 0:
                    for x in range(eight_puzzle_size):
                        for y in range(eight_puzzle_size):
                            if goal[x][y] == val:  # look for position of num on goal board and calc distance
                                heuristic_number += math.sqrt(pow(x-i, 2) + pow(y-j, 2))  # equation for euclidean distance

    return heuristic_number


# checking if child is inside frontier
# True means it is in frontier, false means it is not
def in_frontier(frontier, child):
    # if it's empty, means it can't be in it
    if frontier.empty():
        return False

    # underlying implementation is a list of tuples
    temp = frontier.queue
    for item in temp:
        # the third part of tuple has the node inside it
        if item[2].get_p() == child.get_p():
            return True

    return False


# iterate over set and access the puzzle in each node
# to check if it has already been explored
def in_explored(explored, child):
    for item in explored:
        if item[2].get_p() == child.get_p():
            return True

    return False


def print_trace(node, goal, max_nodes_in_queue, num_of_nodes_expanded):
    # temp holder
    temp = node
    stack = []

    # we push current node onto stack and start moving up the tree
    # while loop exits on initial state, so we just push initial state
    while temp[2].get_parent() is not None:
        stack.append(temp)
        temp = temp[2].get_parent()
    stack.append(temp)

    stack.reverse()
    # begin printing trace
    print("Expanding state")
    temp = stack[0]
    print_board(temp[2].get_p())
    stack.pop(0)

    for val in stack:
        print(
            "The best state to expand with g(n) = " + str(val[2].get_g()) + " h(n) = " + str(val[2].get_h()) + " is...")

        print("The zero moved " + val[2].get_action() + " from the previous state.")
        print_board(val[2].get_p())
        if val[2].get_p() != goal:
            print("Expanding this node\n")
        else:
            print("You have reached the goal.")

    print("GOAL!")
    print("To solve this problem, the search algorithm expanded a total of " + str(num_of_nodes_expanded) + " nodes.")
    print("The maximum number of nodes in the queue at any one time: " + str(max_nodes_in_queue))


def print_board(board):
    for row in board:
        print(row)
    print()


def print_time(start, end, alg_choice):
    if alg_choice == 1:
        print("Uniform cost search took " + str(end-start) + " seconds.")
    if alg_choice == 2:
        print("A* with the Misplaced Tile heuristic took " + str(end-start) + " seconds.")
    if alg_choice == 3:
        print("A* with the Euclidean distance heuristic " + str(end-start) + " seconds.")


# location of main driver code
eight_puzzle_size = 3
start_8_puzzle()


