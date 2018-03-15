# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:
    """
    #print "Start:", problem.getStartState()
    #print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    #print "Start's successors:", problem.getSuccessors(problem.getStartState())

    "*** YOUR CODE HERE ***"

    fringe = util.Stack()  # stack is used for Depth First Search as we need a LIFO Data structure
    action = []  # List of actions to be returned by this function
    visited = {}  # Dictionary to keep track of the visited states
    state = problem.getStartState()  # Getting the start state of the problem
    fringe.push([(state, None, 0)])  # Pushing the start state along with action and cost as a tuple into the stack
    while not fringe.isEmpty():  # We will run the algorithm till the stack is empty
        state_list = fringe.pop()  # Pop the list of states from the stack
        state = state_list[len(state_list)-1]  # Get the last tuple in the state list
        if problem.isGoalState(state[0]):  # Check whether the state is a goal state
            for tuples in state_list[1:]:
                action.append(tuples[1])  # If Goal State get the list of actions from the popped states list
            return action
        if state[0] not in visited:  # Check whether the state is visited or not
            visited[state[0]] = True  # If not make the state as visited
            for child in problem.getSuccessors(state[0]):  # Get the successors of the current state
                temp_list = state_list[:]
                temp_list.append(child)  # Append the child tuple into the state list popped
                fringe.push(temp_list)  # Push the modified list into the stack


def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    fringe = util.Queue()  # Queue is used for Breadth First Search as we need a FIFO Data structure
    action = []  # List of actions to be returned by this function
    visited = {}  # Dictionary to keep track of the visited states
    state = problem.getStartState()  # Getting the start state of the problem
    fringe.push([(state, None, 0)])  # Pushing the start state along with action and cost as a tuple into the Queue
    while not fringe.isEmpty():  # We will run the algorithm till the Queue is empty
        state_list = fringe.pop()  # Pop the list of states from the Queue
        state = state_list[len(state_list) - 1]  # Get the last tuple in the state list
        if problem.isGoalState(state[0]):  # Check whether the state is a goal state
            for tuples in state_list[1:]:
                action.append(tuples[1])  # If Goal State get the list of actions from the popped states list
            return action
        if state[0] not in visited:  # Check whether the state is visited or not
            visited[state[0]] = True  # If not make the state as visited
            for child in problem.getSuccessors(state[0]):  # Get the successors of the current state
                temp_list = state_list[:]
                temp_list.append(child)  # Append the child tuple into the state list popped
                fringe.push(temp_list)  # Push the modified list into the Queue


def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    fringe = util.PriorityQueue()  # Priority Queue is used for Uniform cost search with cost as priority
    action = []  # List of actions to be returned by this function
    visited = {}  # Dictionary to keep track of the visited states
    state = problem.getStartState()  # Getting the start state of the problem
    fringe.push([(state, None, 0)], 0)  # Pushing the start state along with action and cost as a tuple, cost to reach that state into the Priority Queue
    while not fringe.isEmpty():  # We will run the algorithm till the Priority Queue is empty
        state_list = fringe.pop()  # Pop the list of states from the Priority Queue
        state = state_list[len(state_list) - 1]  # Get the last tuple in the state list
        if problem.isGoalState(state[0]):  # Check whether the state is a goal state
            for tuples in state_list[1:]:
                action.append(tuples[1])  # If Goal State get the list of actions from the popped states list
            return action
        if state[0] not in visited:  # Check whether the state is visited or not
            visited[state[0]] = True  # If not make the state as visited
            for child in problem.getSuccessors(state[0]):
                cost = state[2] + child[2]  # Calculate the cost to reach the child node
                s = list(child)
                s[2] = cost  # Update the cost in the state list
                child = tuple(s)
                temp_list = state_list[:]
                temp_list.append(child)  # Append the child tuple into the state list popped
                fringe.push(temp_list, cost)  # Push the modified list into the Queue along with the modified cost


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    fringe = util.PriorityQueue()  # Priority Queue is used for Uniform cost search with cost as priority
    action = []  # List of actions to be returned by this function
    visited = {}  # Dictionary to keep track of the visited states
    state = problem.getStartState()  # Getting the start state of the problem
    priority = heuristic(state, problem)  # Calculating the priority of the state using heuristic function
    fringe.push([(state, None, 0)], priority)  # Pushing the start state along with action and cost as a tuple, priority into the Priority Queue
    while not fringe.isEmpty():  # We will run the algorithm till the Priority Queue is empty
        state_list = fringe.pop()  # Pop the list of states from the Priority Queue
        state = state_list[len(state_list) - 1]  # Get the last tuple in the state list
        if problem.isGoalState(state[0]):  # Check whether the state is a goal state
            for tuples in state_list[1:]:
                action.append(tuples[1])  # If Goal State get the list of actions from the popped states list
            return action
        if state[0] not in visited:  # Check whether the state is visited or not
            visited[state[0]] = True  # If not make the state as visited
            for child in problem.getSuccessors(state[0]):
                cost = state[2] + child[2]  # Calculate the cost to reach the child node
                priority = cost + heuristic(child[0], problem)  # Calculate the priority of the child state
                s = list(child)
                s[2] = cost  # Update the cost in the state list
                child = tuple(s)
                temp_list = state_list[:]
                temp_list.append(child)  # Append the child tuple into the state list popped
                fringe.push(temp_list, priority)  # Push the modified list into the Queue along with the modified cost

# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
