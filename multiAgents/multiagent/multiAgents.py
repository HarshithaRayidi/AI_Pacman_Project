# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        foodDistances = []
        ghostDistances = []
        capsuleDistances = []

        capsulesList = successorGameState.getCapsules()
        score = successorGameState.getScore()

        for food in newFood.asList():
            distance = util.manhattanDistance(food, newPos)
            foodDistances.append(distance)
        if len(foodDistances) > 0:
            minFoodDistance = min(foodDistances) # calculate min distance to food from the new position
            score = score + 5.0/minFoodDistance # when approaching food the score should increase

        if (currentGameState.getNumFood() > successorGameState.getNumFood()):
            score = score + 150.0

        for ghost in newGhostStates:
            distance = util.manhattanDistance(ghost.getPosition(),newPos)
            ghostDistances.append(distance)
        if len(ghostDistances) > 0:
            minGhostDistance = min(ghostDistances)
            if minGhostDistance <= 1:
                return -9999 # if ghost is near the score should be least
            else:
                score = score - 10.0/minGhostDistance # for ghosts the score should decrease

        for capsule in capsulesList:
            distance = util.manhattanDistance(capsule,newPos)
            capsuleDistances.append(distance)
        if len(capsuleDistances) > 0:
            minCapsuleDistance = min(capsuleDistances)
            score = score + 1.0/minCapsuleDistance # when capsules are eaten then the score should increase

        return score

def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        "*** YOUR CODE HERE ***"
        def maxValue (gameState, depth):
            val = []
            maxVal = 0
            if depth ==0 or gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState)
            legalMoves = gameState.getLegalActions(0)
            for move in legalMoves:
                nextState = gameState.generateSuccessor(0,move)
                minVal = minValue(nextState, depth, 1) # finding the min values of the agents
                val.append(minVal)
            maxVal = max(val) # finding the max value of all the successor values
            return maxVal

        def minValue(gameState, depth, agents):
            val = []
            minVal = 0
            if depth == 0 or gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState)
            legalMoves = gameState.getLegalActions(agents)
            for move in legalMoves:
                nextState = gameState.generateSuccessor(agents, move)
                if agents < gameState.getNumAgents()-1: # there are still agents in min level
                    minVal = minValue(nextState, depth, agents+1)
                    val.append(minVal)
                else: # no more agents in min level so going to next max level
                    maxVal = maxValue(nextState, depth-1)
                    val.append(maxVal)
            minVal = min(val)
            return minVal

        legalMoves = gameState.getLegalActions(0)
        val = -99999
        action = Directions.STOP
        for move in legalMoves:
            nextState = gameState.generateSuccessor(0, move)
            minMaxVal = minValue(nextState, self.depth, 1)
            if minMaxVal > val:
                val =  minMaxVal
                action = move

        return action

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        def maxValue (gameState, depth, alpha, beta):
            maxVal = float("-inf")
            if depth ==0 or gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState)
            legalMoves = gameState.getLegalActions(0)
            for move in legalMoves:
                nextState = gameState.generateSuccessor(0, move)
                min_Val = minValue(nextState, depth, 1, alpha, beta)
                maxVal = max(maxVal, min_Val)
                if maxVal > beta: # pruning
                    return maxVal
                alpha = max(alpha, maxVal)
            return maxVal


        def minValue(gameState, depth, agents, alpha, beta):
            minVal = float("inf")
            if depth == 0 or gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState)
            legalMoves = gameState.getLegalActions(agents)
            for move in legalMoves:
                nextState = gameState.generateSuccessor(agents, move)
                if agents < gameState.getNumAgents()-1:
                    min_Val = minValue(nextState, depth, agents+1, alpha, beta)
                    minVal = min(minVal, min_Val)
                else:
                    max_Val = maxValue(nextState, depth-1, alpha, beta)
                    minVal = min(minVal, max_Val)
                if minVal < alpha: # pruning
                    return minVal
                beta = min(beta, minVal)
            return minVal

        legalMoves = gameState.getLegalActions(0)
        alpha = float("-inf")
        beta = float("inf")
        action = Directions.STOP
        for move in legalMoves:
            nextState = gameState.generateSuccessor(0, move)
            minMaxVal = minValue(nextState, self.depth, 1, alpha, beta)
            if minMaxVal > alpha:
                alpha = minMaxVal
                action = move

        return action

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        def maxValue (gameState, depth):
            val = []
            maxVal = 0
            if depth ==0 or gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState)
            legalMoves = gameState.getLegalActions(0)
            for move in legalMoves:
                nextState = gameState.generateSuccessor(0,move)
                minVal = expValue(nextState, depth, 1)
                val.append(minVal)
            maxVal = max(val)
            return maxVal

        def expValue(gameState, depth, agents):
            value = 0
            if depth == 0 or gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState)
            legalMoves = gameState.getLegalActions(agents)
            prob = float(1) / len(legalMoves) # probability = 1/ number of actions possible in that level
            for move in legalMoves:
                nextState = gameState.generateSuccessor(agents, move)
                if agents < gameState.getNumAgents()-1:
                    expVal = expValue(nextState, depth, agents+1)
                    value = value + prob * expVal
                else:
                    maxVal = maxValue(nextState, depth-1)
                    value = value + prob * maxVal
            return value

        legalMoves = gameState.getLegalActions(0)
        val = -99999
        action = Directions.STOP
        for move in legalMoves:
            nextState = gameState.generateSuccessor(0, move)
            minMaxVal = expValue(nextState, self.depth, 1)
            if minMaxVal > val:
                val = minMaxVal
                action = move

        return action


def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"

    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    "*** YOUR CODE HERE ***"
    foodDistances = []
    ghostDistances = []
    nonScaryGhosts = 0
    score =currentGameState.getScore()

    for food in newFood.asList():
        distance = util.manhattanDistance(food, newPos)
        foodDistances.append(distance)
    if len(foodDistances) > 0:
        minFoodDistance = min(foodDistances)
        score = score + 10.0 / minFoodDistance

    for ghost in newGhostStates:
        if ghost.scaredTimer == 0:
            nonScaryGhosts += 1
        distance = util.manhattanDistance(ghost.getPosition(), newPos)
        ghostDistances.append(distance)
    if len(ghostDistances) > 0:
        minGhostDistance = min(ghostDistances)
        if minGhostDistance <= 1:
            return -9999
        else:
            score = score - 1.0 / minGhostDistance - 20 * nonScaryGhosts

    return score


# Abbreviation
better = betterEvaluationFunction

