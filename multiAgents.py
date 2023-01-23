"""
Introduction to Artificial Intelligence, 89570, Bar Ilan University, ISRAEL

Student name: Tal Ishon

"""

# multiAgents.py
# --------------
# Attribution Information: part of the code were created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# http://ai.berkeley.edu.
# We thank them for that! :)


import util

from connect4 import Agent


def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    """
    return currentGameState.getScore()


def get_children_from_state(index, state):
    actions = state.getLegalActions()
    children = []  # will hold all possible states from legal actions

    # create a tuple of state and action that brought us to this state
    for a in actions:  # a refers to an action from legal actions list
        # add tuple to list
        children.append((state.generateSuccessor(index, a), a))  # 1: index of AI agent

    return children  # list of tuples -> for c in children: c[0]: state, c[1]: action


class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxAgent, AlphaBetaAgent & ExpectimaxAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent is another abstract class.
    """

    def __init__(self, evalFn='scoreEvaluationFunction', depth='4'):
        super().__init__()
        self.index = 1  # agent is always index 1
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class BestRandom(MultiAgentSearchAgent):

    def getAction(self, gameState):
        return gameState.pick_best_move()


class MinimaxAgent(MultiAgentSearchAgent):

    def getAction(self, gameState):

        def rb_minimax(state, agent, depth):
            if state.is_terminal() or depth == 0:
                return self.evaluationFunction(state), None

            turn = state.turn
            children = get_children_from_state(self.index, state)

            if turn == agent:
                max_val = float("-inf")
                action = None
                # go over all states to find which action will give highest value
                for c, a in children:
                    c.switch_turn(c.turn)
                    # func returns tuple of value and action
                    v = rb_minimax(c, agent, depth - 1)[0]  # take only first value of tuple
                    if v > max_val:
                        max_val = v
                        action = a

                return max_val, action

            else:
                min_val = float("inf")
                action = None
                # go over all states to find which action will give highest value
                for c, a in children:
                    c.switch_turn(c.turn)
                    # func returns tuple of value and action
                    v = rb_minimax(c, agent, depth - 1)[0]  # take only first value of tuple
                    if v < min_val:
                        min_val = v
                        action = a
                return min_val, action

        # return tuple's (returned value from func) second value which is the action that gave us highest score
        return rb_minimax(gameState, self.index, self.depth)[1]


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
        Your minimax agent with alpha-beta pruning (question 2)
    """

    def getAction(self, gameState):

        def alpha_beta_puring(state, depth):
            action = max_value(state, float("-inf"), float("inf"), depth)
            return action

        def max_value(state, alpha, beta, depth):
            if state.is_terminal() or depth == 0:
                return self.evaluationFunction(state), None

            v, max_val = float("-inf"), float("-inf")
            children = get_children_from_state(self.index, state)

            action = None  # initialize action option to return as approximated best action

            # go over all states to find which action will give highest value
            for c, a in children:  # c: state, a: action
                c.switch_turn(c.turn)
                max_val = max(v, min_value(c, alpha, beta, depth - 1)[0])
                if max_val > beta:  # beta puring
                    return max_val, a
                else:
                    alpha = max(alpha, v)
                    if v < max_val:  # in case no puring -> hold max value of all values to update recommended action
                        v = max_val
                        action = a
            return v, action

        def min_value(state, alpha, beta, depth):
            if state.is_terminal() or depth == 0:
                return self.evaluationFunction(state), None

            v = float("inf")
            children = get_children_from_state(self.index, state)

            action = None  # initialize action option to return as approximated best action

            # go over all states to find which action will give highest value
            for c, a in children:  # c: state, a: action
                c.switch_turn(c.turn)
                min_val = min(v, max_value(c, alpha, beta, depth - 1)[0])
                if min_val < alpha:  # alpha puring
                    return min_val, a
                else:
                    beta = min(beta, v)
                    if v > min_val:  # in case no puring -> hold min value of all values to update recommended action
                        v = min_val
                        action = a

            return v, action

        # return tuple's (returned value from func) second value which is the action that gave us highest score
        return alpha_beta_puring(gameState, self.depth)[1]


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction
        """

        def expectimax(state, depth, max_player):
            if state.is_terminal() or depth == 0:
                return self.evaluationFunction(state), None

            if max_player:
                return max_value(state, depth)

            else:
                return exp_value(state, depth)

        def max_value(state, depth):
            v = float("-inf")
            children = get_children_from_state(self.index, state)

            action = None  # initialize action option to return as approximated best action

            # go over all states to find which action will give highest value
            for c, a in children:  # c: state, a: action
                c.switch_turn(c.turn)
                value = max(v, expectimax(c, depth - 1, False)[0])
                if value > v:  # update v to always hold max value
                    v = value
                    action = a

            return v, action

        def exp_value(state, depth):
            v = 0
            children = get_children_from_state(self.index, state)  # children: list of tuples

            p = 1 / len(children)

            # go over all children to calculate exp value of all states
            for c, a in children:  # c: state, a: action
                c.switch_turn(c.turn)
                v += expectimax(c, depth - 1, True)[0] * p

            return v, None

        # return tuple's (returned value from func) second value which is the action that gave us highest score
        return expectimax(gameState, self.depth, True)[1]
