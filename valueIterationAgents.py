# valueIterationAgents.py
# -----------------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and Pieter 
# Abbeel in Spring 2013.
# For more info, see http://inst.eecs.berkeley.edu/~cs188/pacman/pacman.html

import mdp, util

from learningAgents import ValueEstimationAgent

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0

        # Write value iteration code here
        cur_buffer = util.Counter()
        next_buffer = util.Counter()
       
        def getNextVal(state):
            try:
                return max(map(lambda action:
                        sum(map(lambda nStateAndProb:
                                nStateAndProb[1] * # T(s, a, s')
                                (mdp.getReward(state, action, nStateAndProb[0]) + discount * cur_buffer[nStateAndProb[0]]), # R(s, a, s') + gamma * V^(k-1)(s')
                            mdp.getTransitionStatesAndProbs(state, action))),
                        mdp.getPossibleActions(state)))
            except ValueError: # terminal state
                return 0

        for i in range(iterations):
            for state in mdp.getStates():
                next_buffer[state] = getNextVal(state)
            cur_buffer, next_buffer = next_buffer, cur_buffer

        self.values = cur_buffer


    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        return sum(map(lambda nStateAndProb:
            nStateAndProb[1] *
            (self.mdp.getReward(state, action, nStateAndProb[0]) + self.discount * self.values[nStateAndProb[0]]),
            self.mdp.getTransitionStatesAndProbs(state, action)))

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        try:
            return max(self.mdp.getPossibleActions(state), key=lambda action: self.computeQValueFromValues(state, action))
        except ValueError: # terminal state
            return None

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)
