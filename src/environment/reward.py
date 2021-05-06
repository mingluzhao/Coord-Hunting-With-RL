import numpy as np
import random


def sampleFromDistribution(distribution):
    hypotheses = list(distribution.keys())
    probs = list(distribution.values())
    normlizedProbs = [prob / sum(probs) for prob in probs]
    selectedIndex = list(np.random.multinomial(1, normlizedProbs)).index(1)
    selectedHypothesis = hypotheses[selectedIndex]
    return selectedHypothesis


def computeVectorNorm(vector):
    return np.power(np.power(vector, 2).sum(), 0.5)


class GetAgentsPercentageOfRewards:
    def __init__(self, selfishIndex, collisionMinDist):
        self.selfishIndex = selfishIndex
        self.collisionMinDist = collisionMinDist
        self.getPercent = lambda dist: (dist + 1 - self.collisionMinDist) ** (-self.selfishIndex)
        self.individualReward = (self.selfishIndex > 100)
        # for computational purposes, selfish index > 100 is taken as purely selfish

    def __call__(self, agentsDistanceList, predatorID):
        if self.individualReward:
            percentage = np.zeros(len(agentsDistanceList))
            percentage[predatorID] = 1
            return percentage

        percentageRaw = [self.getPercent(dist) for dist in agentsDistanceList]
        percentage = np.array(percentageRaw)/ np.sum(percentageRaw)

        return percentage


class GetCollisionPredatorReward:
    def __init__(self, biteReward, killReward, killProportion, sampleFromDistribution, terminalCheck):
        self.biteReward = biteReward
        self.killReward = killReward
        self.killProportion = killProportion
        self.sampleFromDistribution = sampleFromDistribution
        self.terminalCheck = terminalCheck

    def __call__(self, numPredators, killRewardPercent, collisionID):
        if self.terminalCheck.terminal: # prey already killed
            return [0]* numPredators

        isKill = self.sampleFromDistribution({1: self.killProportion, 0: 1-self.killProportion})
        if isKill:
            reward = self.killReward * np.array(killRewardPercent)
            self.terminalCheck.isTerminal()
        else:
            reward = [0]* numPredators
            reward[collisionID] = self.biteReward
        return reward


class GetPredatorPreyDistance:
    def __init__(self, computeVectorNorm, getPosFromState):
        self.computeVectorNorm = computeVectorNorm
        self.getPosFromState = getPosFromState

    def __call__(self, predatorsStates, preyState):
        predatorsPosList = [self.getPosFromState(predatorState) for predatorState in predatorsStates]
        preyPos = self.getPosFromState(preyState)
        dists = [self.computeVectorNorm(np.array(preyPos) - np.array(predatorPos)) for predatorPos in predatorsPosList]

        return dists


class TerminalCheck(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.terminal = False

    def isTerminal(self):
        self.terminal = True


class RewardPredatorsWithKillProb:
    def __init__(self, predatorsID, preyGroupID, entitiesSizeList, isCollision, terminalCheck,
                 getPredatorPreyDistance, getAgentsPercentageOfRewards, getCollisionPredatorReward):
        self.predatorsID = predatorsID
        self.preyGroupID = preyGroupID
        self.entitiesSizeList = entitiesSizeList

        self.isCollision = isCollision
        self.terminalCheck = terminalCheck
        self.getPredatorPreyDistance = getPredatorPreyDistance
        self.getAgentsPercentageOfRewards = getAgentsPercentageOfRewards
        self.getCollisionPredatorReward = getCollisionPredatorReward

    def __call__(self, state, action, nextState):
        self.terminalCheck.reset()

        predatorsNextState = [nextState[predatorID] for predatorID in self.predatorsID]
        numPredators = len(self.predatorsID)
        rewardList = np.zeros(numPredators)

        for preyID in self.preyGroupID:
            preyNextState = nextState[preyID]
            preySize = self.entitiesSizeList[preyID]
            predatorsPreyDistance = self.getPredatorPreyDistance(predatorsNextState, preyNextState)

            # randomly order predators so that when more than one predator catches the prey, random one samples first
            predatorsID = self.predatorsID.copy()
            random.shuffle(predatorsID)

            for predatorID in predatorsID:
                predatorSize = self.entitiesSizeList[predatorID]
                predatorNextState = nextState[predatorID]

                if self.isCollision(predatorNextState, preyNextState, predatorSize, preySize):
                    killRewardPercent = self.getAgentsPercentageOfRewards(predatorsPreyDistance, predatorID)
                    predatorReward = self.getCollisionPredatorReward(numPredators, killRewardPercent, predatorID)
                    rewardList = rewardList + np.array(predatorReward)

        return rewardList


class RewardPrey:
    def __init__(self, predatorsID, preyGroupID, entitiesSizeList, getPosFromState, isCollision, punishForOutOfBound,
                 collisionPunishment):
        self.predatorsID = predatorsID
        self.getPosFromState = getPosFromState
        self.entitiesSizeList = entitiesSizeList
        self.preyGroupID = preyGroupID
        self.isCollision = isCollision
        self.collisionPunishment = collisionPunishment
        self.punishForOutOfBound = punishForOutOfBound

    def __call__(self, state, action, nextState):
        reward = []
        for preyID in self.preyGroupID:
            preyReward = 0
            preyNextState = nextState[preyID]
            preyNextPos = self.getPosFromState(preyNextState)
            preySize = self.entitiesSizeList[preyID]

            preyReward -= self.punishForOutOfBound(preyNextPos)
            for predatorID in self.predatorsID:
                predatorSize = self.entitiesSizeList[predatorID]
                predatorNextState = nextState[predatorID]
                if self.isCollision(predatorNextState, preyNextState, predatorSize, preySize):
                    preyReward -= self.collisionPunishment
            reward.append(preyReward)
        return reward