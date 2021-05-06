# Partially adapted from the multi-agent particle environment: https://github.com/openai/multiagent-particle-envs

import numpy as np

getPosFromAgentState = lambda state: np.array([state[0], state[1]])
getVelFromAgentState = lambda agentState: np.array([agentState[2], agentState[3]])


class GetActionCost:
    def __init__(self, costActionRatio, reshapeAction, individualCost):
        self.costActionRatio = costActionRatio
        self.individualCost = individualCost
        self.reshapeAction =reshapeAction

    def __call__(self, agentsActions):
        agentsActions = [self.reshapeAction(action) for action in agentsActions]
        actionMagnitude = [np.linalg.norm(np.array(action), ord=2) for action in agentsActions]
        cost = self.costActionRatio* np.array(actionMagnitude)
        numAgents = len(agentsActions)
        groupCost = cost if self.individualCost else [np.sum(cost)] * numAgents

        return groupCost


class IsCollision:
    def __init__(self, getPosFromState):
        self.getPosFromState = getPosFromState

    def __call__(self, agent1State, agent2State, agent1Size, agent2Size):
        posDiff = self.getPosFromState(agent1State) - self.getPosFromState(agent2State)
        dist = np.sqrt(np.sum(np.square(posDiff)))
        minDist = agent1Size + agent2Size
        return True if dist < minDist else False


class PunishForOutOfBound:
    def __init__(self):
        self.physicsDim = 2

    def __call__(self, agentPos):
        punishment = 0
        for i in range(self.physicsDim):
            x = abs(agentPos[i])
            punishment += self.bound(x)
        return punishment

    def bound(self, x):
        if x < 0.9:
            return 0
        if x < 1.0:
            return (x - 0.9) * 10
        return min(np.exp(2 * x - 2), 10)


class ResetMultiAgentChasing:
    def __init__(self, numTotalAgents, numBlocks):
        self.positionDimension = 2
        self.numTotalAgents = numTotalAgents
        self.numBlocks = numBlocks

    def __call__(self):
        getAgentRandomPos = lambda: np.random.uniform(-1, +1, self.positionDimension)
        getAgentRandomVel = lambda: np.zeros(self.positionDimension)
        agentsState = [list(getAgentRandomPos()) + list(getAgentRandomVel()) for ID in range(self.numTotalAgents)]

        getBlockRandomPos = lambda: np.random.uniform(-0.9, +0.9, self.positionDimension)
        getBlockSpeed = lambda: np.zeros(self.positionDimension)

        blocksState = [list(getBlockRandomPos()) + list(getBlockSpeed()) for blockID in range(self.numBlocks)]
        state = np.array(agentsState + blocksState)
        return state


class Observe:
    def __init__(self, agentID, predatorsID, preyGroupID, blocksID, getPosFromState, getVelFromAgentState):
        self.agentID = agentID
        self.predatorsID = predatorsID
        self.preyGroupID = preyGroupID
        self.blocksID = blocksID
        self.getEntityPos = lambda state, entityID: getPosFromState(state[entityID])
        self.getEntityVel = lambda state, entityID: getVelFromAgentState(state[entityID])

    def __call__(self, state):
        blocksPos = [self.getEntityPos(state, blockID) for blockID in self.blocksID]
        agentPos = self.getEntityPos(state, self.agentID)
        blocksInfo = [blockPos - agentPos for blockPos in blocksPos]

        posInfo = []
        for predatorID in self.predatorsID:
            if predatorID == self.agentID: continue
            predatorPos = self.getEntityPos(state, predatorID)
            posInfo.append(predatorPos - agentPos)

        velInfo = []
        for preyID in self.preyGroupID:
            if preyID == self.agentID: continue
            preyPos = self.getEntityPos(state, preyID)
            posInfo.append(preyPos - agentPos)
            preyVel = self.getEntityVel(state, preyID)
            velInfo.append(preyVel)

        agentVel = self.getEntityVel(state, self.agentID)
        return np.concatenate([agentVel] + [agentPos] + blocksInfo + posInfo + velInfo)


class GetCollisionForce:
    def __init__(self, contactMargin = 0.001, contactForce = 100):
        self.contactMargin = contactMargin
        self.contactForce = contactForce

    def __call__(self, obj1Pos, obj2Pos, obj1Size, obj2Size, obj1Movable, obj2Movable):
        posDiff = obj1Pos - obj2Pos
        dist = np.sqrt(np.sum(np.square(posDiff)))

        minDist = obj1Size + obj2Size
        penetration = np.logaddexp(0, -(dist - minDist) / self.contactMargin) * self.contactMargin

        force = self.contactForce* posDiff / dist * penetration
        force1 = +force if obj1Movable else None
        force2 = -force if obj2Movable else None

        return [force1, force2]


class ApplyActionForce:
    def __init__(self, predatorsID, preyGroupID, entitiesMovableList, actionDim=2):
        self.agentsID = preyGroupID + predatorsID
        self.numAgents = len(self.agentsID)
        self.entitiesMovableList = entitiesMovableList
        self.actionDim = actionDim

    def __call__(self, pForce, actions):
        noise = [None] * self.numAgents
        for agentID in self.agentsID:
            movable = self.entitiesMovableList[agentID]
            agentNoise = noise[agentID]
            if movable:
                agentNoise = np.random.randn(self.actionDim) * agentNoise if agentNoise else 0.0
                pForce[agentID] = np.array(actions[agentID]) + agentNoise
        return pForce


class ApplyEnvironForce:
    def __init__(self, numEntities, entitiesMovableList, entitiesSizeList, getCollisionForce, getPosFromState):
        self.numEntities = numEntities
        self.entitiesMovableList = entitiesMovableList
        self.entitiesSizeList = entitiesSizeList
        self.getCollisionForce = getCollisionForce
        self.getEntityPos = lambda state, entityID: getPosFromState(state[entityID])

    def __call__(self, pForce, state):
        for entity1ID in range(self.numEntities):
            for entity2ID in range(self.numEntities):
                if entity2ID <= entity1ID: continue
                obj1Movable = self.entitiesMovableList[entity1ID]
                obj2Movable = self.entitiesMovableList[entity2ID]
                obj1Size = self.entitiesSizeList[entity1ID]
                obj2Size = self.entitiesSizeList[entity2ID]
                obj1Pos = self.getEntityPos(state, entity1ID)
                obj2Pos = self.getEntityPos(state, entity2ID)

                force1, force2 = self.getCollisionForce(obj1Pos, obj2Pos, obj1Size, obj2Size, obj1Movable, obj2Movable)

                if force1 is not None:
                    if pForce[entity1ID] is None: pForce[entity1ID] = 0.0
                    pForce[entity1ID] = force1 + pForce[entity1ID]

                if force2 is not None:
                    if pForce[entity2ID] is None: pForce[entity2ID] = 0.0
                    pForce[entity2ID] = force2 + pForce[entity2ID]

        return pForce


class IntegrateState:
    def __init__(self, numEntities, entitiesMovableList, massList, entityMaxSpeedList, getVelFromAgentState, getPosFromAgentState, damping=0.25, dt=0.1):
        self.numEntities = numEntities
        self.entitiesMovableList = entitiesMovableList
        self.damping = damping
        self.dt = dt
        self.massList = massList
        self.entityMaxSpeedList = entityMaxSpeedList
        self.getEntityVel = lambda state, entityID: getVelFromAgentState(state[entityID])
        self.getEntityPos = lambda state, entityID: getPosFromAgentState(state[entityID])

    def __call__(self, pForce, state):
        getNextState = lambda entityPos, entityVel: list(entityPos) + list(entityVel)
        nextState = []
        for entityID in range(self.numEntities):
            entityMovable = self.entitiesMovableList[entityID]
            entityVel = self.getEntityVel(state, entityID)
            entityPos = self.getEntityPos(state, entityID)

            if not entityMovable:
                nextState.append(getNextState(entityPos, entityVel))
                continue

            entityNextVel = entityVel * (1 - self.damping)
            entityForce = pForce[entityID]
            entityMass = self.massList[entityID]
            if entityForce is not None:
                entityNextVel += (entityForce / entityMass) * self.dt

            entityMaxSpeed = self.entityMaxSpeedList[entityID]
            if entityMaxSpeed is not None:
                speed = np.sqrt(np.square(entityNextVel[0]) + np.square(entityNextVel[1])) #
                if speed > entityMaxSpeed:
                    entityNextVel = entityNextVel / speed * entityMaxSpeed

            entityNextPos = entityPos + entityNextVel * self.dt
            nextState.append(getNextState(entityNextPos, entityNextVel))

        return nextState


class TransitMultiAgentChasing:
    def __init__(self, numEntities, reshapeAction, applyActionForce, applyEnvironForce, integrateState):
        self.numEntities = numEntities
        self.reshapeAction = reshapeAction
        self.applyActionForce = applyActionForce
        self.applyEnvironForce = applyEnvironForce
        self.integrateState = integrateState

    def __call__(self, state, actions):
        actions = [self.reshapeAction(action) for action in actions]
        p_force = [None] * self.numEntities
        p_force = self.applyActionForce(p_force, actions)
        p_force = self.applyEnvironForce(p_force, state)
        nextState = self.integrateState(p_force, state)

        return nextState


class ReshapeAction:
    def __init__(self):
        self.actionDim = 2
        self.sensitivity = 5

    def __call__(self, action):
        actionX = action[1] - action[2]
        actionY = action[3] - action[4]
        actionReshaped = np.array([actionX, actionY]) * self.sensitivity
        return actionReshaped

