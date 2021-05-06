import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
dirName = os.path.dirname(__file__)
sys.path.append(os.path.join(dirName, '..'))
sys.path.append(os.path.join(dirName, '..', '..'))
import logging
import argparse
logging.getLogger('tensorflow').setLevel(logging.ERROR)

from src.environment.multiAgentEnv import *
from src.functionTools.loadSaveModel import *
from src.functionTools.trajectory import SampleTrajectory
from src.visualize.drawDemo import *
from src.environment.reward import *
from pygame.color import THECOLORS
from src.maddpg.trainer.MADDPG import *

maxEpisode = 60000
maxRunningStepsToSample = 75  # num of timesteps in one eps


class CalcPredatorsTrajKills:
    def __init__(self, predatorsID, killReward):
        self.predatorsID = predatorsID
        self.killReward = killReward
        self.rewardIDinTraj = 2
        
    def __call__(self, traj):        
        getPredatorReward = lambda allAgentsReward: np.sum([allAgentsReward[predatorID] for predatorID in self.predatorsID])
        rewardList = [getPredatorReward(timeStepInfo[self.rewardIDinTraj]) for timeStepInfo in traj]
        trajReward = np.sum(rewardList)
        trajKills = trajReward/self.killReward
        return trajKills


def parse_args():
    parser = argparse.ArgumentParser("Multi-agent chasing experiment evaluation")
    parser.add_argument("--num-predators", type=int, default=3, help="number of predators")
    parser.add_argument("--speed", type=float, default=1.0, help="prey speed multiplier")
    parser.add_argument("--cost", type=float, default=0.0, help="cost-action ratio")
    parser.add_argument("--selfish", type=float, default=0.0, help="selfish index")

    parser.add_argument("--num-traj", type=int, default=10, help="number of trajectories to sample")
    parser.add_argument("--visualize", type=int, default=1, help="generate demo = 1, otherwise 0")
    parser.add_argument("--save-images", type=int, default=1, help="save demo images = 1, otherwise 0")
    return parser.parse_args()


def main():
    arglist = parse_args()
    numPredators = arglist.num_predators
    preySpeedMultiplier = arglist.speed
    costActionRatio = arglist.cost
    selfishIndex = arglist.selfish
    numTrajToSample = arglist.num_traj
    visualize = arglist.visualize
    saveImage = arglist.save_images

    numPrey = 1
    numBlocks = 2
    maxTimeStep = 75
    killReward = 10
    killProportion = 0.2
    biteReward = 0.0

    print("evaluate: {} predators, {} prey, {} blocks, {} episodes with {} steps each eps, preySpeed: {}x, cost: {}, selfish: {}, sample {} trajectories, demo: {}, save: {}".
          format(numPredators, numPrey, numBlocks, maxEpisode, maxTimeStep, preySpeedMultiplier, costActionRatio, selfishIndex, numTrajToSample, visualize, saveImage))

    numAgents = numPredators + numPrey
    numEntities = numAgents + numBlocks
    predatorsID = list(range(numPredators))
    preyGroupID = list(range(numPredators, numAgents))
    blocksID = list(range(numAgents, numEntities))

    predatorSize = 0.075
    preySize = 0.05
    blockSize = 0.2
    entitiesSizeList = [predatorSize] * numPredators + [preySize] * numPrey + [blockSize] * numBlocks

    predatorMaxSpeed = 1.0
    blockMaxSpeed = None
    preyMaxSpeedOriginal = 1.3
    preyMaxSpeed = preyMaxSpeedOriginal * preySpeedMultiplier

    entityMaxSpeedList = [predatorMaxSpeed] * numPredators + [preyMaxSpeed] * numPrey + [blockMaxSpeed] * numBlocks
    entitiesMovableList = [True] * numAgents + [False] * numBlocks
    massList = [1.0] * numEntities

    collisionReward = 10 # originalPaper = 10*3
    isCollision = IsCollision(getPosFromAgentState)
    punishForOutOfBound = PunishForOutOfBound()
    rewardPrey = RewardPrey(predatorsID, preyGroupID, entitiesSizeList, getPosFromAgentState, isCollision,
                              punishForOutOfBound, collisionPunishment = collisionReward)

    collisionDist = predatorSize + preySize
    getAgentsPercentageOfRewards = GetAgentsPercentageOfRewards(selfishIndex, collisionDist)
    terminalCheck = TerminalCheck()

    getCollisionPredatorReward = GetCollisionPredatorReward(biteReward, killReward, killProportion, sampleFromDistribution, terminalCheck)
    getPredatorPreyDistance = GetPredatorPreyDistance(computeVectorNorm, getPosFromAgentState)
    rewardPredator = RewardPredatorsWithKillProb(predatorsID, preyGroupID, entitiesSizeList, isCollision, terminalCheck, getPredatorPreyDistance,
                 getAgentsPercentageOfRewards, getCollisionPredatorReward)

    reshapeAction = ReshapeAction()
    getActionCost = GetActionCost(costActionRatio, reshapeAction, individualCost=True)
    getPredatorsAction = lambda action: [action[predatorID] for predatorID in predatorsID]
    rewardPredatorWithActionCost = lambda state, action, nextState: np.array(rewardPredator(state, action, nextState)) - np.array(getActionCost(getPredatorsAction(action)))

    rewardFunc = lambda state, action, nextState: \
        list(rewardPredatorWithActionCost(state, action, nextState)) + list(rewardPrey(state, action, nextState))

    reset = ResetMultiAgentChasing(numAgents, numBlocks)
    observeOneAgent = lambda agentID: Observe(agentID, predatorsID, preyGroupID, blocksID, getPosFromAgentState,
                                              getVelFromAgentState)
    observe = lambda state: [observeOneAgent(agentID)(state) for agentID in range(numAgents)]

    getCollisionForce = GetCollisionForce()
    applyActionForce = ApplyActionForce(predatorsID, preyGroupID, entitiesMovableList)
    applyEnvironForce = ApplyEnvironForce(numEntities, entitiesMovableList, entitiesSizeList,
                                          getCollisionForce, getPosFromAgentState)
    integrateState = IntegrateState(numEntities, entitiesMovableList, massList,
                                    entityMaxSpeedList, getVelFromAgentState, getPosFromAgentState)
    transit = TransitMultiAgentChasing(numEntities, reshapeAction, applyActionForce, applyEnvironForce, integrateState)

    isTerminal = lambda state: terminalCheck.terminal
    initObsForParams = observe(reset())
    obsShape = [initObsForParams[obsID].shape[0] for obsID in range(len(initObsForParams))]

    sampleTrajectory = SampleTrajectory(maxRunningStepsToSample, transit, isTerminal, rewardFunc, reset)

    worldDim = 2
    actionDim = worldDim * 2 + 1

    layerWidth = [128, 128]

    #  model ------------------------

    buildMADDPGModels = BuildMADDPGModels(actionDim, numAgents, obsShape)
    modelsList = [buildMADDPGModels(layerWidth, agentID) for agentID in range(numAgents)]

    dirName = os.path.dirname(__file__)
    fileName = "model{}predators{}prey{}blocks{}episodes{}stepPreySpeed{}PredatorActCost{}sensitive{}biteReward{}killPercent{}_agent".format(
        numPredators, numPrey, numBlocks, maxEpisode, maxTimeStep, preySpeedMultiplier, costActionRatio, selfishIndex, biteReward, killProportion)
    modelPaths = [os.path.join(dirName, '..', 'trainedModels', fileName + str(i) ) for i in range(numAgents)]
    [restoreVariables(model, path) for model, path in zip(modelsList, modelPaths)]

    actOneStepOneModel = ActOneStep(actByPolicyTrainNoisy)
    policy = lambda allAgentsStates: [actOneStepOneModel(model, observe(allAgentsStates)) for model in modelsList]

    # generate trajectories ------------

    numKillsList = []
    trajToRender = []
    trajList = []
    calcPredatorsTrajKills = CalcPredatorsTrajKills(predatorsID, killReward)
    for i in range(numTrajToSample):
        traj = sampleTrajectory(policy)
        numKills = calcPredatorsTrajKills(traj)
        numKillsList.append(numKills)
        trajToRender = trajToRender + list(traj)
        trajList.append(traj)

    meanTrajKill = np.mean(numKillsList)
    seTrajKill = np.std(numKillsList) / np.sqrt(len(numKillsList) - 1)
    print('meanTrajKill', meanTrajKill, 'se ', seTrajKill)

    # save trajectories ------------

    trajectoryDirectory = os.path.join(dirName, '..', 'trajectories')
    if not os.path.exists(trajectoryDirectory):
        os.makedirs(trajectoryDirectory)

    trajFileName = "model{}predators{}prey{}blocks{}episodes{}stepPreySpeed{}PredatorActCost{}sensitive{}biteReward{}killPercent{}_Traj".format(
        numPredators, numPrey, numBlocks, maxEpisode, maxTimeStep, preySpeedMultiplier, costActionRatio,
        selfishIndex, biteReward, killProportion)

    trajSavePath = os.path.join(trajectoryDirectory, trajFileName)
    saveToPickle(trajList, trajSavePath)

    # visualize ------------

    if visualize:
        trajList = loadFromPickle(trajSavePath)

        screenWidth = 700
        screenHeight = 700
        screen = pg.display.set_mode((screenWidth, screenHeight))
        screenColor = THECOLORS['black']
        xBoundary = [0, 700]
        yBoundary = [0, 700]
        lineColor = THECOLORS['white']
        lineWidth = 4
        drawBackground = DrawBackground(screen, screenColor, xBoundary, yBoundary, lineColor, lineWidth)

        FPS = 10
        numBlocks = 2
        predatorColor = [255, 255, 255]
        preyColor = [0, 250, 0]
        blockColor = [200, 200, 200]
        circleColorSpace = [predatorColor] * numPredators + [preyColor] * numPrey + [blockColor] * numBlocks
        viewRatio = 1.5
        preySize = int(0.05 * screenWidth / (2 * viewRatio))
        predatorSize = int(0.075 * screenWidth / (3 * viewRatio))
        blockSize = int(0.2 * screenWidth / (3 * viewRatio))
        circleSizeSpace = [predatorSize] * numPredators + [preySize] * numPrey + [blockSize] * numBlocks
        positionIndex = [0, 1]
        agentIdsToDraw = list(range(numPredators + numPrey + numBlocks))

        conditionName = "model{}predators{}prey{}blocks{}episodes{}stepPreySpeed{}PredatorActCost{}sensitive{}biteReward{}killPercent{}".format(
            numPredators, numPrey, numBlocks, maxEpisode, maxTimeStep, preySpeedMultiplier, costActionRatio,
            selfishIndex, biteReward, killProportion)
        imageSavePath = os.path.join(dirName, '..', 'trajectories', conditionName)
        if not os.path.exists(imageSavePath):
            os.makedirs(imageSavePath)
        imageFolderName = str('forDemo')
        saveImageDir = os.path.join(os.path.join(imageSavePath, imageFolderName))
        if not os.path.exists(saveImageDir):
            os.makedirs(saveImageDir)
        imaginedWeIdsForInferenceSubject = list(range(numPredators))

        updateColorSpaceByPosterior = None
        outsideCircleAgentIds = imaginedWeIdsForInferenceSubject
        outsideCircleColor = np.array([[255, 0, 0]] * numPredators)
        outsideCircleSize = int(predatorSize * 1.5)
        drawCircleOutside = DrawCircleOutsideEnvMADDPG(screen, viewRatio, outsideCircleAgentIds, positionIndex,
                                                       outsideCircleColor, outsideCircleSize)
        drawState = DrawStateEnvMADDPG(FPS, screen, viewRatio, circleColorSpace, circleSizeSpace, agentIdsToDraw,
                                       positionIndex, saveImage, saveImageDir, preyGroupID, predatorsID,
                                       drawBackground, sensitiveZoneSize=None, updateColorByPosterior=updateColorSpaceByPosterior, drawCircleOutside=drawCircleOutside)

        # MDP Env
        interpolateState = None
        stateIndexInTimeStep = 0
        actionIndexInTimeStep = 1
        posteriorIndexInTimeStep = None

        stateID = 0
        nextStateID = 3
        predatorSizeForCheck = 0.075
        preySizeForCheck = 0.05
        checkStatus = CheckStatus(predatorsID, preyGroupID, isCollision, predatorSizeForCheck, preySizeForCheck, stateID, nextStateID)
        chaseTrial = ChaseTrialWithTrajWithKillNotation(stateIndexInTimeStep, drawState, checkStatus, interpolateState, actionIndexInTimeStep, posteriorIndexInTimeStep)
        [chaseTrial(trajectory) for trajectory in np.array(trajList[:20])]


if __name__ == '__main__':
    main()