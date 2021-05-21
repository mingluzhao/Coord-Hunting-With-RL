import pygame as pg
import numpy as np
import os


class DrawBackground:
    def __init__(self, screen, screenColor, xBoundary, yBoundary, lineColor, lineWidth, xObstacles = None, yObstacles = None):
        self.screen = screen
        self.screenColor = screenColor
        self.xBoundary = xBoundary
        self.yBoundary = yBoundary
        self.lineColor = lineColor
        self.lineWidth = lineWidth
        self.xObstacles = xObstacles
        self.yObstacles = yObstacles

    def __call__(self):
        for event in pg.event.get():
            if event.type == pg.QUIT:
                pg.quit()
            if event.type == pg.KEYDOWN:
                if event.key == pg.K_ESCAPE:
                    exit()
        self.screen.fill(self.screenColor)
        rectPos = [self.xBoundary[0], self.yBoundary[0], self.xBoundary[1], self.yBoundary[1]]
        pg.draw.rect(self.screen, self.lineColor, rectPos, self.lineWidth)
        if self.xObstacles and self.yObstacles:
            for xObstacle, yObstacle in zip(self.xObstacles, self.yObstacles):
                rectPos = [xObstacle[0], yObstacle[0], xObstacle[1] - xObstacle[0], yObstacle[1] - yObstacle[0]]
                pg.draw.rect(self.screen, self.lineColor, rectPos)
        return


class DrawCircleOutside:
    def __init__(self, screen, outsideCircleAgentIds, positionIndex, circleColors, circleSize, viewRatio = 1):
        self.screen = screen
        self.viewRatio = viewRatio
        self.screenX, self.screenY = self.screen.get_width(), self.screen.get_height()
        self.outsideCircleAgentIds = outsideCircleAgentIds
        self.xIndex, self.yIndex = positionIndex
        self.circleColors = circleColors
        self.circleSize = circleSize

    def __call__(self, state):
        for agentIndex in self.outsideCircleAgentIds:
            agentPos = [np.int((state[agentIndex][self.xIndex] / self.viewRatio + 1) * (self.screenX / 2)), 
                    np.int((state[agentIndex][self.yIndex] / self.viewRatio + 1) * (self.screenY / 2))]
            agentColor = tuple(self.circleColors[list(self.outsideCircleAgentIds).index(agentIndex)])
            pg.draw.circle(self.screen, agentColor, agentPos, self.circleSize)
        return



class DrawState:
    def __init__(self, fps, screen, colorSpace, circleSizeSpace, agentIdsToDraw, positionIndex, saveImage,
                 imagePath, preyGroupID, predatorsID, drawBackGround, drawCircleOutside = None, viewRatio = 1):
        self.fps = fps
        self.screen = screen
        self.viewRatio = viewRatio
        self.screenX, self.screenY = self.screen.get_width(), self.screen.get_height()
        self.colorSpace = colorSpace
        self.circleSizeSpace = circleSizeSpace
        self.agentIdsToDraw = agentIdsToDraw
        self.xIndex, self.yIndex = positionIndex
        self.saveImage = saveImage
        self.imagePath = imagePath
        self.drawBackGround = drawBackGround
        self.drawCircleOutside = drawCircleOutside
        self.preyGroupID = preyGroupID
        self.predatorsID = predatorsID

        self.biteCount = 0
        self.killCount = 0

    def __call__(self, state, agentsStatus, posterior = None):
        fpsClock = pg.time.Clock()

        self.drawBackGround()
        circleColors = self.colorSpace
        if self.drawCircleOutside:
            self.drawCircleOutside(state)

        for agentIndex in self.agentIdsToDraw:
            agentPos = [np.int((state[agentIndex][self.xIndex] / self.viewRatio + 1) * (self.screenX / 2)), 
                    np.int((state[agentIndex][self.yIndex] / self.viewRatio + 1) * (self.screenY / 2))]
            agentColor = tuple(circleColors[agentIndex])
            circleSize = self.circleSizeSpace[agentIndex]

            if agentIndex in self.preyGroupID:
                agentStatus = agentsStatus[agentIndex]

                if agentStatus == 'kill' or self.killCount != 0:
                    killPreyColor = [0, 120, 0]
                    pg.draw.circle(self.screen, killPreyColor, agentPos, circleSize)
                    self.killCount += 1
                    if self.killCount == 2:
                        self.killCount = 0

                elif agentStatus == 'bite' or self.biteCount != 0:
                    bitePreyColor = [200, 255, 200]
                    pg.draw.circle(self.screen, bitePreyColor, agentPos, circleSize)
                    self.biteCount += 1
                    if self.biteCount == 2:
                        self.biteCount = 0
                else:
                    pg.draw.circle(self.screen, agentColor, agentPos, circleSize)

            elif agentIndex in self.predatorsID:
                agentStatus = agentsStatus[agentIndex]
                agentColorToDraw = [100, 0, 0] if agentStatus == 'bite' else agentColor
                pg.draw.circle(self.screen, agentColorToDraw, agentPos, circleSize)

            else:
                pg.draw.circle(self.screen, agentColor, agentPos, circleSize)

        pg.display.flip()
        
        if self.saveImage == True:
            filenameList = os.listdir(self.imagePath)
            pg.image.save(self.screen, self.imagePath + '/' + str(len(filenameList))+'.png')
        
        fpsClock.tick(self.fps)
        return self.screen


class ChaseTrialWithKillNotation:
    def __init__(self, stateIndex, drawState, checkStatus):
        self.stateIndex = stateIndex
        self.drawState = drawState
        self.checkStatus = checkStatus

    def __call__(self, trajectory):
        for timeStepIndex in range(len(trajectory)):
            timeStep = trajectory[timeStepIndex]
            nextTimeStep = trajectory[timeStepIndex+1] if timeStepIndex != len(trajectory)-1 else None
            agentsStatus = self.checkStatus(timeStep, nextTimeStep)

            state = timeStep[self.stateIndex]
            posterior = None
            screen = self.drawState(state, agentsStatus, posterior)
        return


class CheckStatus:
    def __init__(self, predatorsID, preyGroupID, isCollision, predatorSize, preySize, stateID, nextStateID):
        self.predatorsID = predatorsID
        self.preyGroupID = preyGroupID
        self.isCollision = isCollision
        self.predatorSize= predatorSize
        self.preySize = preySize
        self.stateID = stateID
        self.nextStateID = nextStateID

    def __call__(self, timeStep, nextTimeStep):
        agentsStatus = [0] * (len(self.predatorsID) + len(self.preyGroupID))
        killed = np.any([tuple(a) != tuple(b) for a, b in zip(nextTimeStep[self.stateID], timeStep[self.nextStateID])]) if nextTimeStep is not None else False

        for predatorID in self.predatorsID:
            for preyID in self.preyGroupID:
                predatorNextState = timeStep[self.nextStateID][predatorID]
                preyNextState = timeStep[self.nextStateID][preyID]
                if self.isCollision(predatorNextState, preyNextState, self.predatorSize, self.preySize):
                    agentsStatus[predatorID] = 'bite'
                    agentsStatus[preyID] = 'kill' if killed else 'bite'

        return agentsStatus

