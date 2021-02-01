import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['KMP_DUPLICATE_LIB_OK']='True'
dirName = os.path.dirname(__file__)
sys.path.append(os.path.join(dirName, '..'))
sys.path.append(os.path.join(dirName, '..', '..'))

from functionTools.loadSaveModel import saveToPickle, restoreVariables, loadFromPickle
import itertools as it
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def modify(df):
    selfish = df.index.get_level_values('Selfish Index')[0]
    dist = df.index.get_level_values('Distance to kill')[0]
    if selfish  == 3:
        selfish  =  1e4
    percent =  (dist + 1 - 0.125) ** -selfish

    return pd.Series({'Percentage of Rewards (unnormalized)': percent})



def main():
    independentVariables = dict()
    independentVariables['Selfish Index'] = [0,  0.5, 1, 1.5, 2, 2.5, 3]
    independentVariables['Distance to kill'] = np.arange(0.125, 2, 0.01).tolist()

    levelNames = list(independentVariables.keys())
    levelValues = list(independentVariables.values())
    levelIndex = pd.MultiIndex.from_product(levelValues, names=levelNames)
    toSplitFrame = pd.DataFrame(index=levelIndex)
    resultDF = toSplitFrame.groupby(levelNames).apply(modify)
    print(resultDF)
    g = sns.lineplot(data=resultDF.reset_index(), x = 'Distance to kill', y = 'Percentage of Rewards (unnormalized)', hue = 'Selfish Index', legend=  False)
    plt.legend(title='Selfish Index', loc='lower left', labels=[0,  0.5, 1, 1.5, 2, 2.5, 10000])

    # plt.show()
    plt.savefig(os.path.join(os.path.join(dirName, '..', 'evalResults'), 'rewardPlot.png'), dpi=600)

    # sensitivity = 10000
    #
    # x = np.arange(0.125, 2, 0.01).tolist()
    # y = [(i + 1 - 0.125) ** -sensitivity for i in x]
    #
    # plt.ylim([0, 1.05])
    # plt.xlabel('distance')
    # plt.ylabel('percentage of rewards')
    # plt.suptitle('Selfish Index = {}'.format(sensitivity))
    #
    # plt.plot(x, y)
    #
    # labels = [0.125] + np.round(np.arange(0.4, 2, 0.2), 1).tolist()
    # plt.xticks(labels)
    # plt.show()




if __name__ == '__main__':
    main()