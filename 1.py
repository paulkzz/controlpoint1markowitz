# Загрузка необходимых библиотек и автоматического заполнения данных котировок акций из сервиса Yahoo Finance
import pandas as pd
import pandas_datareader as web
import numpy as np

# Выбор компаний для портфеля (Apple, Electronic Arts, Activision Blizzard, AMD) и выгрузка данных

aapleQuote = web.DataReader('AAPL', data_source='yahoo', start='2020-01-01', end='2020-11-30')
eaQuote = web.DataReader('EA', data_source='yahoo', start='2020-01-01', end='2020-11-30')
atviQuote = web.DataReader('ATVI', data_source='yahoo', start='2020-01-01', end='2020-11-30')
amdQuote = web.DataReader('AMD', data_source='yahoo', start='2020-01-01', end='2020-11-30')

# Оформление данных

stocks = ['AAPL', 'EA', 'ATVI', 'AMD']
numAssets = len(stocks)
source = 'yahoo'
start = '2020-01-01'
end = '2020-11-30'

data = pd.DataFrame(columns=stocks)
for symbol in stocks:
    data[symbol] = web.DataReader(symbol, data_source=source, start=start, end=end)['Adj Close']

# Подсчет логарифмов log returns

percent_change = data.pct_change()
returns = np.log(1 + percent_change)
returns.head()
print(percent_change)
print(returns)

# Построение ковариационной матрицы

meanDailyReturns = returns.mean()
covMatrix = returns.cov()
print(meanDailyReturns)
print(covMatrix)

# Формирование случайного портфеля

weights = np.array([0.5, 0.2, 0.2, 0.1])
portReturn = np.sum(meanDailyReturns * weights)
portStdDev = np.sqrt(np.dot(weights.T, np.dot(covMatrix, weights)))

print(portReturn * 365)
print(portStdDev * portStdDev * 365)

# Визулизация

import matplotlib.pyplot as plt

plt.figure(figsize=(14, 7))
for c in returns.columns.values:
    plt.plot(returns.index, returns[c], lw=3, alpha=0.8, label=c)

plt.legend(loc='upper left', fontsize=12)
plt.ylabel('returns')

# Оптимизация портфолио

import scipy.optimize as sco


def calcPortfolioPerf(weights, meanReturns, covMatrix):
    portReturn = np.sum(meanReturns * weights)
    portStdDev = np.sqrt(np.dot(weights.T, np.dot(covMatrix, weights)))
    return portReturn, portStdDev


# Коеффициент Шарпа

def negSharpeRatio(weights, meanReturns, covMatrix, riskFreeRate):
    p_ret, p_var = calcPortfolioPerf(weights, meanReturns, covMatrix)

    return -(p_ret - riskFreeRate) / p_var


def getPortfolioVol(weights, meanReturns, covMatrix):
    return calcPortfolioPerf(weights, meanReturns, covMatrix)[1]


def findMaxSharpeRatioPortfolio(meanReturns, covMatrix, riskFreeRate):
    numAssets = len(meanReturns)
    args = (meanReturns, covMatrix, riskFreeRate)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for asset in range(numAssets))

    opts = sco.minimize(negSharpeRatio, numAssets * [1. / numAssets, ], args=args, method='SLSQP', bounds=bounds,
                        constraints=constraints)

    return opts


# optimizing our portfolio based off of minimizing volatility

def findEfficientReturn(meanReturns, covMatrix, targetReturn):
    numAssets = len(meanReturns)
    args = (meanReturns, covMatrix)

    def getPortfolioReturn(weights):
        return calcPortfolioPerf(weights, meanReturns, covMatrix)[0]

    constraints = ({'type': 'eq', 'fun': lambda x: getPortfolioReturn(x) - targetReturn},
                   {'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for asset in range(numAssets))

    return sco.minimize(getPortfolioVol, numAssets * [1. / numAssets, ], args=args, method='SLSQP', bounds=bounds,
                        constraints=constraints)


def findEfficientFrontier(meanReturns, covMatrix, rangeOfReturns):
    efficientPortfolios = []
    for ret in rangeOfReturns:
        efficientPortfolios.append(findEfficientReturn(meanReturns, covMatrix, ret))

    return efficientPortfolios


print(findEfficientReturn)
