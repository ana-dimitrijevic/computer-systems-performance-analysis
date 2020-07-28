import numpy
import scipy.linalg as splin
import sys
import time
from random import randrange
from random import uniform
from math import pow
from math import log
import matplotlib.pylab as pl
import xlwt
from xlwt import Workbook

# VARIABLES

wb = {}
sheet = {}
currentRow = [1,1,1]

hours = 40
resources = {}

simulationTime = 1000 * 60 * 60 * hours
currentTime = 0
startTimestamp = 0

_K = range(2, 9)
_n = [10, 15, 20]

U = {}
R = {}
RSystem = {}


def exp(x):
    return -log(1 - uniform(0,1)) * x


def buzen_optimized(x_vect, n):
    """ Takes a dim x 1 vector of normalized time demands and level of multiprogramming and returns a Gordon-Newell vector.
        O(dim(x_vect) + n) space, O(dim(x_vect), n) time

        Parameters:
        x_vect (np.mat): vector of normalized time demands
        n (int): level of multiprogramming

        Returns:
        G (list): list of n+1 Gordon-Newell constants for the system
    """

    G = [0] * (n + 1)
    G[0] = 1
    x_list = x_vect.T[0]
    for xi in (x_list):
        for j in range(1, n + 1):
            G[j] = G[j] + G[j - 1] * xi
    return G


def gordon_newell(K, n):

    s = [5, 20, 15, 15]  # in ms
    for i in range(1, K + 1):
        s.insert(len(s), 20)

    u = [1000 / si for si in s]  # in jobs/s
    dim = len(s)

    P0 = [0.1, 0.1, 0.1, 0.1]
    for i in range(1, K + 1):
        P0.insert(len(P0), 0.6 / K)

    P1 = [0.4, 0, 0, 0]
    P2 = [0.4, 0, 0, 0]
    P3 = [0.4, 0, 0, 0]

    for i in range(1, K + 1):
        P1.insert(len(P1), 0.6 / K)
        P2.insert(len(P2), 0.6 / K)
        P3.insert(len(P3), 0.6 / K)

    P = [P0, P1, P2, P3]  # probability transition matrix, Pij is the probability of transition from server i+1 to server j+1

    Pi = [1, 0, 0, 0]

    for i in range(1, K + 1):
        Pi.insert(len(Pi), 0)

    for i in range(1, K + 1):
        P.insert(len(P), Pi)

    P = numpy.mat(P)
    # print("P:", "\n", P, "\n")

    A = numpy.mat(P).T - numpy.identity(dim)  # A = P.T-I
    # print("A = P.T - I :", "\n", A)

    U = [[0] * i + [ui] + [0] * (dim - i - 1) for i, ui in
         enumerate(u)]  # diagonal matrix with u's as values on the main diagonal
    U = numpy.mat(U)
    # print("U:\n", U, "\n")


    # Now we need to solve K * X = 0 for X.
    # A * x = b -> solve(A,b)
    # Can't use X = np.linalg.solve(K, [0]*dim), because matrix K is singular.
    # We can fix this by normalizing or just find null space of K instead!
    X = splin.null_space(A * U)
    X = X / X[0]  # normalization
    # print(X)
    if (n==10):
        outputGordonNewellResults(K, X)

    # run for N = 3:
    G = buzen_optimized(X, n)
    # print(G)

    x = X.T[0]
    usages = [xi * G[n - 1] / G[n] for xi in x]
    productivities = [usagei * ui for usagei, ui in zip(usages, u)]  # Xi = Ui / si = Ui * ui

    J = []

    for i in range(K + 4):
        s = 0
        for j in range(1,n):
            s +=  pow(x[i], j) * G[n - j] / G[n]
        J.insert(len(J), s)

    Xsystem = 0.1 * productivities[0]
    R = n / Xsystem

    outputBuzenResults(K, n, usages, productivities, J, R)


def start(n):
    global resources
    global startTime

    resources[0]['queue'] = n - 1
    resources[0]['finishTime'] = exp(resources[0]['S'])


def initialiseSimulationParameters(K, n):

    global resources
    global currentTime

    currentTime = 0

    resources.clear()

    CPU = {
        'name': 'CPU',
        'S': 5,
        'queue': 0,
        'finishTime': -1,
        'U': 0.0,
        'J': 0.0,
        'R': 0.0,
        'X': 0.0
    }

    SysDisc1 = {
        'name': 'SysDisc 1',
        'S': 20,
        'queue': 0,
        'finishTime': -1,
        'U': 0.0,
        'J': 0.0,
        'R': 0.0,
        'X': 0.0
    }

    SysDisc2 = {
        'name': 'SysDisc 2',
        'S': 15,
        'queue': 0,
        'finishTime': -1,
        'U': 0.0,
        'J': 0.0,
        'R': 0.0,
        'X': 0.0
    }

    SysDisc3 = {
        'name': 'SysDisc 3',
        'S': 15,
        'queue': 0,
        'finishTime': -1,
        'U': 0.0,
        'J': 0.0,
        'R': 0.0,
        'X': 0.0
    }


    resources[0] = CPU
    resources[1] = SysDisc1
    resources[2] = SysDisc2
    resources[3] = SysDisc3



    for i in range(K):
        UserDisc = {
            'name': 'UserDisc ',
            'S': 20,
            'queue': 0,
            'finishTime': -1,
            'U': 0.0,
            'J': 0.0,
            'R': 0.0,
            'X': 0.0
        }
        resources[i + 4] = UserDisc
        resources[i + 4]['name'] += str(i+1)


def getFirstToFinish():

    firstToFinish = 0
    minTime = sys.maxsize

    for i in range(len(resources)):
        if resources[i]['finishTime'] >= 0 and resources[i]['finishTime'] < minTime:
            minTime = resources[i]['finishTime']
            firstToFinish = i

    return firstToFinish


def loadNewJob(i):
    global resources

    if resources[i]['finishTime'] == -1 and resources[i]['queue'] > 0:
        resources[i]['finishTime'] = currentTime + exp(resources[i]['S'])
        resources[i]['queue'] -= 1

def pickCPUNext(totalResources):
    probability = uniform(0, 1)

    if probability < 0.1:
        return 0
    if probability < 0.2:
        return 1
    if probability < 0.3:
        return 2
    if probability < 0.4:
        return 3
    else:
        return randrange(4, totalResources)

def pickSysDiscNext(totalResources):
    probability = uniform(0, 1)
    if probability < 0.4:
        return 0
    else:
        return randrange(4, totalResources)

def pickUserDiscNext():
    return 0

def pickNextResource(currentResource, totalResources):

    if currentResource == 0:
        return pickCPUNext(totalResources)

    if currentResource == 1 or currentResource == 2 or currentResource == 3:
        return pickSysDiscNext(totalResources)

    return pickUserDiscNext()

def transferFinishedJob(currentResource, totalResources):

    global resources

    nextResource = pickNextResource(currentResource, len(resources))
    resources[nextResource]['queue'] += 1

    return nextResource

def simulation(K, n):

    global resources
    global currentTime

    initialiseSimulationParameters(K, n)
    start(n)

    startTimestamp = time.time()
    while currentTime < simulationTime:

        firstToFinish = getFirstToFinish()
        currentResource = resources[firstToFinish]

        timeDiff = currentResource['finishTime'] - currentTime
        currentTime = currentResource['finishTime']

        for i in range(len(resources)):
            if resources[i]['finishTime'] >= 0:
                resources[i]['U'] += timeDiff
                resources[i]['J'] += (resources[i]['queue'] + 1) * timeDiff

        currentResource['finishTime'] = -1

        nextResource = transferFinishedJob(firstToFinish, len(resources))

        loadNewJob(firstToFinish)
        loadNewJob(nextResource)


    duration = time.time() - startTimestamp
    print("Completed! Time elapsed : " + str(duration))

    for i in range(len(resources)):
        resources[i]["U"] /=  currentTime
        resources[i]["J"] /= currentTime
        resources[i]["X"] = resources[i]["U"] / resources[i]["S"]
        resources[i]["R"] = resources[i]["J"] / resources[i]["X"]


    Xsystem = 0.1 * resources[0]['U'] * 1 / resources[0]['S']
    R = n / (Xsystem*1000)

    outputSimulationResults(K,n, resources, R)


def outputGordonNewellResults(K, X):

    global currentRow

    for i in range(len(X)):
       addRowForX(sheet[2],resources[i]["name"],str(X[i][0]), currentRow[2])
       currentRow[2] += 1


    sheet[2].write(currentRow[2], 0, "K = " + str(K))
    currentRow[2] += 2

def outputBuzenResults(K, n, _U, flow, J, _R):

    global currentRow
    global U
    global R
    global RSystem

    for i in range(len(_U)):
        if (resources[i]["name"] == "CPU"
                or resources[i]["name"] == "SysDisc 1"
                or resources[i]["name"] == "SysDisc 2"
                or resources[i]["name"] == "UserDisc 1"
        ):
            addRowForResource(sheet[1], resources[i]["name"], str(_U[i] * 100), str(flow[i]/1000), str(J[i]), currentRow[1])
            U.get(n)[i].insert(len(U.get(n)[i]), _U[i] * 100)
            R.get(n)[i].insert(len(R.get(n)[i]), (J[i])/(flow[i]))
            currentRow[1]+=1

    currentRow[1] += 1

    sheet[1].write(currentRow[1], 0, "(K,n) = (" + str(K) + "," + str(n) + ")")
    sheet[1].write(currentRow[1], 1, "R = " + str(_R))

    RSystem.get(n).insert(len(RSystem.get(n)), _R)

    currentRow[1] += 2


def outputSimulationResults(K,n, resources, _R):
    global currentRow

    for i in range(len(resources)):
        if (resources[i]["name"]=="CPU"
                or resources[i]["name"]=="SysDisc 1"
                or resources[i]["name"]=="SysDisc 2"
                or resources[i]["name"] == "UserDisc 1"
            ):
            addRowForResource(sheet[0], resources[i]["name"], str(resources[i]["U"] * 100), str(resources[i]["X"]), str(resources[i]["J"]), currentRow[0])
            currentRow[0] += 1


    currentRow[0] += 1

    sheet[0].write(currentRow[0], 0, "(K,n) = (" + str(K) + "," + str(n) + ")")
    sheet[0].write(currentRow[0], 1, "R = " + str(_R))

    currentRow[0] += 2


def addRowForResource(sheet, name, U, X, J, currentRow):

    sheet.write(currentRow, 0, name)
    sheet.write(currentRow, 1, U)
    sheet.write(currentRow, 2, X)
    sheet.write(currentRow, 3, J)



def addRowForX(sheet, name, x, currentRow):

    sheet.write(currentRow, 0, name)
    sheet.write(currentRow, 1, x)

def getPlotColor(name):
    if(name=="CPU") :
        return '#C7980A'
    if (name == "SysDisc 1"):
        return '#ACD338'
    if(name=="SysDisc 2") :
        return '#82D8A7'
    if (name == "UserDisc 1"):
        return '#575E76'


def prepareFor2DPlottingMultiple(y, n, dependency):

    fig = pl.figure(dependency + "(K) for n=" + str(n))
    ax = fig.add_subplot(111)

    for i,_y in enumerate(y):
        if (resources.get(i)["name"]=="CPU" or resources.get(i)["name"]=="SysDisc 1"  or resources.get(i)["name"]=="SysDisc 2" or resources.get(i)["name"]=="UserDisc 1"):
            x = numpy.linspace(2, 8, len(_y))
            plot2D(x,_y, ax, dependency, getPlotColor(resources.get(i)["name"]), resources.get(i)["name"])



    pl.legend(title="Resource", fontsize='small', fancybox=True)


def prepareFor2DPlotting(y, n, dependency):

    x = numpy.linspace(2, 8, len(y))
    fig = pl.figure(dependency + "(K) for n=" + str(n))
    ax = fig.add_subplot(111)

    plot2D(x,y, ax, dependency, 'b', "System response time")


def plot2D(x,y, ax, dependency, color, label):

    ax.plot(x, y, color=color, label=label)

    if (dependency=="RSystem"):
        ax.set_xlabel('K')
        ax.set_ylabel('System response time [s]')
    if (dependency == "U"):
        ax.set_xlabel('K')
        ax.set_ylabel('U [%]')
    if (dependency == "R"):
        ax.set_xlabel('K')
        ax.set_ylabel('R [s]')


def plotResults():

    for key in U:
        prepareFor2DPlottingMultiple(U.get(key), key, "U")

    for key in R:
        prepareFor2DPlottingMultiple(R.get(key), key, "R")

    for key in RSystem:
        prepareFor2DPlotting(RSystem.get(key), key, "RSystem")

    pl.show()

def prepareXLS(title):

    global currentRow

    wb = Workbook()
    sheet = wb.add_sheet(title)

    style = xlwt.easyxf('font: bold 1;')
    sheet.write(0, 0, 'Resource name', style)

    if (title == "Normalizing constants"):
        sheet.write(0, 1, 'x', style)
    else:
        sheet.write(0, 1, 'U', style)
        sheet.write(0, 2, 'X', style)
        sheet.write(0, 3, 'J', style)

    return sheet, wb

def prepareForPlotting():

    global U
    global R
    global RSystem

    for i in _n:
        U[i] = []
        R[i] = []
        RSystem[i] = []

    for i in _n:
        for j in range(12):
            U[i].insert(len(U[i]), [])
            R[i].insert(len(R[i]), [])


prepareForPlotting()
sheet[0], wb[0] = prepareXLS('Results simulation')

for K in _K:
    for n in _n:
        print('K: %d, n: %d' % (K, n))
        simulation(K, n)

wb[0].save("results_simulation.xls")

sheet[1], wb[1] = prepareXLS('Results analytical')
sheet[2], wb[2] = prepareXLS('Normalizing constants')
for K in _K:
    for n in _n:
        gordon_newell(K, n)
wb[1].save("results_analytical.xls")
wb[2].save("normalizing_constants.xls")

plotResults()
