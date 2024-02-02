import numpy as np
import matplotlib.pyplot as plt
import matplotlib

plt.rc('text',usetex=True)
plt.rc('font',family='serif')
plt.rcParams['text.usetex'] = True
matplotlib.rcParams.update({'font.size':30})
plt.figure(figsize=(8,6))

# Used scipy for this

alpha = 1   # 0,1,2
d = 10       # 3,10,50
lam = 10    # 1,10,100

dMap = {3:0, 10:1, 50:2}
lamMap = {1:0, 10:1, 100:2}

offset = 4284 * alpha + 1428 * dMap[d]  + 476 * lamMap[lam]

# fig, ax = plt.subplots(4, 1, figsize=(8,20))

plotNumber = 4

conv = {
    4: lambda x: 0,
}
data = np.loadtxt('Data/dataPaper2.csv', delimiter=';', skiprows=1, converters=conv)

data = data[offset:]
xpoints = data[:,5]
ypoints = data[:,6] / 10000
stderror = data[:,7] / 10000

if plotNumber == 1:

    xpointsGlobal = xpoints[5:17]
    ypointsGlobal = ypoints[5:17]
    stdGlobal = stderror[5:17]
    plt.plot(xpointsGlobal,ypointsGlobal, label=f"SSP (global)")
    plt.errorbar(xpointsGlobal, ypointsGlobal, yerr = stdGlobal, fmt=None, color='black')

    xpointsOur = xpoints[17:34]
    ypointsOur = ypoints[17:34]
    stdOur = stderror[17:34]
    plt.plot(xpointsOur,ypointsOur, label=f"p=1 (LTM)")
    plt.errorbar(xpointsOur, ypointsOur, yerr = stdOur, fmt=None, color='black')

    xpoints09 = xpoints[34:51]
    ypoints09 = ypoints[34:51]
    std09 = stderror[34:51]
    plt.plot(xpoints09, ypoints09, label=f"p=0.9")
    plt.errorbar(xpoints09, ypoints09, yerr = std09, fmt=None, color='black')

    xpoints08 = xpoints[51:68]
    ypoints08 = ypoints[51:68]
    std08 = stderror[51:68]
    plt.plot(xpoints08, ypoints08, label=f"p=0.8")
    plt.errorbar(xpoints08, ypoints08, yerr = std08, fmt=None, color='black')

    xpoints07 = xpoints[68:85]
    ypoints07 = ypoints[68:85]
    std07 = stderror[68:85]
    plt.plot(xpoints07, ypoints07, label=f"p=0.7")
    plt.errorbar(xpoints07, ypoints07, yerr = std07, fmt=None, color='black')

    xpoints06 = xpoints[85:102]
    ypoints06 = ypoints[85:102]
    std06 = stderror[85:102]
    plt.plot(xpoints06, ypoints06, label=f"p=0.6")
    plt.errorbar(xpoints06, ypoints06, yerr = std06, fmt=None, color='black')

    xpoints05 = xpoints[102:119]
    ypoints05 = ypoints[102:119]
    std05 = stderror[102:119]
    plt.plot(xpoints05, ypoints05, label=f"p=0.5")
    plt.errorbar(xpoints05, ypoints05, yerr = std05, fmt=None, color='black')

    xpointsLocal = xpoints[119:136]
    ypointsLocal = ypoints[119:136]
    stdLocal = stderror[119:136]
    plt.plot(xpointsLocal,ypointsLocal, label=f"p=0 (local)")
    plt.errorbar(xpointsLocal, ypointsLocal, yerr = stdLocal, fmt=None, color='black')

    # plt.set_title("epsilon=0.01")

    plt.ylabel(r"error $\times 10^4$")
    plt.xlabel("n")#,labelpad=-10)
    # Put xlabel to the right
    plt.legend(loc='upper left', fontsize='20')

    plt.savefig('Figures/plotEpsilon001.pdf', bbox_inches='tight')
    # plt.show()

elif plotNumber == 2:

    data = data[136:]
    xpoints = data[:,5]
    ypoints = data[:,6] / 1000
    stderror = data[:,7] / 1000

    xpointsGlobal = xpoints[5:17]
    ypointsGlobal = ypoints[5:17]
    stdGlobal = stderror[5:17]
    plt.plot(xpointsGlobal,ypointsGlobal, label=f"SSP (global)")
    plt.errorbar(xpointsGlobal, ypointsGlobal, yerr = stdGlobal, fmt=None, color='black')

    xpointsOur = xpoints[17:34]
    ypointsOur = ypoints[17:34]
    stdOur = stderror[17:34]
    plt.plot(xpointsOur,ypointsOur, label=f"p=1 (LTM)")
    plt.errorbar(xpointsOur, ypointsOur, yerr = stdOur, fmt=None, color='black')

    xpoints09 = xpoints[34:51]
    ypoints09 = ypoints[34:51]
    std09 = stderror[34:51]
    plt.plot(xpoints09, ypoints09, label=f"p=0.9")
    plt.errorbar(xpoints09, ypoints09, yerr = std09, fmt=None, color='black')

    xpoints08 = xpoints[51:68]
    ypoints08 = ypoints[51:68]
    std08 = stderror[51:68]
    plt.plot(xpoints08, ypoints08, label=f"p=0.8")
    plt.errorbar(xpoints08, ypoints08, yerr = std08, fmt=None, color='black')

    xpoints07 = xpoints[68:85]
    ypoints07 = ypoints[68:85]
    std07 = stderror[68:85]
    plt.plot(xpoints07, ypoints07, label=f"p=0.7")
    plt.errorbar(xpoints07, ypoints07, yerr = std07, fmt=None, color='black')

    xpoints06 = xpoints[85:99] # 102
    ypoints06 = ypoints[85:99]
    std06 = stderror[85:99]
    plt.plot(xpoints06, ypoints06, label=f"p=0.6")
    plt.errorbar(xpoints06, ypoints06, yerr = std06, fmt=None, color='black')

    xpoints05 = xpoints[102:116] # 119
    ypoints05 = ypoints[102:116]
    std05 = stderror[102:116]
    plt.plot(xpoints05, ypoints05, label=f"p=0.5")
    plt.errorbar(xpoints05, ypoints05, yerr = std05, fmt=None, color='black')

    xpointsLocal = xpoints[119:133] # 136
    ypointsLocal = ypoints[119:133]
    stdLocal = stderror[119:133]
    plt.plot(xpointsLocal,ypointsLocal, label=f"p=0 (local)")
    plt.errorbar(xpointsLocal, ypointsLocal, yerr = stdLocal, fmt=None, color='black')

    # plt.set_title("epsilon=0.03")

    plt.ylabel(r"error $\times 10^3$")
    plt.xlabel("n")#,labelpad=-10)
    # Put xlabel to the right
    # plt.legend(loc='upper left')

    plt.savefig('Figures/plotEpsilon003.pdf', bbox_inches='tight')


elif plotNumber == 3:

    data = data[136:]
    data = data[136:]
    xpoints = data[:,5]
    ypoints = data[:,6] / 1000
    stderror = data[:,7] / 1000

    xpointsGlobal = xpoints[5:17]
    ypointsGlobal = ypoints[5:17]
    stdGlobal = stderror[5:17]
    plt.plot(xpointsGlobal,ypointsGlobal, label=f"SSP (global)")
    plt.errorbar(xpointsGlobal, ypointsGlobal, yerr = stdGlobal, fmt=None, color='black')

    xpointsOur = xpoints[17:34]
    ypointsOur = ypoints[17:34]
    stdOur = stderror[17:34]
    plt.plot(xpointsOur,ypointsOur, label=f"p=1 (LTM)")
    plt.errorbar(xpointsOur, ypointsOur, yerr = stdOur, fmt=None, color='black')

    xpoints09 = xpoints[34:51]
    ypoints09 = ypoints[34:51]
    std09 = stderror[34:51]
    plt.plot(xpoints09, ypoints09, label=f"p=0.9")
    plt.errorbar(xpoints09, ypoints09, yerr = std09, fmt=None, color='black')

    xpoints08 = xpoints[51:68]
    ypoints08 = ypoints[51:68]
    std08 = stderror[51:68]
    plt.plot(xpoints08, ypoints08, label=f"p=0.8")
    plt.errorbar(xpoints08, ypoints08, yerr = std08, fmt=None, color='black')

    xpoints07 = xpoints[68:85] #85
    ypoints07 = ypoints[68:85]
    std07 = stderror[68:85]
    plt.plot(xpoints07, ypoints07, label=f"p=0.7")
    plt.errorbar(xpoints07, ypoints07, yerr = std07, fmt=None, color='black')

    xpoints06 = xpoints[85:98] # 102
    ypoints06 = ypoints[85:98]
    std06 = stderror[85:98]
    plt.plot(xpoints06, ypoints06, label=f"p=0.6")
    plt.errorbar(xpoints06, ypoints06, yerr = std06, fmt=None, color='black')

    xpoints05 = xpoints[102:115] # 119
    ypoints05 = ypoints[102:115]
    std05 = stderror[102:115]
    plt.plot(xpoints05, ypoints05, label=f"p=0.5")
    plt.errorbar(xpoints05, ypoints05, yerr = std05, fmt=None, color='black')

    xpointsLocal = xpoints[119:132] # 136
    ypointsLocal = ypoints[119:132]
    stdLocal = stderror[119:132]
    plt.plot(xpointsLocal,ypointsLocal, label=f"p=0 (local)")
    plt.errorbar(xpointsLocal, ypointsLocal, yerr = stdLocal, fmt=None, color='black')

    # plt.set_title("epsilon=0.05")

    plt.ylabel(r"error $\times 10^3$")
    plt.xlabel("n")#,labelpad=-10)
    # Put xlabel to the right
    # plt.legend(loc='upper left')

    plt.savefig('Figures/plotEpsilon005.pdf', bbox_inches='tight')


elif plotNumber == 4:

    data = data[136:]
    data = data[136:]
    data = data[136:]
    xpoints = data[:,5]
    ypoints = data[:,6] / 100
    stderror = data[:,7] / 100

    xpointsGlobal = xpoints[5:17]
    ypointsGlobal = ypoints[5:17]
    stdGlobal = stderror[5:17]
    plt.plot(xpointsGlobal,ypointsGlobal, label=f"SSP (global)")
    plt.errorbar(xpointsGlobal, ypointsGlobal, yerr = stdGlobal, fmt=None, color='black')

    xpointsOur = xpoints[17:34]
    ypointsOur = ypoints[17:34]
    stdOur = stderror[17:34]
    plt.plot(xpointsOur,ypointsOur, label=f"p=1 (LTM)")
    plt.errorbar(xpointsOur, ypointsOur, yerr = stdOur, fmt=None, color='black')

    xpoints09 = xpoints[34:51]
    ypoints09 = ypoints[34:51]
    std09 = stderror[34:51]
    plt.plot(xpoints09, ypoints09, label=f"p=0.9")
    plt.errorbar(xpoints09, ypoints09, yerr = std09, fmt=None, color='black')

    xpoints08 = xpoints[51:68]
    ypoints08 = ypoints[51:68]
    std08 = stderror[51:68]
    plt.plot(xpoints08, ypoints08, label=f"p=0.8")
    plt.errorbar(xpoints08, ypoints08, yerr = std08, fmt=None, color='black')

    xpoints07 = xpoints[68:85] #85
    ypoints07 = ypoints[68:85]
    std07 = stderror[68:85]
    plt.plot(xpoints07, ypoints07, label=f"p=0.7")
    plt.errorbar(xpoints07, ypoints07, yerr = std07, fmt=None, color='black')

    xpoints06 = xpoints[85:90] # 102
    ypoints06 = ypoints[85:90]
    std06 = stderror[85:90]
    plt.plot(xpoints06, ypoints06, label=f"p=0.6")
    plt.errorbar(xpoints06, ypoints06, yerr = std06, fmt=None, color='black')

    xpoints05 = xpoints[102:107] # 119
    ypoints05 = ypoints[102:107]
    std05 = stderror[102:107]
    plt.plot(xpoints05, ypoints05, label=f"p=0.5")
    plt.errorbar(xpoints05, ypoints05, yerr = std05, fmt=None, color='black')

    xpointsLocal = xpoints[119:124] # 136
    ypointsLocal = ypoints[119:124]
    stdLocal = stderror[119:124]
    plt.plot(xpointsLocal,ypointsLocal, label=f"p=0 (local)")
    plt.errorbar(xpointsLocal, ypointsLocal, yerr = stdLocal, fmt=None, color='black')

    # plt.set_title("epsilon=0.1")

    plt.ylabel(r"error $\times 10^2$")
    plt.xlabel("n")#,labelpad=-10)
    # Put xlabel to the right
    # plt.legend(loc='upper left')

    plt.savefig('Figures/plotEpsilon01.pdf', bbox_inches='tight')
