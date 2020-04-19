import numpy as np
from scipy import optimize
from sympy.solvers import solve
from scipy.optimize import fsolve
from scipy.optimize import root
from sympy import Symbol
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.optimize import newton
from scipy.optimize import bisect
import six 
from six.moves import zip
import csv
from numpy import genfromtxt
import xlsxwriter as xlwt
from matplotlib.ticker import FuncFormatter
import pandas as pd
# ========================================== parameters ========================================== #
global tsN, tuN, N, Aa, Am, Da, Dm, Dr, Ar
# ========================================== parameters ========================================== #
T = 6                                               # Time horizon
n = 2                                               # Tnumber of scenarios
nrcp = 4
nsc = 5
alpha = 0.55/2                                      # Agricultural share in Consumption function
eps = 0.75                                          # Elasticity of substitution in Consumption function

coeffTmax = [12.4, 0.8798]
# ========================================== SSP Scenarios =========================================== # 
RCPName = ['RCP 2.6 ', 'RCP 4.5', 'RCP 6.0', 'RCP 8.5']
#TempMean = ([14.00817, 14.54351, 14.83102, 15.07116, 15.04836, 15.04016], [14.00817, 14.40499, 15.29290, 16.39785, 17.61407, 18.79937])
#TempMax = ([23.39715, 24.345, 24.39025, 24.78146, 24.66104, 24.44457], [23.37335, 24.09103, 24.99267, 26.31668, 27.35580, 28.39855])

# ========================================== Temperature =========================================== #
# Pole temperature 0 and Equator temperature 28
# Temp = - 28/(pi/2) * (lat - pi/2) 
mu1 = 0.21                                  #in Desmet it is 0.0003 but they report it after dividing by 1000/pi/2
mu2 = 0.5
mu3 = 0.0238

# ========================================== Damages =========================================== #
# D = g0 + g1 * T + g2 * T^2

# Agricultural parameters
g0a = -2.24
g1a = 0.308
g2a = -0.0073

# Manufacturing parameters
g0m = 0.3
g1m = 0.08
g2m = -0.0023

# ========================================== Variables =========================================== #                    

# == temperature == #
Temp = np.zeros((T, nrcp, nsc))                     # Mean Temperature
# == child-rearing time == #
gamma0 = 0.4                                        # Share of children's welbeing in Utility function of parents in 1980

# == Age matrix == #
nu = np.zeros((T, nrcp, nsc))                       # number of unskilled children
ns = np.zeros((T, nrcp, nsc))                       # number of skilled children
L = np.zeros((T, nrcp, nsc))                        # Number of unskilled parents
H = np.zeros((T, nrcp, nsc))                        # Number of skilled parents
h = np.zeros((T, nrcp, nsc))                        # Ratio of skilled to unskilled labor h=H/L
hN = np.zeros((T, nrcp, nsc))                       # Ratio of skilled to unskilled children h=ns/nu
N = np.zeros((T, nrcp, nsc))                        # Adult population
Pop = np.zeros((T, nrcp, nsc))                      # total population
Pgr = np.zeros((T, nrcp, nsc))                      # population growth rate

# == Prices == #
pa = np.zeros((T, nrcp, nsc))                       # Pice of AgricuLtural good
pm = np.zeros((T, nrcp, nsc))                       # Pice of Manufacturing good
pr = np.zeros((T, nrcp, nsc))                       # Relative pice of Manufacturing to Agricultural goods

# == Wages == #
wu = np.zeros((T, nrcp, nsc))                       # Wage of unskilled labor
ws = np.zeros((T, nrcp, nsc))                       # Wage of skilled labor
wr = np.zeros((T, nrcp, nsc))                       # Wage ratio of skilled to unskilled labor

# == Technology == #
Aa = np.zeros((T, nrcp, nsc))                       # Technological growth function for Agriculture
Am = np.zeros((T, nrcp, nsc))                       # Technological growth function for Manufacurng
Ar = np.zeros((T, nrcp, nsc))                       # ratio of Technology in Manufacurng to Agriculture
Aag = np.zeros((nrcp))                              # growth rate of Agricultural productivity
Amg = np.zeros((nrcp))                              # growth rate of Manufacturing productivity
Amgr = 0.01                                         # annual growth rate of Manufacturing productivity

# == Output == #
Y = np.zeros((T, nrcp, nsc))                        # Total output
Ya = np.zeros((T, nrcp, nsc))                       # AgricuLtural output
Ym = np.zeros((T, nrcp, nsc))                       # Manufacturing output
Yr = np.zeros((T, nrcp, nsc))                       # Ratio of Manufacturing output to Agricultural output
Cr = np.zeros((T, nrcp, nsc))                       # Ratio of Manufacturing output to Agricultural consumption
Ypc = np.zeros((T, nrcp, nsc))                      # Output per adult

# == Output == #
Da = np.zeros((T, nrcp, nsc))                       # AgricuLtural damage
Dm = np.zeros((T, nrcp, nsc))                       # Manufacturing damage
Dr = np.zeros((T, nrcp, nsc))                       # Ratio of Manufacturing damages to Agricultural damages

# == Availability == #
Su = 1 + np.zeros((T, nrcp, nsc))                   # Availability of unskilled labor
Ss = 1 + np.zeros((T, nrcp, nsc))                   # Availability of skilled labor
Sr = np.zeros((T, nrcp, nsc))                       # Ratio of Availability of skilled to unskilled labor

delta = np.zeros((T, nrcp, nsc))                    # food consumption factor

# == Consumption == #
cau = np.zeros((T, nrcp, nsc))                      # consumption of agricultural good unskilled
cas = np.zeros((T, nrcp, nsc))                      # consumption of agricultural good skilled
cmu = np.zeros((T, nrcp, nsc))                      # consumption of manufacturing good unskilled
cms = np.zeros((T, nrcp, nsc))                      # consumption of manufacturing good skilled
cu = np.zeros((T, nrcp, nsc))                       # consumption of all goods unskilled
cs = np.zeros((T, nrcp, nsc))                       # consumption of all goods skilled

# ============================================== Country Calibration ============================================== #
#Noed = [15.10436, 24.74125, 33.85946, 37.16224, 34.96213, 30.5804]
#Prim = [7.71398, 15.8682, 28.26843, 39.29705, 42.667, 37.66566]
#Sec = [1.16623, 4.14363, 11.89684, 26.33155, 45.01256, 62.36497]
#Ter = [0.22855, 0.73771, 2.31713, 6.19394, 13.29229, 23.1466]
#urb = [12.082, 17.28766044, 26.83436038, 37.08907872, 46.45558788, 54.16787286]
#Qntdata = [6.2, 10.4, 14.6, 20.6, 48.2]
#gdp = [0.00697831, 0.026344246, 0.104490753, 0.36820906, 1.058161757, 2.396862943]
Noed = [0] * T
Prim = [0] * T
Sec = [0] * T
Ter = [0] * T
urb = [0] * T
gdp = [0] * T
Yearname = [i for i in range(2000, 2101, 20)]

# =========== Scenarios ============ #
Pro_data = pd.read_csv('C:/Users/Shayegh/Dropbox/Research/Papers/1_In Progress/14-CFS_Climate and Food Security/Model/Input/CFS_Labor.csv')
Cal_data = pd.read_csv('C:/Users/Shayegh/Dropbox/Research/Papers/1_In Progress/14-CFS_Climate and Food Security/Model/Input/CFS_Cal.csv')
Tempdata = pd.read_csv('C:/Users/Shayegh/Dropbox/Research/Papers/1_In Progress/14-CFS_Climate and Food Security/Model/Input/Temp.csv')
Tdata = pd.read_csv('C:/Users/Shayegh/Dropbox/Research/Papers/1_In Progress/14-CFS_Climate and Food Security/Model/Input/witch_dataset_long_uganda.csv')

Ndata = np.zeros((T, nrcp))
Ldata = np.zeros((T, nrcp))
Hdata = np.zeros((T, nrcp))
hdata = np.zeros((T, nrcp))
Condata = np.zeros((T, nrcp, nsc))
Init = np.zeros((3))

for i in range(T):
    urb[i] = float(Tdata.loc[(Tdata['n'] == 'uganda')&(Tdata['variable'] == 'urbanisation')&(Tdata['year'] == Yearname[i])]['value'].values[0])
    Noed[i] = Tdata.loc[(Tdata['n'] == 'uganda')&(Tdata['variable'] == 'education')&(Tdata['year'] == Yearname[i])&(Tdata['edu'] == 'No Education')]['value2'].values[0] * (1 - urb[i]/100)
    Prim[i] = Tdata.loc[(Tdata['n'] == 'uganda')&(Tdata['variable'] == 'education')&(Tdata['year'] == Yearname[i])&(Tdata['edu'] == 'Primary Education')]['value2'].values[0] * (1 - urb[i]/100)
    Sec[i] = Tdata.loc[(Tdata['n'] == 'uganda')&(Tdata['variable'] == 'education')&(Tdata['year'] == Yearname[i])&(Tdata['edu'] == 'Secondary Education')]['value2'].values[0] * (1 - urb[i]/100)
    Ter[i] = Tdata.loc[(Tdata['n'] == 'uganda')&(Tdata['variable'] == 'education')&(Tdata['year'] == Yearname[i])&(Tdata['edu'] == 'Tertiary Education')]['value2'].values[0] * (1 - urb[i]/100)
    gdp[i] = float(Tdata.loc[(Tdata['n'] == 'uganda')&(Tdata['variable'] == 'ykali')&(Tdata['year'] == Yearname[i])]['value'].values[0]) * (1 - urb[i]/100)
#
    for j in range(nrcp):
        Ldata[i, j] = (Noed[i] + Prim[i])
        Hdata[i, j] = (Sec[i] + Ter[i])
        Ndata[i, j] = Ldata[i, j] + Hdata[i, j]
        hdata[i, j] = Hdata[i, j]/Ldata[i, j]
        Temp[i, j, :] = Tempdata.value[i + 6 * j]
        
urb1980 = float(Tdata.loc[(Tdata['n'] == 'uganda')&(Tdata['variable'] == 'urbanisation')&(Tdata['year'] == 1980)]['value'].values[0])
Noed1980 = Tdata.loc[(Tdata['n'] == 'uganda')&(Tdata['variable'] == 'education')&(Tdata['year'] == 1980)&(Tdata['edu'] == 'No Education')]['value2'].values[0] * (1 - urb1980/100)
Prim1980 = Tdata.loc[(Tdata['n'] == 'uganda')&(Tdata['variable'] == 'education')&(Tdata['year'] == 1980)&(Tdata['edu'] == 'Primary Education')]['value2'].values[0] * (1 - urb1980/100)
Sec1980 = Tdata.loc[(Tdata['n'] == 'uganda')&(Tdata['variable'] == 'education')&(Tdata['year'] == 1980)&(Tdata['edu'] == 'Secondary Education')]['value2'].values[0] * (1 - urb1980/100)
Ter1980 = Tdata.loc[(Tdata['n'] == 'uganda')&(Tdata['variable'] == 'education')&(Tdata['year'] == 1980)&(Tdata['edu'] == 'Tertiary Education')]['value2'].values[0] * (1 - urb1980/100)
Ldata1980 = (Noed1980 + Prim1980)
Hdata1980 = (Sec1980 + Ter1980)
Ndata1980 = Ldata1980 + Hdata1980

Init = [Ndata1980, gdp[0] * 1e6, Temp[0, 0, 0]]
### ============================================== Model Calibration ============================================== #
def Calib(hdatax, Ndatax, INITx, Amgrx):
    [N00, Y0, Temp0] = INITx
    h0 = hdatax[0]
    popg0 = (Ndatax[0] - N00)/N00
    nu0 = (1 + popg0) / (1 + h0)
    ns0 = (1 + popg0) * h0 / (1 + h0)
    
    TIndex = int(round((Temp0 - 5) * 5))
    scale_u0 = max(Pro_data.hours)
    Su0 = (Pro_data.hours[TIndex])/(scale_u0)
    Ss0 = 1
    Sr0 = Ss0/Su0
    
    Iu0 = 412400.00 #https://tradingeconomics.com/uganda/wages-low-skilled
    Is0 = 1216600.00 #https://tradingeconomics.com/uganda/wages-high-skilled
    tr0 = Is0/Iu0 * Sr0
    tu0 = gamma0 / (tr0 * ns0 + nu0)
    ts0 = tu0 * tr0
    
    TIndexCal = int(round((Temp0 - 10) * 5))
    scale_cal0 = np.exp(Cal_data.lncal[TIndexCal])
    delta0 = np.exp(Cal_data.lncal[TIndexCal])/(scale_cal0)
    alpha0 = alpha * delta0
    
    hx = hdatax[T - 1]
    
    Da0 = max(0.001, g0a + g1a * Temp0 + g2a * Temp0**2)
    Dm0 = max(0.001, g0m + g1m * Temp0 + g2m * Temp0**2)
    Dr0 = Dm0/Da0
    
    L0 = nu0 * N00
    H0 = ns0 * N00
    N0 =  H0 + L0
    
    Ar0 = np.exp((eps/(1-eps)) * (np.log((1-alpha0)/alpha0) - np.log(tr0) + 1/eps * (- np.log(h0)))- np.log(Dr0) - np.log(Sr0))
    Am0 = Y0/((alpha0 * (Su0 * L0 * Da0 / Ar0)**((eps - 1)/eps) + (1 - alpha0) * (Ss0 * H0 * Dm0)**((eps - 1)/eps))**(eps/(eps - 1)))
    Aa0 = Am0/Ar0
    
    Arx = np.exp((eps/(1-eps)) * (np.log((1-alpha0)/alpha0) - np.log(tr0) + 1/eps * (- np.log(hx)))- np.log(Dr0) - np.log(Sr0))

    Arg = np.exp((np.log(Arx/Ar0))/((2100 - 2000)/20)) - 1
    
    Amgx = (1 + Amgrx)**20 - 1
    Aagx = (1 + Amgx)/(1 + Arg) - 1
    
    Ya0 = Aa0 * Su0 * L0 * Da0
    Ym0 = Am0 * Ss0 * H0 * Dm0
    Yr0 = Ym0 / Ya0
    
    pr0 = (Yr0)**(-1/eps) * (1 - alpha0) / alpha0
    
    YA0 = Y0 / N0
    cmu0 = Ym0 / (H0 * tr0 * Su0/Ss0 + L0)
    cms0 = cmu0 * tr0 * Su0/Ss0
    cau0 = Ya0 / (H0 * tr0 * Su0/Ss0 + L0)
    cas0 = cau0 * tr0 * Su0/Ss0    
    cu0 = (alpha0 * (cau0)**((eps - 1)/eps) + (1 - alpha0) * cmu0**((eps - 1)/eps))**(eps/(eps - 1))
    cs0 = (alpha0 * (cas0)**((eps - 1)/eps) + (1 - alpha0) * cms0**((eps - 1)/eps))**(eps/(eps - 1))
    wu0 = cu0 / (1 - gamma0)
    ws0 = cs0 / (1 - gamma0)
    pa0 = wu0 / (Da0 * Aa0)
    pm0 = ws0 / (Dm0 * Am0)
    wr0 = ws0/wu0
    Cr0 = (Ym0 * pm0) / (Ya0 * pa0)
    
    Outputx = [N0, h0, Da0, Dm0, Dr0, Su0, Ss0, Sr0, H0, L0, Y0, YA0, Ya0, Ym0, Yr0, Aa0, Am0, cmu0, cms0, cau0, cas0, cu0, cs0, wu0, ws0, wr0, pa0, pm0, pr0, delta0, Cr0]
    Ratex = [Aagx, Amgx, tu0, ts0]
    return (Outputx, Ratex)

# ============================================== Model Dynamics ============================================== #
#sc = 0 : No climate chnage
#sc = 1 : productivity damages
#sc = 2 : availability damages
#sc = 3 : consumption damages
#sc = 4 : all damages
for sc in range(nsc):
    for j in range(nrcp):
        [Output, Rate] = Calib(hdata[:,j], Ndata[:,j], Init, Amgr)
        [Aag[j], Amg[j], tu, ts] = Rate
        [N[0, j, sc], h[0, j, sc], Da[0, j, sc], Dm[0, j, sc], Dr[0, j, sc], Su[0, j, sc], Ss[0, j, sc], Sr[0, j, sc], H[0, j, sc], L[0, j, sc], Y[0, j, sc], Ypc[0, j, sc], Ya[0, j, sc], Ym[0, j, sc], Yr[0, j, sc], Aa[0, j, sc], Am[0, j, sc], cmu[0, j, sc], cms[0, j, sc], cau[0, j, sc], cas[0, j, sc], cu[0, j, sc], cs[0, j, sc], wu[0, j, sc], ws[0, j, sc], wr[0, j, sc], pa[0, j, sc], pm[0, j, sc], pr[0, j, sc], delta[0, j, sc], Cr[0, j, sc]] = Output
                         
        for i in range(T - 1):
            if i == 0:
                Temp[i, j, sc] = Temp[0, 0, 0]
            trN = ts/tu
            if sc == 0:
                Temp[i + 1, j, sc] = Temp[0, 0, 0]
                
            TempN = Temp[i + 1, j, sc]
            if sc == 0 or sc == 1 or sc == 4:    
                Da[i + 1, j, sc] = max(0.001, g0a + g1a * TempN + g2a * TempN**2)
                Dm[i + 1, j, sc] = max(0.001, g0m + g1m * TempN + g2m * TempN**2)
            else:
               Da[i + 1, j, sc] = Da[i + 1, j, 0]
               Dm[i + 1, j, sc] = Dm[i + 1, j, 0]
            Dr[i + 1, j, sc] = Dm[i + 1, j, sc]/Da[i + 1, j, sc]
            DrN = Dr[i + 1, j, sc]
            
            Aa[i + 1, j, sc] = Aa[i, j, sc] * (1 + Aag[j])
            Am[i + 1, j, sc] = Am[i, j, sc] * (1 + Amg[j])
            Ar[i + 1, j, sc] = Am[i + 1, j, sc]/Aa[i + 1, j, sc]
            ArN = Ar[i + 1, j, sc]
            
            TempIndex = int(round((TempN - 5) * 5))
            scale_u = max(Pro_data.hours)
            scale_s = 1
            if sc == 0 or sc == 2 or sc == 4:
                Su[i + 1, j, sc] = (Pro_data.hours[TempIndex])/(scale_u)
                Ss[i + 1, j, sc] = 1
            else:
                Su[i + 1, j, sc] = Su[i + 1, j, 0]
                Ss[i + 1, j, sc] = Ss[i + 1, j, 0]
            Sr[i + 1, j, sc] = Ss[i + 1, j, sc]/Su[i + 1, j, sc]
            SrN = Sr[i + 1, j, sc]
    
            TempIndexCal = int(round((TempN - 10) * 5))
            scale_cal = np.exp(Cal_data.lncal[int(round((Temp[0, 0, 0] - 10) * 5))])
            if sc == 0 or sc == 3 or sc == 4:
                delta[i + 1, j, sc] = np.exp(Cal_data.lncal[TempIndexCal])/(scale_cal)
            else:
                delta[i + 1, j, sc] = delta[i + 1, j, 0]
            deltaN = delta[i + 1, j, sc]
            alphaN = deltaN * alpha
    
            N[i + 1, j, sc] = Ndata[i + 1, j]
            NN = N[i + 1, j, sc]
            
            hN = np.exp(eps * (np.log((1-alphaN)/alphaN) - np.log(trN)) - (1 - eps) * (np.log(DrN) + np.log(ArN) + np.log(SrN)))
            
            h[i + 1, j, sc] = hN
            L[i + 1, j, sc] = NN/(hN + 1)
            H[i + 1, j, sc] = hN * L[i + 1, j, sc]
            
            Ya[i + 1, j, sc] = Aa[i + 1, j, sc] * L[i + 1, j, sc] * Su[i + 1, j, sc] * Da[i + 1, j, sc]
            Ym[i + 1, j, sc] = Am[i + 1, j, sc] * H[i + 1, j, sc] * Ss[i + 1, j, sc] * Dm[i + 1, j, sc]
            Yr[i + 1, j, sc] = Ym[i + 1, j, sc] / (Ya[i + 1, j, sc])
            
            pr[i + 1, j, sc] = (Yr[i + 1, j, sc])**(-1/eps) * alphaN / (1 - alphaN)
            
            cmu[i + 1, j, sc] = Ym[i + 1, j, sc] / (H[i + 1, j, sc] * trN/SrN + L[i + 1, j, sc])
            cms[i + 1, j, sc] = cmu[i + 1, j, sc] * trN/SrN
            cau[i + 1, j, sc] = Ya[i + 1, j, sc] / (H[i + 1, j, sc] * trN/SrN + L[i + 1, j, sc])
            cas[i + 1, j, sc] = cau[i + 1, j, sc] * trN/SrN
            cu[i + 1, j, sc] = (alphaN * (cau[i + 1, j, sc])**((eps - 1)/eps) + (1 - alphaN) * cmu[i + 1, j, sc]**((eps - 1)/eps))**(eps/(eps - 1))
            cs[i + 1, j, sc] = (alphaN * (cas[i + 1, j, sc])**((eps - 1)/eps) + (1 - alphaN) * cms[i + 1, j, sc]**((eps - 1)/eps))**(eps/(eps - 1))
            wu[i + 1, j, sc] = cu[i + 1, j, sc] / (1 - gamma0)
            ws[i + 1, j, sc] = cs[i + 1, j, sc] / (1 - gamma0)
            pa[i + 1, j, sc] = wu[i + 1, j, sc] / (Da[i + 1, j, sc] * Aa[i + 1, j, sc])
            pm[i + 1, j, sc] = ws[i + 1, j, sc] / (Dm[i + 1, j, sc] * Am[i + 1, j, sc])
            Y[i + 1, j, sc] = (pa[i + 1, j, sc] * deltaN * Ya[i + 1, j, sc] + pm[i + 1, j, sc] * Ym[i + 1, j, sc]) * (1 - gamma0)
            wr[i + 1, j, sc] = ws[i + 1, j, sc]/wu[i + 1, j, sc]        
            Ypc[i + 1, j, sc] = Y[i + 1, j, sc] / (N[i + 1, j, sc])
            Cr[i + 1, j, sc] = (Ym[i + 1, j, sc] * pm[i + 1, j, sc]) / (Ya[i + 1, j, sc] * pa[i + 1, j, sc])

# ===================================================== Output ===================================================== #    
x = [2000, 2020, 2040, 2060, 2080, 2100]

for j in range(nrcp):
    plt.plot(x, h[:, j, 0], 'b', label = "Baseline")
    plt.plot(x, h[:, j, 1], 'g', label = "Only production damages")
    plt.plot(x, h[:, j, 2], 'r', label = "Only labor damages")
    plt.plot(x, h[:, j, 3], 'y', label = "Only food consumption damages")
    plt.plot(x, h[:, j, 4], 'k', label = "All damages")
    plt.xlabel('Time')
    plt.ylabel('skill ratio')
    plt.title('high-skilled to low-skilled labor under ' + RCPName[j])
    axes = plt.gca()
    #axes.yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.0%}'.format(y))) 
    plt.xticks(np.arange(min(x), max(x) + 1, 20))
    plt.legend(loc=2, prop={'size':8})
    plt.show()
    
# =============================================== Export into Excel =============================================== #

def output(filename, sheet, list1, v):
    book = xlwt.Workbook(filename)
    sh = book.add_worksheet(sheet)

    v1_desc = 'alpha'
    v2_desc = 'eps'
    v3_desc = 'gamma0'
    v4_desc = 'Tu'
    v5_desc = 'Ts'

    desc = [v1_desc, v2_desc, v3_desc, v4_desc, v5_desc]
    m = 0
    for v_desc, v_v in zip(desc, v):
        sh.write(m, 0, v_desc)
        sh.write(m, 1, v_v)
        m = m + 1
        
    varname = ['Time', 'delta', 'Yr', 'Cr', 'L', 'H', 'N', 'h', 'wu', 'ws', 'Temp', 'Temp', 'Da', 'Dm', 'Su', 'Ss', 'Pa', 'Pm', 'Ya', 'Ym', 'Ypc']
    
    for n in range(nsc):
        m = 6 + 25 * n
        for j in range(nrcp):
            for indx , q in enumerate(range(2000, 2120, 20), 1):
                sh.write(m + 0, j * 10 + indx, q)
                sh.write(m + 0, j * 10, varname[0])
            for k in range(20):
                for indx in range(T):
                    sh.write(m + k + 1, j * 10 + indx + 1, list1[k][indx][j][n])
                    sh.write(m + k + 1, j * 10, varname[k + 1]) 
    book.close()
    
output1 = [delta, Yr, Cr, L, H, N, h, wu, ws, Temp, Temp, Da, Dm, Su, Ss, pa, pm, Ya, Ym, Ypc]
par = [alpha, eps, gamma0, tu, ts]

#output('CFS_Uganda_Output.xlsx', 'Sheet1', output1, par)