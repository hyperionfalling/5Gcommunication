from math import log, log2, pow
from random import randrange
import numpy as np
import matplotlib.pyplot as plt
from pylab import *
#mpl.rcParams['font.sans-serif'] = ['SimHei']
from cvxpy import *
import cvxpy as cp

###固定参数
Bm = 1                                  #传输带宽 MHz -
N0 = -114                               #高斯白噪声功率 dBm -
N0 = (np.power(10,N0/10))               #换算为功率 单位mW
am = 3.5                                #路损因子 -

e = 10**(-9)
log2v = 0.6931471805599453              #log(2)用于换底
M2bits = 10 ** 6                        #MHz换算
log10v = 2.302585092994046

###1.随用户数目 M EE的变化趋势
def MwithEE(Bm,N0,M2bits,am,e,log2v):
    r = 300                             #半径 m
    pmax = 25                           #最大
    pmax = (np.power(10,pmax/10))       #换算为功率，单位mW
    Rms = 10 ** 4                       #最小传输速率 bits/s
    M1 = [k for k in range(2,9)]        #纵坐标用户数数目 2-8
    y1 = np.zeros(7)                    #保存平均分配算法的能效
    y2 = np.zeros(7)                    #保存优化算法的能效
    ###迭代用户数
    countM = 0
    for M in range(2,9,1):
        ###参数设定
        bm = 1/M                                    #权重
        dm = [r*(i/(M+1)) for i in range(1,M+1)]    #用户距离
        gm2 = np.random.gamma(128.,1/128.,M)        #gm^2 符合伽马分布
        mid = (1+np.power(dm,am))
        hm2 = np.multiply(gm2,1/mid)                #计算hm^2 公式直接化简计算

        ###平均分配算法计算
        pm = [pmax/M for _ in range(0,M)]           #平均分配 pm mW
        Rm = Bm * np.log2(1+10*log10(pm/N0*hm2))    #平均分配算法的传输速率Rm 单位 Mbs/s
        Rm = Rm * M2bits                            #速率和 单位 bits/s
        di = np.sum(np.multiply(pm,bm))             #带权重的功率和 单位 mW
        di = di/1000                                #功率和 单位 W=J/s
        y = np.sum(Rm)/di                           #平均分配能效 bits/s/(J/s) = bits/J
        y1[countM] = y                              #画图使用

        ###优化算法计算
        pmx = Variable(M)                   #优化分配 pmx mW
        yx = 0.1                            #优化分配能效 yx
        flag1 = 0                           #是否继续迭代
        c = 0                               #迭代轮数限制
        anew = np.multiply(1 / N0, hm2)

        ##迭代求最优值
        while(flag1 == 0):                  
            constraints = [cp.log(1+pmx*anew) * M2bits * 1000 / log2v >= Rms,
                           pmx >= 0.001,
                           cp.sum(pmx) == pmax] #约束，最小值，最大值
            ##优化问题
            obj = Maximize(cp.sum([cp.sum(cp.log(1+pmx*anew)), -cp.multiply(yx,cp.sum(cp.multiply(pmx,bm)))]))
            prob = Problem(obj, constraints)
            outv = prob.solve()
            nx = pmx.value
            if(np.abs(outv) >= e):
                yx = np.sum(np.log(1 + np.multiply(nx, anew))) / np.sum(np.multiply(nx, bm))
                c += 1
                print("函数F的值"+str(outv))
                print("第"+str(c)+"轮")
                print("函数值"+str(outv))
                print("能效为"+str(yx))
                print("功率分配为"+str(nx))
                #print("")
            else:
                yx = np.sum(np.log(1 + np.multiply(nx, anew))) / np.sum(np.multiply(nx, bm))
                flag1 = 1
                print("Done!")
            if(c == 10):
                flag1 = 1
                print("Overtime!")

        y2[countM] = yx * M2bits * 1000 / log2v
        countM += 1
        print("----")
    plt.plot(M1, y1, 'ro-', color='#4169E1', alpha=0.8, linewidth=1, label='average')
    plt.plot(M1, y2, 'ro-', color='#FF0000', alpha=0.8, linewidth=1, label='Dinkelbach')
    plt.legend(loc="upper right")
    plt.xlabel('usernumbers M')
    plt.ylabel('ee / J/s')
    plt.show()

###2.随传输距离 dm EE的变化趋势
def dmwithEE(Bm, N0, M2bits, am, e, log2v):
    M = 10   #用户数固定
    pmax = 25  # 最大
    pmax = (np.power(10, pmax / 10))  # 换算为功率，单位mW
    Rms = 10 ** 4  # 最小传输速率 bits/s
    r1 = [k for k in range(800, 1500, 100)]  # 纵坐标用户数数目 2-8
    y1 = np.zeros(7)  # 保存平均分配算法的能效
    y2 = np.zeros(7)  # 保存优化算法的能效
    ###迭代用户数
    countM = 0
    for r in range(200, 900, 100):
        ###参数设定
        bm = 1 / M  # 权重
        dm = [r * (i / (M + 1)) for i in range(1, M + 1)]  # 用户距离
        gm2 = np.random.gamma(128., 1 / 128., M)  # gm^2 符合伽马分布
        mid = (1 + np.power(dm, am))
        hm2 = np.multiply(gm2, 1 / mid)  # 计算hm^2 公式直接化简计算

        ###平均分配算法计算
        pm = [pmax / M for _ in range(0, M)]  # 平均分配 pm mW
        Rm = Bm * np.log2(1 + 10 * log10(pm / N0 * hm2))  # 平均分配算法的传输速率Rm 单位 Mbs/s
        Rm = Rm * M2bits  # 速率和 单位 bits/s
        di = np.sum(np.multiply(pm, bm))  # 带权重的功率和 单位 mW
        di = di / 1000  # 功率和 单位 W=J/s
        y = np.sum(Rm) / di  # 平均分配能效 bits/s/(J/s) = bits/J
        y1[countM] = y  # 画图使用

        ###优化算法计算
        pmx = Variable(M)  # 优化分配 pmx mW
        yx = 0.1  # 优化分配能效 yx
        flag1 = 0  # 是否继续迭代
        c = 0  # 迭代轮数限制
        anew = np.multiply(1 / N0, hm2)

        ##迭代求最优值
        while (flag1 == 0):
            constraints = [cp.log(1 + pmx * anew) * M2bits * 1000 / log2v >= Rms,
                           pmx >= 0.001,
                           cp.sum(pmx) == pmax]  # 约束，最小值，最大值
            ##优化问题
            obj = Maximize(cp.sum([cp.sum(cp.log(1 + pmx * anew)), -cp.multiply(yx, cp.sum(cp.multiply(pmx, bm)))]))
            prob = Problem(obj, constraints)
            outv = prob.solve()
            nx = pmx.value
            if (np.abs(outv) >= e):
                yx = np.sum(np.log(1 + np.multiply(nx, anew))) / np.sum(np.multiply(nx, bm))
                c += 1
                print("函数F的值" + str(outv))
                print("第" + str(c) + "轮")
                print("函数值" + str(outv))
                print("能效为" + str(yx))
                print("功率分配为" + str(nx))
                # print("")
            else:
                yx = np.sum(np.log(1 + np.multiply(nx, anew))) / np.sum(np.multiply(nx, bm))
                flag1 = 1
                print("Done!")
            if (c == 10):
                flag1 = 1
                print("Overtime!")

        y2[countM] = yx * M2bits * 1000 / log2v
        countM += 1
        print("----")
    plt.plot(r1, y1, 'ro-', color='#4169E1', alpha=0.8, linewidth=1, label='average')
    plt.plot(r1, y2, 'ro-', color='#FF0000', alpha=0.8, linewidth=1, label='Dinkelbach')
    plt.legend(loc="upper right")
    plt.xlabel('distances dm / m')
    plt.ylabel('ee / J/s')
    plt.show()

###3.随最大功率 pmax EE的变化趋势
def pmaxwithEE(Bm, N0, M2bits, am, e, log2v):
    M = 10   #用户数固定
    r = 300
    Rms = 10 ** 4  # 最小传输速率 bits/s
    pmax1 = [k for k in range(15, 50, 5)]  # 纵坐标用户数数目 2-8
    y1 = np.zeros(7)  # 保存平均分配算法的能效
    y2 = np.zeros(7)  # 保存优化算法的能效
    ###迭代用户数
    countM = 0
    for pmaxn in range(15, 50, 5):
        pmax = (np.power(10, pmaxn / 10))  # 换算为功率，单位mW
        ###参数设定
        bm = 1 / M  # 权重
        dm = [r * (i / (M + 1)) for i in range(1, M + 1)]  # 用户距离
        gm2 = np.random.gamma(128., 1 / 128., M)  # gm^2 符合伽马分布
        mid = (1 + np.power(dm, am))
        hm2 = np.multiply(gm2, 1 / mid)  # 计算hm^2 公式直接化简计算

        ###平均分配算法计算
        pm = [pmax / M for _ in range(0, M)]  # 平均分配 pm mW
        Rm = Bm * np.log2(1 + 10 * log10(pm / N0 * hm2))  # 平均分配算法的传输速率Rm 单位 Mbs/s
        Rm = Rm * M2bits  # 速率和 单位 bits/s
        di = np.sum(np.multiply(pm, bm))  # 带权重的功率和 单位 mW
        di = di / 1000  # 功率和 单位 W=J/s
        y = np.sum(Rm) / di  # 平均分配能效 bits/s/(J/s) = bits/J
        y1[countM] = y  # 画图使用

        ###优化算法计算
        pmx = Variable(M)  # 优化分配 pmx mW
        yx = 0.1  # 优化分配能效 yx
        flag1 = 0  # 是否继续迭代
        c = 0  # 迭代轮数限制
        anew = np.multiply(1 / N0, hm2)

        ##迭代求最优值
        while (flag1 == 0):
            constraints = [cp.log(1 + pmx * anew) * M2bits * 1000 / log2v >= Rms,
                           pmx >= 0.001,
                           cp.sum(pmx) == pmax]  # 约束，最小值，最大值
            ##优化问题
            obj = Maximize(cp.sum([cp.sum(cp.log(1 + pmx * anew)), -cp.multiply(yx, cp.sum(cp.multiply(pmx, bm)))]))
            prob = Problem(obj, constraints)
            outv = prob.solve()
            nx = pmx.value
            if (np.abs(outv) >= e):
                yx = np.sum(np.log(1 + np.multiply(nx, anew))) / np.sum(np.multiply(nx, bm))
                c += 1
                print("函数F的值" + str(outv))
                print("第" + str(c) + "轮")
                print("函数值" + str(outv))
                print("能效为" + str(yx))
                print("功率分配为" + str(nx))
                # print("")
            else:
                yx = np.sum(np.log(1 + np.multiply(nx, anew))) / np.sum(np.multiply(nx, bm))
                flag1 = 1
                print("Done!")
            if (c == 10):
                flag1 = 1
                print("Overtime!")

        y2[countM] = yx * M2bits * 1000 / log2v
        countM += 1
        print("----")
    plt.plot(pmax1, y1, 'ro-', color='#4169E1', alpha=0.8, linewidth=1, label='average')
    plt.plot(pmax1, y2, 'ro-', color='#FF0000', alpha=0.8, linewidth=1, label='Dinkelbach')
    plt.legend(loc="upper right")
    plt.xlabel('maxpower pmax / dBm')
    plt.ylabel('ee / J/s')
    plt.show()

###4.随最低速率要求 Rms EE的变化趋势
def RmswithEE(Bm, N0, M2bits, am, e, log2v):
    M = 10   #用户数固定
    r = 300
    pmax = 9.9  # 最大
    pmax = (np.power(10, pmax / 10))  # 换算为功率，单位mW
    #Rms = 10 ** 4  # 最小传输速率 bits/s
    Rms1 = [10**4, 10**5, 10**6, 10**7, 10**8, 10**9, 10**10]  # 纵坐标用户数数目 2-8
    y1 = np.zeros(7)  # 保存平均分配算法的能效
    y2 = np.zeros(7)  # 保存优化算法的能效
    ###迭代用户数
    countM = 0
    for Rms in Rms1:
        ###参数设定
        bm = 1 / M  # 权重
        dm = [r * (i / (M + 1)) for i in range(1, M + 1)]  # 用户距离
        gm2 = np.random.gamma(128., 1 / 128., M)  # gm^2 符合伽马分布
        mid = (1 + np.power(dm, am))
        hm2 = np.multiply(gm2, 1 / mid)  # 计算hm^2 公式直接化简计算

        ###平均分配算法计算
        pm = [pmax / M for _ in range(0, M)]  # 平均分配 pm mW
        Rm = Bm * np.log2(1 + 10 * log10(pm / N0 * hm2))  # 平均分配算法的传输速率Rm 单位 Mbs/s
        Rm = Rm * M2bits  # 速率和 单位 bits/s
        di = np.sum(np.multiply(pm, bm))  # 带权重的功率和 单位 mW
        di = di / 1000  # 功率和 单位 W=J/s
        y = np.sum(Rm) / di  # 平均分配能效 bits/s/(J/s) = bits/J
        y1[countM] = y  # 画图使用

        ###优化算法计算
        pmx = Variable(M)  # 优化分配 pmx mW
        yx = 0.1  # 优化分配能效 yx
        flag1 = 0  # 是否继续迭代
        c = 0  # 迭代轮数限制
        anew = np.multiply(1 / N0, hm2)

        ##迭代求最优值
        while (flag1 == 0):
            constraints = [cp.log(1 + pmx * anew) * M2bits * 1000 / log2v >= Rms,
                           pmx >= 0.001,
                           cp.sum(pmx) == pmax]  # 约束，最小值，最大值
            ##优化问题
            obj = Maximize(cp.sum([cp.sum(cp.log(1 + pmx * anew)), -cp.multiply(yx, cp.sum(cp.multiply(pmx, bm)))]))
            prob = Problem(obj, constraints)
            outv = prob.solve()
            nx = pmx.value
            if (np.abs(outv) >= e):
                yx = np.sum(np.log(1 + np.multiply(nx, anew))) / np.sum(np.multiply(nx, bm))
                c += 1
                print("函数F的值" + str(outv))
                print("第" + str(c) + "轮")
                print("函数值" + str(outv))
                print("能效为" + str(yx))
                print("功率分配为" + str(nx))
                # print("")
            else:
                yx = np.sum(np.log(1 + np.multiply(nx, anew))) / np.sum(np.multiply(nx, bm))
                flag1 = 1
                print("Done!")
            if (c == 10):
                flag1 = 1
                print("Overtime!")

        y2[countM] = yx * M2bits * 1000 / log2v
        countM += 1
        print("----")
    plt.xscale('log')
    plt.plot(Rms1, y1, 'ro-', color='#4169E1', alpha=0.8, linewidth=1, label='average')
    plt.plot(Rms1, y2, 'ro-', color='#FF0000', alpha=0.8, linewidth=1, label='Dinkelbach')
    plt.legend(loc="upper right")
    plt.xlabel('maxpower pmax / dBm')
    plt.ylabel('ee / J/s')
    plt.show()

#MwithEE(Bm=Bm,N0=N0,am=am,e=e,log2v=log2v,M2bits=M2bits)
#dmwithEE(Bm=Bm,N0=N0,am=am,e=e,log2v=log2v,M2bits=M2bits)
#pmaxwithEE(Bm=Bm,N0=N0,am=am,e=e,log2v=log2v,M2bits=M2bits)
#RmswithEE(Bm=Bm,N0=N0,am=am,e=e,log2v=log2v,M2bits=M2bits)



















