
import random
import numpy
import math
import matplotlib.pyplot as plt
from time import *
import tracemalloc
import numpy as np
from scipy.optimize import minimize


def is_zero(x):
    if x < 0:
        return x


def init():
    Positions = numpy.zeros((SearchAgents_no, dim))
    for i in range(dim):  # 形成SearchAgents_no*30个数[-100，100)以内
        Positions[:, i] = numpy.random.uniform(0, 1, SearchAgents_no) * (ub - lb) + lb
    return Positions

# 列表相减
def substract(a_list, b_list):
    a = len(a_list)
    new_list = []
    for i in range(0, a):
        new_list.append(a_list[i] - b_list[i])
    return new_list


# 列表相加
def add(a_list, b_list):
    a = len(a_list)
    new_list = []
    for i in range(0, a):
        new_list.append(a_list[i] + b_list[i])
    return new_list


# 列表的数乘
def multiply(a, b_list):
    b = len(b_list)
    new_list = []
    for i in range(0, b):
        new_list.append(a * b_list[i])
    return new_list


# 变异
#变异可以确保遗传基因多样性，防止陷入局部解
def mutation(np_list,NP,probDE1,F,ub,lb):
    # print("@")
    # print(probDE1)
    '''mutation'''
    bb=np.random.rand(NP)
    probiter=probDE1.copy()
    l2= probDE1[0]+probDE1[1]
    op_1 = []
    op_2 = []
    op_3 = []
    for i in bb:
        if i <= probiter[0]:
            op_1.append(1)
        else:
            op_1.append(0)
        if i > probiter[0] and i <= l2:
            op_2.append(1)
        else:
            op_2.append(0)
        if i > l2 and i <= 1:
            op_3.append(1)
        else:
            op_3.append(0)

    pNP = max(round(0.25 * NP), 1) # choose at least two best solutions
    randindex = np.ceil(np.random.rand(NP) * pNP) # select from [1, 2, 3, ..., pNP]
    phix = np.array([np_list[int(i)] for i in randindex])

    pNP_1 = max(round(0.5 * NP), 1)  # choose at least two best solutions
    randindex_1 = np.ceil(np.random.rand(NP) * pNP_1)  # select from [1, 2, 3, ..., pNP]
    phix_1 = np.array([np_list[int(i)] for i in randindex_1])

    v_list = []
    for i in range(0, NP):
        r1 = random.randint(0, NP - 1)
        while r1 == i:     #r1不能等于i，不能等于i的原因是防止之后进行的交叉操作出现自身和自身交叉的结果
            r1 = random.randint(0, NP - 1)
        r2 = random.randint(0, NP - 1)
        while r2 == r1 | r2 == i:
            r2 = random.randint(0, NP - 1)
        r3 = random.randint(0, NP - 1)
        while r3 == r2 | r3 == r1 | r3 == i:
            r3 = random.randint(0, NP - 1)
        #在DE中常见的差分策略是随机选取种群中的两个不同的个体，将其向量差缩放后与待变异个体进行向量合成
        #F为缩放因子F越小，算法对局部的搜索能力更好，F越大算法越能跳出局部极小点，但是收敛速度会变慢。此外，F还影响种群的多样性。

        if op_1[i]==1:
            vv_list=np.array(add(np_list[i], multiply(F[i], substract(add(substract(phix[random.randint(0,len(phix)-1)], np_list[i]),np_list[r1]),np_list[r2]))))
            vv_list[vv_list > ub] =ub
            vv_list[vv_list < lb] = lb
            v_list.append(vv_list)
        if op_2[i]==1:
            vv_list=np.array(add(np_list[i], multiply(F[i], substract(add(substract(phix[random.randint(0,len(phix)-1)], np_list[i]),np_list[r1]),np_list[r3]))))
            vv_list[vv_list > ub] =ub
            vv_list[vv_list < lb] = lb
            v_list.append(vv_list)

        if op_3[i]==1:
            vv_list=np.array(add(multiply(F[i], np_list[r1]), substract(phix[random.randint(0,len(phix)-1)], np_list[r3])))
            vv_list[vv_list > ub] =ub
            vv_list[vv_list < lb] = lb
            v_list.append(vv_list)

        # if i==0:
        #     print(f'r1:{r1};r2:{r2};r3:{r3}')
        #     print(v_list[0])
    return np.array(v_list), op_1, op_2, op_3


# 交叉
def crossover(np_list, v_list,NP,dim,cr):
    u_list = []
    # print("???")
    # print(len(u_list))
    # print(len(v_list))
    # mask=np.zeros(NP, dim)
    if random.random() < 0.4:
        for i in range(0, NP):
            vv_list = []
            jrand = random.randint(0, dim - 1)
            for j in range(0, dim):   #len_x=10
                if (random.random() > cr[0]) | (j == jrand):
                    #(j == random.randint(0, len_x - 1)是为了使变异中间体至少有一个基因遗传给下一代
                    vv_list.append(np_list[i][j])
                else:
                    # print(i)
                    # print(v_list[i])
                    vv_list.append(v_list[i][j])
            u_list.append(vv_list)
    else:
        u_list = np_list.copy()
        for i in range(0, NP):
            startLoc = random.randint(0, dim-1)
            l = startLoc
            while (random.random() < cr[i] and l < dim):
                l = l + 1
            for j in range(startLoc,l):
                # print(u_list[i])
                # print(v_list[i])
                u_list[i, j] = v_list[i, j]
    return np.array(u_list)
        #CR主要反映的是在交叉的过程中，子代与父代、中间变异体之间交换信息量的大小程度。CR的值越大，信息量交换的程度越大。
        #反之，如果CR的值偏小，将会使种群的多样性快速减小，不利于全局寻优。

# 选择
# def selection(u_list, np_list,NP):
#     for i in range(0, NP):
#         if object_function(u_list[i]) <= object_function(np_list[i]):
#             np_list[i] = u_list[i]
#         else:
#             np_list[i] = np_list[i]
#     return np_list

def IMODE_main(Positions, objf, lb, ub, dim, SearchAgents_no,Max_iter):
    Convergence_curve_IMODE = []
    '''define variables'''
    Printing = 1
    iter = 0
    current_eval = 0  # current fitness evaluations
    PS1 = SearchAgents_no  # define PS1
    PS2 = 100

    current_eval = current_eval + PS1
    Max_FES = 10000000
    InitPop = PS1

    '''calc. fit. and update FES'''
    Positions = numpy.array(sorted(Positions, key=lambda x: objf(x)))
    fitx=np.array([objf(i) for i in Positions])
    bestx=Positions[0]
    bestold = objf(Positions[0])
    res_det=np.array([bestold for i in range(PS1)])

    '''IMODE'''
    EA_1 = Positions
    EA_obj1 = [objf(i) for i in Positions]
    Pos = Positions.copy()
    Pos = list(Pos)
    random.shuffle(Pos)
    EA_1old = np.array(Pos)

    '''prob. of each DE operator'''
    probDE1=[1/3, 1/3, 1/3]

    '''archive data '''
    arch_rate = 2.6
    archive_NP = int(arch_rate * PS1)  # the maximum size of the archive
    archive_pop = []  # the solutions stored in te archive
    # print(archive_pop)
    archive_funvalues = []  # the function value of the archived solutions 需要加入到IMODE()

    '''to adapt CR and F'''
    hist_pos = 1
    memory_size = 20 * dim
    archive_f = np.ones(memory_size) * 0.2
    archive_Cr = np.ones(memory_size) * 0.2
    archive_T = np.ones(memory_size) * 0.1
    archive_freq = np.ones(memory_size) * 0.5

    stop_con = 0
    avgFE = Max_FES
    thrshold = 1e-08
    cy=0
    index=0
    Probs=np.ones(2)
    prob_ls=0.1

    F = np.random.normal(loc=0.5, scale=0.15, size=(1, PS1))  # 生成随机正态分布数
    cr = np.random.normal(loc=0.5, scale=0.15, size=(1, PS1))

    while stop_con==0:
        iter+=1
        cy+=1
        # print(current_eval,iter)

        if current_eval < Max_FES:
            UpdPopSize = round((((4 - InitPop) / Max_FES) * current_eval) + InitPop)
            # print(PS1,UpdPopSize)
            if PS1 > UpdPopSize:
                reduction_ind_num = PS1 - UpdPopSize
                if PS1 - reduction_ind_num < 4:
                    reduction_ind_num = PS1 - 4
                for r in range(reduction_ind_num):
                    vv = PS1
                    EA_1=np.delete(EA_1, vv - 1, axis=0)
                    EA_1old=np.delete(EA_1old, vv - 1, axis=0)
                    EA_obj1=np.delete(EA_obj1, vv - 1, axis=0)
                    PS1 = PS1 - 1
                # print(f'{PS1}//*/*')
                archive_NP=int(round(arch_rate*PS1))
                # print(archive_pop)
                if len(archive_pop) > archive_NP:
                    rndpos=list(range(len(archive_pop)))
                    random.shuffle(rndpos)
                    rndpos = rndpos[1: archive_NP]
                    archive_pop = [archive_pop[k] for k in rndpos]
            # print(probDE1)
            EA_1, EA_1old, EA_obj1, probDE1, bestold, bestx, hist_pos, archive_f, archive_Cr, current_eval, res_det, F, \
            cr ,archive_NP,archive_pop,archive_funvalues= \
                IMODE(EA_1, EA_1old, probDE1, bestold,bestx, hist_pos, memory_size, archive_f, archive_Cr, objf, lb, ub, dim, PS1,
                      current_eval, res_det, Printing,archive_NP,archive_pop,archive_funvalues)

            Convergence_curve_IMODE.append(bestold)

        '''LS2'''
        if current_eval > 0.85 * Max_FES and current_eval < Max_FES:
            if random.random()<prob_ls:
                old_fit_eva=current_eval
                LS_FE = min(np.ceil(20.0000e-003 * Max_FES), (Max_FES - current_eval))
                res = minimize(objf, bestx, method='SLSQP')
                if bestold-res.fun>0:
                    succ=1
                    bestold=res.fun
                    bestx=res.x
                else:
                    succ=0
                current_eval=current_eval+res.nfev

                if succ==1:
                    EA_1[len(EA_1)-1]=bestx
                    EA_obj1[len(EA_obj1)-1]=bestold
                    EA_1 = numpy.array(sorted(EA_1, key=lambda x: objf(x)))
                    EA_obj1=np.array([objf(i) for i in EA_1])
                    prob_ls=0.1
                else:
                    prob_ls=0.01

                if Printing == 1:
                    res_det = np.hstack((res_det, np.array([bestold for i in range(current_eval-old_fit_eva)])))

            Convergence_curve_IMODE.append(bestold)

        '''stopping criterion check'''
        if current_eval >= Max_FES - 4 * UpdPopSize or iter>Max_iter:
            stop_con = 1
            avgFE = current_eval

    return Convergence_curve_IMODE



def IMODE(Positions,xold,prob, bestold,bestx, hist_pos,memory_size, archive_f,archive_Cr, objf, lb, ub, dim, SearchAgents_no,
          current_eval,res_det,Printing,archive_NP,archive_pop,archive_funvalues):

    '''calc CR and F'''
    mem_rand_index = np.ceil(memory_size * np.random.rand(SearchAgents_no))
    mu_sf = np.array([archive_f[int(i)-1] for i in mem_rand_index])
    mu_cr = np.array([archive_Cr[int(i)-1] for i in mem_rand_index])
    '''generate CR'''
    cr = np.array([np.random.normal(loc=i, scale=0.1) for i in mu_cr])
    cr[cr > 1] = 1
    cr[cr < 0] = 0

    '''for generating scaling factor'''
    F = mu_sf + 0.1 * np.tan(np.pi * (np.random.rand(SearchAgents_no) - 0.5))
    pos = list(filter(is_zero, F))

    while pos:
        # print(pos)
        # print(F)
        # print(mu_sf)
        for i in pos:
            # print(np.random.rand(len(pos)))
            F[list(F).index(i)] = mu_sf[list(F).index(i)] + 0.1 * np.tan(np.pi * (random.random() - 0.5))
        pos = list(filter(is_zero, F))
    # 防止越界处理
    F[F > 1] = 1
    Positions = numpy.array(sorted(Positions, key=lambda x: objf(x)))
    fitx=np.array([objf(i) for i in Positions])
    # print(fitx)
    cr = np.array(sorted(cr))

    # print("!!!")
    # print(len(Positions),SearchAgents_no,prob)
    v_list, op_1, op_2, op_3 = mutation(Positions,SearchAgents_no,prob,F,ub,lb)    #变异
    # print(op_1,op_2,op_3)
    # print(v_list)
    u_list = crossover(Positions, v_list,SearchAgents_no,dim,cr)    #交叉np_list, v_list,NP,len_x,cr, dim
    # np_list = selection(u_list, Positions,NP)    #选择

    fitx_new=np.array([objf(i) for i in u_list])

    '''update FITNESS EVALUATIONS'''
    current_eval = current_eval + SearchAgents_no

    '''calc. imprv. for Cr and F'''
    # print(len(fitx))
    # print(len(fitx_new))
    diff=abs(fitx-fitx_new)
    I=(fitx_new < fitx)
    goodCR=np.array([cr[i] for i in range(SearchAgents_no) if I[i]==True])
    goodF=np.array([F[i] for i in range(SearchAgents_no) if I[i]==True])

    '''update archive'''

    popAll=archive_pop+[Positions[i] for i in range(SearchAgents_no) if I[i]==True]
    funvalues=archive_funvalues+[fitx[i] for i in range(SearchAgents_no) if I[i]==True]
    if len(popAll) <= archive_NP: # add all new individuals
        archive_pop = popAll
        archive_funvalues = funvalues
    else: # randomly remove some solutions
        # print(archive_NP)
        archive_pop = popAll[0:archive_NP]
        archive_funvalues = funvalues[0:archive_NP]

    '''update Prob. of each DE'''
    diff2=[]
    for i in range(len(fitx)):
        if fitx[i]==0:
            diff2.append(0)
        else:
            diff2.append(max(0,fitx[i]-fitx_new[i])/abs(fitx[i]))   #=np.array([max(0,fitx[i]-fitx_new[i])/abs(fitx[i]) for i in range(len(fitx))])
    diff2=np.array(diff2)
    diff2_op1=[]
    diff2_op2=[]
    diff2_op3=[]
    for i in range(len(op_1)):
        if op_1[i]==1:
            diff2_op1.append(diff2[i])
        if op_2[i]==1:
            diff2_op2.append(diff2[i])
        if op_3[i]==1:
            diff2_op3.append(diff2[i])
    if not diff2_op1:
        count_S1=0
    else:
        count_S1=max(np.mean(diff2_op1),0)
    if not diff2_op2:
        count_S2=0
    else:
        count_S2=max(np.mean(diff2_op2),0)
    if not diff2_op3:
        count_S3=0
    else:
        count_S3=max(np.mean(diff2_op3),0)
    # print("@@@")
    # print(diff2_op1)
    # print(diff2_op2)
    # print(diff2_op3)

    '''update probs.'''
    # print(count_S1,count_S2,count_S3)
    if count_S1 != 0 and count_S2 != 0 and count_S3 != 0:
        prob[0] = max(min(count_S1 / (count_S1 + count_S2 + count_S3), 0.9), 0.1)
        prob[1] = max(min(count_S2 / (count_S1 + count_S2 + count_S3), 0.9), 0.1)
        prob[2] = max(min(count_S3 / (count_S1 + count_S2 + count_S3), 0.9), 0.1)
    else:
        prob=[1/3, 1/3, 1/3]

    '''update x and fitx'''
    for i in range(SearchAgents_no):
        if I[i]==True:
            fitx[i]=fitx_new[i]
            xold[i] = Positions[i]
            Positions[i] = u_list[i]

    '''update memory cr and F'''
    num_success_params = len(goodCR)
    if num_success_params>0:
        sum_fiff_I1=0
        for i in range(SearchAgents_no):
            if I[i]==True:
                sum_fiff_I1+=diff[i]
        weightsDE = np.array([diff[i]/sum_fiff_I1 for i in range(SearchAgents_no) if I[i]==True]) #for updating the memory of scaling factor
        archive_f[hist_pos] = sum(weightsDE*np.power(goodF,2))/sum(weightsDE*goodF)

        if max(goodCR) == 0 or archive_Cr[hist_pos] == -1:
            archive_Cr[hist_pos] = -1
        else:
            archive_Cr[hist_pos] =sum(weightsDE*np.power(goodCR,2))/sum(weightsDE*goodCR)

        hist_pos+=1
        if hist_pos>=memory_size:
            hist_pos=1
    else:
        archive_Cr[hist_pos] = 0.5
        archive_f[hist_pos] = 0.5

    '''sort new x, fitness'''
    Positions = numpy.array(sorted(Positions, key=lambda x: objf(x)))
    xold = numpy.array(sorted(xold, key=lambda x: objf(x)))
    fitx=np.array([objf(i) for i in Positions])

    '''record the best value after checking its feasiblity status'''
    if fitx[0]<bestold and min(Positions[0])>=lb and max(Positions[0])<=ub:
        bestold=fitx[0]
        bestx=Positions[0]

    '''check to print'''
    if Printing==1:
        res_det = np.hstack((res_det,np.array([bestold for i in range(SearchAgents_no)])))


    return Positions, xold, fitx,prob,bestold,bestx,hist_pos, archive_f,archive_Cr,current_eval,res_det,F,cr,archive_NP,archive_pop,archive_funvalues



def F1(x):#[-100,100]
    dim=30
    lb=-100
    ub=100
    cc = 0
    for i in range(len(x)):
        cc+=(i+1)*(x[i]**4)
    s=cc+random.random()
    return s

def F2(x):#[-100,100]
    dim=30
    lb=-100
    ub=100
    cc=0
    c=1
    for i in range(len(x)):
        cc+=abs(x[i])
        c*=abs(x[i])
    s=cc+c
    return s

def F3(x):#[-100,100]
    dim=30
    lb=-100
    ub=100
    cc=0
    for i in range(1,len(x)+1):
        c=0
        for j in range(0,i):
            c+=x[j]
        cc+=c**2
    s=cc
    return s

def F4(x):#[-100,100]
    dim=30
    lb=-100
    ub=100
    cc=-99999999
    for i in range(0,len(x)):
        if abs(x[i])>cc:
            cc=abs(x[i])
    s=cc
    return s

def F5(x):#[-100,100]
    dim=30
    lb=-100
    ub=100
    ss = numpy.sum(x ** 2)
    s = ss**2
    return s

def F6(x):#[-100,100]
    dim=30
    lb=-100
    ub=100
    ss = 0
    for i in range(len(x)):
        ss+=abs(x[i])
    s = ss
    return s

def F7(x):
    dim=30
    lb=-32
    ub=32
    ss=numpy.sum(x**2)
    cc=0
    for i in range(len(x)):
        cc+=math.cos(2*math.pi*x[i])
    s=-20*math.exp(-0.2*math.sqrt(ss/len(x)))-math.exp(cc/len(x))+20+math.exp(1)
    return s

def F8(x):#[-100,100]
    dim=30
    lb=-100
    ub=100
    ss = numpy.sum(x**2/4000)+1
    c=1
    for i in range(1,len(x)+1):
        c*=math.cos(x[i-1]/math.sqrt(i))
    s = ss-c
    return s

def F9(x):#[-100,100]
    dim=30
    lb=-100
    ub=100
    ss = numpy.sum(x**2)
    s = 1-math.cos(2*math.pi*math.sqrt(ss))+0.1*math.sqrt(ss)
    return s

def F10(x):
    dim=30
    lb=-50
    ub=50
    a=5
    k=100
    m=4
    c=(math.sin(3*math.pi*x[0]))**2+((x[29]-1)**2)*(1+(math.sin(2*math.pi*x[29]))**2)
    for i in range(len(x)):
        c += ((x[i]-1)**2)*(1+(math.sin(3*math.pi*x[i])+1)**2)
    c=c*0.1
    for i in range(len(x)):
        if x[i]>a:
            c+=k*(x[i]-a)**m
        elif x[i]<-a:
            c+=k*(-x[i]-a)**m
    s=c
    return s

def F11(x):#[-1,1]
    dim=30
    lb=-5.12
    ub=5.12
    cc = 0
    for i in range(0, len(x)):
        cc += (x[i] ** 2-10*math.cos(2*math.pi*x[i])+10)
    s = cc
    return s

def F12(x):#[-1,1]
    dim = 30
    lb = -50
    ub = 50
    a = 10
    k = 100
    m = 4
    c = 10*(math.sin(math.pi * ((x[0]+1)/4+1)))**2 + ((x[29] + 1)/4) ** 2
    for i in range(len(x)-1):
        c += (((x[i] + 1)/4) ** 2)*(1+10*(math.sin(math.pi * ((x[i+1]+1)/4+1)))**2)
    c = c * (math.pi/30)
    for i in range(len(x)):
        if x[i] > a:
            c += k * (x[i] - a) ** m
        elif x[i] < -a:
            c += k * (-x[i] - a) ** m
    s = c
    return s

def F13(x):#[-5,10]
    dim = 30
    lb = -100
    ub = 100
    c=0
    for i in range(0, len(x)):
        c += (abs(x[i]+0.5))**2
    s=c
    return s

def F14(x):#[-30,30]
    dim = 30
    lb = -30
    ub = 30
    c=0
    for i in range(0, len(x)-1):
        c += (100*(x[i+1]-x[i]**2)**2+(x[i]-1)**2)
    s=c
    return s

def F15(x):#[-30,30]
    dim = 6
    lb = 0
    ub = 1
    A=[[10,3,17,3.5,1.7,8],
       [0.05,10,17,0.1,8,14],
       [3,3.5,1.7,10,17,8],
       [17,8,0.05,10,0.1,14]]
    C=[1,1.2,3,3.2]
    P=[[0.1312, 0.1696, 0.5569, 0.0124, 0.8283, 0.5586],
       [0.2329, 0.4135, 0.8307, 0.3736, 0.1004, 0.9991],
       [0.2348, 0.1451, 0.3522, 0.2883, 0.3047, 0.6650],
       [0.4047, 0.8828, 0.8732, 0.5743, 0.1091, 0.0381]]
    c=0
    for i in range(4):
        cc=0
        for j in range(6):
            cc+=A[i][j]*((x[j]-P[i][j])**2)
        c += C[i]*math.exp(-cc)
    s=-c+3.322
    return s

def F16(x):#[-30,30]
    dim = 4
    lb = 0
    ub = 10
    A=[[4,4,4,4],
       [1,1,1,1],
       [8,8,8,8],
       [6,6,6,6],
       [3,7,3,7],
       [2,9,2,9],
       [5,5,3,3],
       [8,1,8,1],
       [6,2,6,2],
       [7,3.6,7,3.6]]
    C=[0.1,0.2,0.2,0.4,0.4,0.6,0.3,0.7,0.5,0.5]
    c=0
    for i in range(10):
        cc = 0
        for j in range(4):
            cc += (x[j]-A[i][j])**2
        c += 1/(C[i] + cc)
    s=-c+10.5363
    return s

def F17(x):#[-30,30]
    dim = 4
    lb = 0
    ub = 10
    A=[[4,4,4,4],
       [1,1,1,1],
       [8,8,8,8],
       [6,6,6,6],
       [3,7,3,7]]
    C=[0.1,0.2,0.2,0.4,0.4]
    c=0
    for i in range(5):
        cc = 0
        for j in range(4):
            cc += (x[j]-A[i][j])**2
        c += 1/(C[i] + cc)
    s=-c+10.1532
    return s

def F18(x):#[-30,30]
    dim = 2
    lb = -100
    ub = 100
    s=x[0]**2+2*(x[1]**2)-0.3*math.cos(3*math.pi*x[0]+4*math.pi*x[1])+0.3
    return s


def F19(x):#[-30,30]
    dim = 30
    lb = -10
    ub = 10
    c=0
    cc=0
    ccc=0
    for i in range(0, len(x)):
        c += math.sin(x[i])**2
        cc+=x[i]**2
        ccc+=math.sin(math.sqrt(abs(x[i])))**2
    s=(c-math.exp(-cc))*math.exp(-ccc)+1
    return s


func_details = [[F1, -100, 100, 30],[F2, -100, 100, 30],[F3, -100, 100, 30],
                [F4, -100, 100, 30],[F5, -100, 100, 30],[F6, -100, 100, 30],
                [F7, -32, 32, 30],[F8, -100, 100, 30],[F9, -100, 100, 30],
                [F10, -50, 50, 30],[F11, -5.12, 5.12, 30],[F12, -50, 50, 30],
                [F13, -100, 100, 30],[F14, -30, 30, 30],[F15, 0, 1, 6],
                [F16, 0, 10, 4],[F17, 0, 10, 4],[F18, -100, 100, 2],
                [F19, -10, 10, 30]]


for fun in range(0,len(func_details)):
    Fx = func_details[fun][0]
    # outfile='result of experimental.txt'
    # output_path = outfile
    # with open(output_path, 'a', encoding='utf-8') as file1:
    #     print(f'F{fun + 1}',file=file1)
    print(f'F{fun+1}')

    Max_iter = 90#迭代次数
    lb = func_details[fun][1]#下界
    ub = func_details[fun][2]#上届
    dim = func_details[fun][3]#狼的寻值范围
    SearchAgents_no = 50#寻值的狼的数量
    positions=init()

    Y=[]
    for tn in range(10):
        begin_time = time()
        positions_2 = positions.copy()
        y = IMODE_main(positions_2, Fx, lb, ub, dim, SearchAgents_no,Max_iter)
        end_time = time()
        run_time = end_time - begin_time
        Y.append(y[len(y) - 1])
        # output_path = outfile
        # with open(output_path, 'a', encoding='utf-8') as file1:
        #     print(y, file=file1)
        #     print(run_time, file=file1)
        print(y)
        print(run_time,len(y))
    print(f'IMODE平均值{np.mean(Y)}标准差{np.std(Y, ddof=1)}')
    # output_path = outfile
    # with open(output_path, 'a', encoding='utf-8') as file1:
    #     print(f'传统灰狼平均值{np.mean(Y)}标准差{np.std(Y, ddof=1)}', file=file1)
    #     print(file=file1)
    print()