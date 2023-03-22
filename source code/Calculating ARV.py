"""
类别: 算法
名称: 基于退火算子和差分算子的灰狼优化算法
作者: 孙质方
邮件: zf_sun@vip.hnist.edu.cn
日期: 2021年12月26日
说明:
"""
import random
import numpy
import math
import matplotlib.pyplot as plt
from time import *
import tracemalloc
import numpy as np


# 种群初始化
def initialtion(NP,len_x,value_down_range,value_up_range):
    np_list = []  # 种群，染色体
    for i in range(0, NP):
        x_list = []  # 个体，基因
        for j in range(0, len_x):
            x_list.append(value_down_range + random.random() * (value_up_range - value_down_range))
        np_list.append(x_list)
    return np_list


# 列表相减
def substract(a_list, b_list,lb,ub):
    a = len(a_list)
    new_list = []
    for i in range(0, a):
        temp=a_list[i] - b_list[i]
        if temp < lb:
            temp = lb
        elif temp > ub:
            temp = ub
        new_list.append(temp)
    return new_list


# 列表相加
def add(a_list, b_list,lb,ub):
    a = len(a_list)
    new_list = []
    for i in range(0, a):
        temp = a_list[i] + b_list[i]
        if temp < lb:
            temp = lb
        elif temp > ub:
            temp = ub
        new_list.append(temp)
    return new_list


# 列表的数乘
def multiply(a, b_list,lb,ub):
    b = len(b_list)
    new_list = []
    for i in range(0, b):
        temp = a * b_list[i]
        if temp < lb:
            temp = lb
        elif temp > ub:
            temp = ub
        new_list.append(temp)
    return new_list

#灰狼
def init():
    Positions = numpy.zeros((SearchAgents_no, dim))
    for i in range(dim):  # 形成SearchAgents_no*30个数[-100，100)以内
        Positions[:, i] = numpy.random.uniform(0, 1, SearchAgents_no) * (ub - lb) + lb
    return Positions

def PSO_GWO_init():
    Positions = numpy.zeros((SearchAgents_no, dim))
    for i in range(SearchAgents_no):
        x0=random.random()
        x0_list=[-1,-1,-1,x0]
        while x0==0.2 or x0==0.4 or x0==0.6 or x0==0.8:
            x0 = random.random()
        Positions[i,0]=x0
        for j in range(1,dim):
            if Positions[i, j - 1] <= 0.5:
                Positions[i, j] = Positions[i, j - 1] * 2
                if Positions[i, j]==0 or Positions[i, j]==0.25 or Positions[i, j]==0.5 or Positions[i, j]==0.75 or (Positions[i, j] in x0_list):
                    x0=(x0+random.random()) % 1
                    x0_list.append(x0)
                    x0_list.remove(x0_list[0])
                    Positions[i, j]=x0
            else:
                Positions[i, j] = (1 - Positions[i, j - 1]) * 2
                if Positions[i, j]==0 or Positions[i, j]==0.25 or Positions[i, j]==0.5 or Positions[i, j]==0.75 or (Positions[i, j] in x0_list):
                    x0=(x0+random.random()) % 1
                    x0_list.append(x0)
                    x0_list.remove(x0_list[0])
                    Positions[i, j]=x0
    for i in range(SearchAgents_no):
        for j in range(dim):
            Positions[i,j]=Positions[i,j]* (ub - lb) + lb
    return Positions

def beta_init():
    Positions = numpy.zeros((SearchAgents_no, dim))
    for i in range(SearchAgents_no):
        for j in range(dim):
            Positions[i, j] = random.betavariate(1.2,1.2) * (ub - lb) + lb
    return Positions

def re_gene(Positions,objf,lenth):
    init_Positions = numpy.zeros((SearchAgents_no +lenth, dim))
    # positions = numpy.zeros((SearchAgents_no, dim))
    for i in range(dim):  # 形成SearchAgents_no*30个数[-100，100)以内
        init_Positions[:SearchAgents_no, i] = Positions[:, i]
        for j in range(lenth):
            if init_Positions[j][i]>=(lb + ub)/2:
                init_Positions[SearchAgents_no + j][i] = (lb + ub)/2 - (init_Positions[j][i]-(lb + ub)/2)
            else:
                init_Positions[SearchAgents_no + j][i] = (lb + ub) / 2 + ((lb + ub) / 2-init_Positions[j][i] )
    init_Positions=numpy.array(sorted(init_Positions,key=lambda x:objf(x)))
    for i in range(SearchAgents_no):
        Positions[i, :] = init_Positions[i,:]
    return Positions

def VGWO(Positions,objf, lb, ub, dim, SearchAgents_no, Max_iter,k,Tmax,Tmin,t,begin_time):
    #F = 0.6  # 缩放因子
    tim=time()-begin_time
    T = Tmax
    T0 = Tmax
    K = k
    EPS = Tmin
    ccc = []
    ddd = []
    Positions=re_gene(Positions,objf,SearchAgents_no)
    Convergence_curve_1 = []
    best_ans=objf(Positions[0])
    #迭代寻优
    l=0
    while T > EPS and tim<t:
        # Positions=numpy.array(sorted(Positions,key=lambda x:objf(x)))
        # Alpha_score=objf(Positions[0])
        Alpha_pos =Positions[0]
        Beta_pos =Positions[1]
        Delta_pos =Positions[2]
        # 以上的循环里，Alpha、Beta、Delta
        Gamma=math.log(EPS/T0,K)
        a = 2 - l * ((2) / Gamma)  #   a从2线性减少到0
        # a = 2-(math.log(1+1.3*math.tan((l/Gamma)**3),2))**6
        # print(a)
        for i in range(0, SearchAgents_no):
            r1=np.random.rand(dim)
            r2=np.random.rand(dim)
            A1 = 2 * a * r1 - a  # (-a.a)
            C1 = 2 * r2  # (0,2)
            D_alpha = np.abs(
                C1 * Alpha_pos - Positions[i])  # abs() 函数返回数字的绝对值。Alpha_pos[j]表示Alpha位置，Positions[i,j])候选灰狼所在位置
            X1 = Alpha_pos - A1 * D_alpha  # X1表示根据alpha得出的下一代灰狼位置向量

            r3=np.random.rand(dim)
            r4=np.random.rand(dim)
            A2 = 2 * a * r3 - a
            C2 = 2 * r4
            D_beta = np.abs(C2 * Beta_pos - Positions[i])
            X2 = Beta_pos - A2 * D_beta

            r5=np.random.rand(dim)
            r6=np.random.rand(dim)
            A3 = 2 * a * r5 - a
            C3 = 2 * r6
            D_delta = np.abs(C3 * Delta_pos - Positions[i])
            X3 = Delta_pos - A3 * D_delta

            temp = (2 / 3) * (l / Gamma) * X1 + (1 / 3) * X2 + ((2 / 3) - (2 / 3) * (l / Gamma)) * X3
            temp[temp < lb] = lb
            temp[temp > ub] = ub
            Positions[i] = temp

        #差分进化算子
        r1 = random.randint(0, SearchAgents_no - 1)
        r2 = random.randint(0, SearchAgents_no - 1)
        while r2 == r1:
            r2 = random.randint(0, SearchAgents_no - 1)
        r3 = random.randint(0, SearchAgents_no - 1)
        while r3 == r2 | r3 == r1:
            r3 = random.randint(0, SearchAgents_no - 1)
        # 在DE中常见的差分策略是随机选取种群中的两个不同的个体，将其向量差缩放后与待变异个体进行向量合成
        # F为缩放因子F越小，算法对局部的搜索能力更好，F越大算法越能跳出局部极小点，但是收敛速度会变慢。此外，F还影响种群的多样性。
        F=1.2*(T0-T)/(T0-EPS)
        v_list = add(Positions[r1], multiply(F, substract(Positions[r2], Positions[r3],lb,ub),lb,ub),lb,ub)
        v_list_ans = objf(numpy.array(v_list))
        # Positions=numpy.array(sorted(Positions, key=lambda x: objf(x)))
        ant = 0
        lab=[]
        for key in range(SearchAgents_no):
            jf=objf(Positions[key])
            lab.append(jf)
            if jf > v_list_ans:
                ant += 1
        lab=np.array(lab)
        arrIndex = np.array(lab).argsort()
        Positions = Positions[arrIndex]
        p=1/(1+math.exp((ant/SearchAgents_no)*T))#退火算子
        # p = math.exp(((ant / SearchAgents_no) - 1))
        T*=K
        l+=1
        if random.random() <= p:
            for j in range(dim):
                Positions[len(Positions)-1,j]=v_list[j]
        re_lenth = SearchAgents_no-SearchAgents_no * (Gamma - l) / Gamma  # re_lenth从pop_size线性减小到1
        # print(re_lenth)
        if re_lenth < 1:
            re_lenth = 1
        Positions=re_gene(Positions,objf,math.ceil(re_lenth))
        Alpha_score = objf(Positions[0])
        if Alpha_score<best_ans:
            best_ans=Alpha_score
        Convergence_curve_1.append(best_ans)

        tim=time()-begin_time
    return Convergence_curve_1


def GWO(Positions,objf, lb, ub, dim, SearchAgents_no, Max_iter):
    Convergence_curve_2 = []
    #迭代寻优
    for l in range(0, Max_iter):  # 迭代1000
        Positions = numpy.array(sorted(Positions, key=lambda x: objf(x)))
        Alpha_score = objf(Positions[0])
        Alpha_pos = list(Positions[0])
        Beta_pos = list(Positions[1])
        Delta_pos = list(Positions[2])
        a = 2 - l * ((2) / Max_iter);  #   a从2线性减少到0

        for i in range(0, SearchAgents_no):
            for j in range(0, dim):
                r1 = random.random()  # r1 is a random number in [0,1]主要生成一个0-1的随机浮点数。
                r2 = random.random()  # r2 is a random number in [0,1]

                A1 = 2 * a * r1 - a;  #  (-a.a)
                C1 = 2 * r2;  #  (0.2)
                # D_alpha表示候选狼与Alpha狼的距离
                D_alpha = abs(C1 * Alpha_pos[j] - Positions[i, j]);  # abs() 函数返回数字的绝对值。Alpha_pos[j]表示Alpha位置，Positions[i,j])候选灰狼所在位置
                X1 = Alpha_pos[j] - A1 * D_alpha;  # X1表示根据alpha得出的下一代灰狼位置向量
                #if i==0 and l==0:
                    #print(f'A1:{A1};C1:{C1};D_alpha:{D_alpha};X1:{X1}')

                r1 = random.random()
                r2 = random.random()

                A2 = 2 * a * r1 - a;  #
                C2 = 2 * r2;

                D_beta = abs(C2 * Beta_pos[j] - Positions[i, j]);
                X2 = Beta_pos[j] - A2 * D_beta;
                #if i==0 and l==0:
                    #print(f'A2:{A2};C2:{C2};D_beta:{D_beta};X2:{X2}')

                r1 = random.random()
                r2 = random.random()

                A3 = 2 * a * r1 - a;
                C3 = 2 * r2;

                D_delta = abs(C3 * Delta_pos[j] - Positions[i, j]);
                X3 = Delta_pos[j] - A3 * D_delta;
                temp = (X1 + X2 + X3) / 3
                if temp < lb:
                    temp = (ub + lb) / 2 - (-temp) % ((ub - lb) / 2)
                elif temp > ub:
                    temp = (ub + lb) / 2 + (temp) % ((ub - lb) / 2)
                Positions[i, j] = temp
                # 候选狼的位置更新为根据Alpha、Beta、Delta得出的下一代灰狼地址。

        Convergence_curve_2.append(Alpha_score)
    return Convergence_curve_2

def CMAGWO(Positions,objf, lb, ub, dim, SearchAgents_no, Max_iter):
    Convergence_curve_2 = []
    Positions = numpy.array(sorted(Positions, key=lambda x: objf(x)))
    Alpha_score = objf(Positions[0])
    Alpha_pos = list(Positions[0])
    Beta_pos = list(Positions[1])
    Delta_pos = list(Positions[2])
    #迭代寻优
    for l in range(0, Max_iter//2):  # 迭代1000
        a = 2 - l * ((2) / Max_iter//2);  #   a从2线性减少到0

        for i in range(0, SearchAgents_no):
            for j in range(0, dim):
                r1 = random.random()  # r1 is a random number in [0,1]主要生成一个0-1的随机浮点数。
                r2 = random.random()  # r2 is a random number in [0,1]

                A1 = 2 * a * r1 - a;  #  (-a.a)
                C1 = 2 * r2;  #  (0.2)
                # D_alpha表示候选狼与Alpha狼的距离
                D_alpha = abs(C1 * Alpha_pos[j] - Positions[i, j]);  # abs() 函数返回数字的绝对值。Alpha_pos[j]表示Alpha位置，Positions[i,j])候选灰狼所在位置
                X1 = Alpha_pos[j] - A1 * D_alpha;  # X1表示根据alpha得出的下一代灰狼位置向量
                #if i==0 and l==0:
                    #print(f'A1:{A1};C1:{C1};D_alpha:{D_alpha};X1:{X1}')

                r1 = random.random()
                r2 = random.random()

                A2 = 2 * a * r1 - a;  #
                C2 = 2 * r2;

                D_beta = abs(C2 * Beta_pos[j] - Positions[i, j]);
                X2 = Beta_pos[j] - A2 * D_beta;
                #if i==0 and l==0:
                    #print(f'A2:{A2};C2:{C2};D_beta:{D_beta};X2:{X2}')

                r1 = random.random()
                r2 = random.random()

                A3 = 2 * a * r1 - a;
                C3 = 2 * r2;

                D_delta = abs(C3 * Delta_pos[j] - Positions[i, j]);
                X3 = Delta_pos[j] - A3 * D_delta;
                temp = (X1 + X2 + X3) / 3
                if temp < lb:
                    temp = (ub + lb) / 2 - (-temp) % ((ub - lb) / 2)
                elif temp > ub:
                    temp = (ub + lb) / 2 + (temp) % ((ub - lb) / 2)
                Positions[i, j] = temp
                # 候选狼的位置更新为根据Alpha、Beta、Delta得出的下一代灰狼地址。
        Positions = numpy.array(sorted(Positions, key=lambda x: objf(x)))
        Alpha_score = objf(Positions[0])
        Alpha_pos = list(Positions[0])
        Beta_pos = list(Positions[1])
        Delta_pos = list(Positions[2])
        Convergence_curve_2.append(Alpha_score)

#CMA-ES
    # lambd=4+math.floor(3*math.log(dim))#种群大小
    mu=3#精英种群大小
    weights=[1,0.1,0.01]#精英种群权重
    weights=np.array(weights)
    weights=weights/numpy.sum(weights)#精英集中成员的权重归一化Normalize recombination weights array
    mueff=(numpy.sum(weights))**2/numpy.sum(weights**2)#精英集中成员权重的方差-有效性Variance-effectiveness of sum w_i x_i
    pc=numpy.zeros(dim)#协方差的演化路径Evolution paths for C
    ps = numpy.zeros(dim)#sigma步长的演化路径p_sigma
    cc=(4+mueff/dim)/(dim+4+2*mueff/dim)#协方差路径的系数，cc为alph_cp,Time constant for cumulation for C
    cs = (mueff + 2) / (dim + mueff + 5)#alpa_sigma，t-const for cumulation for sigma control
    c1 = 2 / ((dim + 1.3) ** 2 + mueff)#alpha_c1，Learning rate for rank-one update of C
    cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2) ** 2 + mueff))#alpha_c_lambda，and for rank-mu update
    damps = 1 + 2 * max(0.0, math.sqrt((mueff - 1) / (dim + 1)) - 1) + cs#d_sigma，Damping for sigma
    chiN = dim ** 0.5 * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))#高斯分布二范数的期望Expectation of ||N(0,I)|| == norm(randn(N,1))


    mean_alpha=Positions[0]
    mean_beta=Positions[1]
    mean_delta=Positions[2]
    xmean = (mean_alpha + mean_beta + mean_delta) / 3
    CM=np.cov(Positions,rowvar=False)
    ave_x=Positions.mean(axis=0)
    sum_d=0
    for i in range(SearchAgents_no):
        sum_d+=numpy.linalg.norm(Positions[i] - ave_x)
    theta_alpha=numpy.linalg.norm(mean_alpha - ave_x)/sum_d
    theta_beta=numpy.linalg.norm(mean_beta - ave_x)/sum_d
    theta_delta=numpy.linalg.norm(mean_delta - ave_x)/sum_d

    eigeneval=0
    l=0
    while l< Max_iter//2:
        # print(CM)
        # print("***")
        sigma = (theta_alpha + theta_beta + theta_delta) / 3
        # 协方差矩阵CM的特征值矩阵和特征向量矩阵(正交基矩阵)eigen_vals, eigen_vecs
        eigen_vals, eigen_vecs = np.linalg.eigh(CM)
        eigen_vals=np.sqrt(eigen_vals)
        # print(eigen_vals, eigen_vecs)
        invsqrtC = np.dot(np.dot(np.transpose(eigen_vecs) , numpy.diag(1/eigen_vals)) , eigen_vecs)
        for i in range(0, SearchAgents_no):
            X1=mean_alpha+theta_alpha*np.dot(eigen_vals*numpy.random.randn(dim),eigen_vecs)
            X2 = mean_beta + theta_beta * 0.1*np.dot(eigen_vals*numpy.random.randn(dim),eigen_vecs)
            X3 = mean_delta + theta_delta * 0.01*np.dot(eigen_vals*numpy.random.randn(dim),eigen_vecs)

            temp = (X1 + X2 + X3) / 3
            for j in range(0, dim):
                if temp[j] < lb:
                    temp[j] = (ub + lb) / 2 - (-temp[j]) % ((ub - lb) / 2)
                elif temp[j] > ub:
                    temp[j] = (ub + lb) / 2 + (temp[j]) % ((ub - lb) / 2)
            Positions[i] = temp
            l+=1
        Positions = numpy.array(sorted(Positions, key=lambda x: objf(x)))

        # xmean=weights*Positions[0:mu]
        mean_alpha = Positions[0]
        mean_beta = Positions[1]
        mean_delta = Positions[2]

        #mean_alpha
        xold=xmean
        # print(f'11{xold}')
        xmean=np.dot(weights,Positions[0:mu])
        # print(f'22{xmean}')
        # print(f'33{invsqrtC}')
        #更新进化路径，与理论不一致没有用精英集大小mu，mueff与mu有关
        ps = (1 - cs) * ps + np.dot((xmean - xold),math.sqrt(cs * (2 - cs) * mueff) * invsqrtC) / sigma
        # print("!")
        # print(np.linalg.norm(ps))
        hsig = np.linalg.norm(ps) / math.sqrt(1 - (1 - cs) ** (2 * l /SearchAgents_no)) / chiN <1.4 + 2 / (dim + 1)
        # print("@")
        # print(hsig)
        #cc为alph_cp。hsig为记录的符号信息
        pc = (1 - cc) * pc + hsig * math.sqrt(cc * (2 - cc) * mueff) * (xmean - xold) / sigma
        # print("#")
        # print(pc)
        artmp = (1 / sigma) * (Positions[0:mu]-np.tile(xold, (mu, 1)))
        # print("$")
        # print(artmp)
        CM = (1 - c1 - cmu) * CM+ c1 * (np.dot(np.transpose(pc),pc)+ (1-hsig) * cc*(2-cc) * CM)+ \
             np.dot(np.dot(cmu * np.transpose(artmp) , numpy.diag(weights)) , artmp)

        # print((cs/damps)*(np.linalg.norm(ps)/chiN - 1))
        theta_alpha=theta_alpha* math.exp((cs/damps)*(np.linalg.norm(ps)/chiN - 1))
        theta_beta = theta_beta * math.exp((cs / damps) * (np.linalg.norm(ps) / chiN - 1))
        theta_delta = theta_delta * math.exp((cs / damps) * (np.linalg.norm(ps) / chiN - 1))
        if l-eigeneval>SearchAgents_no/(c1+cmu)/dim/10:
            eigeneval=l
            CM=numpy.triu(CM)+np.transpose(numpy.triu(CM,1))

        Alpha_score = objf(Positions[0])
        Convergence_curve_2.append(Alpha_score)
    return Convergence_curve_2

def PSO_GWO(Positions,objf, lb, ub, dim, SearchAgents_no, Max_iter):
    c1=2.05
    c2=2.05
    Convergence_curve_2 = []
    #迭代寻优
    Positions = numpy.array(sorted(Positions, key=lambda x: objf(x)))
    pbest = Positions
    Alpha_pos = list(Positions[0])
    Beta_pos = list(Positions[1])
    Delta_pos = list(Positions[2])
    for l in range(0, Max_iter):  # 迭代1000
        Alpha_score = objf(numpy.array(Alpha_pos))
        a = 0 - (0-2)*(l/Max_iter)**2  #   a从2线性减少到0

        for i in range(0, SearchAgents_no):
            for j in range(0, dim):
                r1 = random.random()  # r1 is a random number in [0,1]主要生成一个0-1的随机浮点数。
                r2 = random.random()  # r2 is a random number in [0,1]

                A1 = 2 * a * r1 - a;  #  (-a.a)
                C1 = 2 * r2;  #  (0.2)
                # D_alpha表示候选狼与Alpha狼的距离
                D_alpha = abs(C1 * Alpha_pos[j] - Positions[i, j]);  # abs() 函数返回数字的绝对值。Alpha_pos[j]表示Alpha位置，Positions[i,j])候选灰狼所在位置
                X1 = Alpha_pos[j] - A1 * D_alpha;  # X1表示根据alpha得出的下一代灰狼位置向量
                #if i==0 and l==0:
                    #print(f'A1:{A1};C1:{C1};D_alpha:{D_alpha};X1:{X1}')

                r1 = random.random()
                r2 = random.random()

                A2 = 2 * a * r1 - a;  #
                C2 = 2 * r2;

                D_beta = abs(C2 * Beta_pos[j] - Positions[i, j]);
                X2 = Beta_pos[j] - A2 * D_beta;
                #if i==0 and l==0:
                    #print(f'A2:{A2};C2:{C2};D_beta:{D_beta};X2:{X2}')

                r1 = random.random()
                r2 = random.random()

                A3 = 2 * a * r1 - a;
                C3 = 2 * r2;

                D_delta = abs(C3 * Delta_pos[j] - Positions[i, j]);
                X3 = Delta_pos[j] - A3 * D_delta;

                w1=X1/(X1 + X2 + X3)
                w2 = X2 / (X1 + X2 + X3)
                w3 = X3 / (X1 + X2 + X3)
                r1 = random.random()
                r2 = random.random()
                temp = c1*r1*(w1*X1+w2*X2+w3*X3)+c2*r2*(pbest[i,j]-Positions[i, j])
                # print(temp)
                # temp = (X1 + X2 + X3) / 3
                if temp < lb:
                    temp = (ub + lb) / 2 - (-temp) % ((ub - lb) / 2)
                elif temp > ub:
                    temp = (ub + lb) / 2 + (temp) % ((ub - lb) / 2)
                Positions[i, j] = temp
                # 候选狼的位置更新为根据Alpha、Beta、Delta得出的下一代灰狼地址。
            ans=objf(Positions[i])
            if ans<objf(numpy.array(Alpha_pos)):
                Delta_pos=Beta_pos
                Beta_pos=Alpha_pos
                Alpha_pos=list(Positions[i])
            elif ans<objf(numpy.array(Beta_pos)):
                Delta_pos=Beta_pos
                Beta_pos=list(Positions[i])
            elif ans<objf(numpy.array(Delta_pos)):
                Delta_pos = list(Positions[i])

            if ans<objf(pbest[i]):
                pbest[i]=Positions[i]

        Convergence_curve_2.append(Alpha_score)
    Convergence_curve_2.append(objf(numpy.array(Alpha_pos)))
    return Convergence_curve_2

def vw_GWO(Positions,objf, lb, ub, dim, SearchAgents_no, Max_iter):
    Convergence_curve_2 = []
    #迭代寻优
    for l in range(0, Max_iter):  # 迭代1000
        Positions = numpy.array(sorted(Positions, key=lambda x: objf(x)))
        Alpha_score = objf(Positions[0])
        Alpha_pos = list(Positions[0])
        Beta_pos = list(Positions[1])
        Delta_pos = list(Positions[2])
        # a = 2 - l * ((2) / Max_iter);  #   a从2线性减少到0
        a = 1.6*math.exp(-l/Max_iter)
        for i in range(0, SearchAgents_no):
            for j in range(0, dim):
                r1 = random.random()  # r1 is a random number in [0,1]主要生成一个0-1的随机浮点数。
                r2 = random.random()  # r2 is a random number in [0,1]

                A1 = 2 * a * r1 - a;  #  (-a.a)
                C1 = 2 * r2;  #  (0.2)
                # D_alpha表示候选狼与Alpha狼的距离
                D_alpha = abs(C1 * Alpha_pos[j] - Positions[i, j]);  # abs() 函数返回数字的绝对值。Alpha_pos[j]表示Alpha位置，Positions[i,j])候选灰狼所在位置
                X1 = Alpha_pos[j] - A1 * D_alpha;  # X1表示根据alpha得出的下一代灰狼位置向量
                #if i==0 and l==0:
                    #print(f'A1:{A1};C1:{C1};D_alpha:{D_alpha};X1:{X1}')

                r1 = random.random()
                r2 = random.random()

                A2 = 2 * a * r1 - a;  #
                C2 = 2 * r2;

                D_beta = abs(C2 * Beta_pos[j] - Positions[i, j]);
                X2 = Beta_pos[j] - A2 * D_beta;
                #if i==0 and l==0:
                    #print(f'A2:{A2};C2:{C2};D_beta:{D_beta};X2:{X2}')

                r1 = random.random()
                r2 = random.random()

                A3 = 2 * a * r1 - a;
                C3 = 2 * r2;

                D_delta = abs(C3 * Delta_pos[j] - Positions[i, j]);
                X3 = Delta_pos[j] - A3 * D_delta;
                #if i==0 and l==0:
                    #print(f'A3:{A3};C3:{C3};D_delta:{D_delta};X3:{X3}')
                fai=0.5*math.atan(l)
                ceita=(2/math.pi)*math.acos(1/3)*math.atan(l)
                w1=math.cos(ceita)
                w2=0.5*math.sin(ceita)*math.cos(fai)
                temp = w1*X1 + w2*X2 + (1-w1-w2)*X3
                if temp < lb:
                    temp = (ub+lb)/2-(-temp)%((ub-lb)/2)
                elif temp > ub:
                    temp = (ub+lb)/2+(temp)%((ub-lb)/2)
                Positions[i, j] = temp
                #候选狼的位置更新为根据Alpha、Beta、Delta得出的下一代灰狼地址。

        Convergence_curve_2.append(Alpha_score)
    return Convergence_curve_2

#适应度函数

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
    s=-c+3.32236
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
    s=-c+10.5319
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
    s=-c+10.1499
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


tracemalloc.start()
#主程序

tn = 5
func_details = [[F1, -100, 100, 30],[F2, -100, 100, 30],[F3, -100, 100, 30],
                [F4, -100, 100, 30],[F5, -100, 100, 30],[F6, -100, 100, 30],
                [F7, -32, 32, 30],[F8, -100, 100, 30],[F9, -100, 100, 30],
                [F10, -50, 50, 30],[F11, -5.12, 5.12, 30],[F12, -50, 50, 30],
                [F13, -100, 100, 30],[F14, -30, 30, 30],[F15, 0, 1, 6],
                [F16, 0, 10, 4],[F17, 0, 10, 4],[F18, -100, 100, 2],
                [F19, -10, 10, 30]]
stla=[7.941861248,7.947866774,18.24994609,6.933228576,6.467637587,7.095196462,7.540277255,
7.661875534,6.720435369,12.41295357,9.060303926,14.94295622,8.294937444,11.2097746,
3.669719827,4.950158608,2.916625667,0.632087612,9.120521414]
orthogonal=[[0.99,2.5,0.1],[0.99,2,0.001],[0.99,1.5,0.00001],[0.99,1,0.01],[0.99,0.5,0.0001],
            [0.96,2.5,0.00001],[0.96,2,0.01],[0.96,1.5,0.0001],[0.96,1,0.1],[0.96,0.5,0.001],
            [0.93,2.5,0.0001],[0.93,2,0.1],[0.93,1.5,0.001],[0.93,1,0.00001],[0.93,0.5,0.01],
            [0.9,2.5,0.001],[0.9,2,0.00001],[0.9,1.5,0.01],[0.9,1,0.0001],[0.9,0.5,0.1],
            [0.87,2.5,0.01],[0.87,2,0.0001],[0.87,1.5,0.1],[0.87,1,0.001],[0.87,0.5,0.00001]]
for i in orthogonal:
    ARV=[]
    print(i)
    for j in range(tn):
        print(j)
        arv=[]
        for fun in range(len(func_details)):
            Fx = func_details[fun][0]
            # output_path = 'out1.txt'
            # with open(output_path, 'a', encoding='utf-8') as file1:
            #     print(f'F{fun + 1}',end=" ",file=file1)
            print(f'F{fun+1}')
            Max_iter = 300#迭代次数
            lb = func_details[fun][1]#下界-10000000000000000000000
            ub = func_details[fun][2]#上届10000000000000000000000
            dim = func_details[fun][3]#狼的寻值范围30
            SearchAgents_no = 50#寻值的狼的数量
            positions=init()

            # X=[]
            # for i in range(tn):
            # positions_1=positions.copy()
            positions_1=beta_init()
            k=i[0]
            Tmax=i[1]
            Tmin=i[2]
            begin_time = time()
            x = VGWO(positions_1,Fx, lb, ub, dim, SearchAgents_no, Max_iter,k,Tmax,Tmin,stla[fun],begin_time)#改进灰狼
            # X.append(x[len(x)-1])
            end_time = time()
            run_time = end_time - begin_time
            # print(run_time)
            print(x[len(x)-1])
            arv.append(x[len(x)-1])
        ARV.append(arv)
    print(ARV)
    output_path = 'result of Calculating ARV.txt'
    with open(output_path, 'a', encoding='utf-8') as file1:
        print(ARV, file=file1)

out1=[
[0.001784452296575978, 3.048202563931833e-263, 0.0, 1.5381137529299796e-224, 0.0, 4.4862280870861895e-263, 4.440892098500626e-16, 0.0, 0.0, 3.4964269101975733, 0.0, 0.3358821047055178, 3.7586565434103107, 28.82981117684572, 0.34736807658869573, 5.4034267014274775, 5.094712546991357, 0.0, 0.0],
[5.745067860851716e-05, 3.9923294200134125e-256, 0.0, 2.881713480805181e-221, 0.0, 3.0629790829986997e-264, 4.440892098500626e-16, 0.0, 0.0, 4.466920146136625, 0.0, 0.3337982712216656, 4.126698569993611, 27.204120767640582, 0.43243194835663123, 8.837564129242693, 5.0947714989710775, 0.0, 0.0],
[0.0006391676381324318, 2.600687358888956e-262, 0.0, 1.4189539581999398e-225, 0.0, 2.8403687772384836e-263, 4.440892098500626e-16, 0.0, 0.0, 3.3044834362098707, 0.0, 0.44573586075874355, 3.50439339804331, 28.030020548002476, 0.4068752638219584, -0.0024395126345950757, 5.094847022304951, 0.0, 0.0],
[0.0014093405693751393, 3.4176710485648296e-260, 0.0, 5.935417304060885e-221, 0.0, 3.212012068493485e-263, 4.440892098500626e-16, 0.0, 0.0, 3.091976421390469, 0.0, 0.27070010608339157, 3.2430220507686545, 28.69172363235338, 0.7051999540504301, 5.403423213480642, 5.0947079781230045, 0.0, 0.0],
[0.00012582581002062998, 3.842875349315308e-266, 0.0, 1.0800588335769368e-221, 0.0, 7.215498153909858e-262, 4.440892098500626e-16, 0.0, 0.0, 4.117785738046057, 0.0, 0.40437661096441446, 3.7275453931775164, 28.610768050678523, 0.2098676550557168, -0.004408056907678315, -0.0031454267884694076, 0.0, 0.0],
[0.003616008475867627, 0.0, 0.0, 0.0, 0.0, 0.0, 4.440892098500626e-16, 0.0, 0.0, 3.833484841346804, 0.0, 0.49934019724869166, 4.030844217144502, 28.046006967599688, 0.4189107644078294, -0.004327672717129971, 5.094710046322086, 0.0, 0.0],
[0.00010739116884828093, 0.0, 0.0, 0.0, 0.0, 0.0, 4.440892098500626e-16, 0.0, 0.0, 3.939727307353166, 0.0, 0.5543881485906078, 4.012061993955922, 27.75285955978494, 0.24341960122884432, 0.8638990427581046, 5.09470937145546, 0.0, 0.0],
[0.0011282962110349404, 0.0, 0.0, 0.0, 0.0, 0.0, 4.440892098500626e-16, 0.0, 0.0, 4.038959081406636, 0.0, 0.45526066221519207, 3.84416987926143, 28.678738905695546, 0.762831284032595, 5.403480066448853, 5.0947044035483815, 0.0, 0.0],
[0.0011596081425771785, 0.0, 0.0, 0.0, 0.0, 0.0, 4.440892098500626e-16, 0.0, 0.0, 3.703771374211335, 0.0, 0.31383189868685607, 3.7500107904747564, 28.701696357942325, 0.1488905449590865, -0.004396508527845455, 5.09474329088885, 0.0, 0.0],
[0.0015464274749396045, 0.0, 0.0, 0.0, 0.0, 0.0, 4.440892098500626e-16, 0.0, 0.0, 3.9242002503342626, 0.0, 0.486823225426831, 4.0000256144564625, 28.073145585641882, 0.3156509470966409, 5.403430130623198, 5.094703437527134, 0.0, 0.0],
[8.658416287388171e-05, 2e-323, 0.0, 5e-324, 0.0, 1.5e-323, 4.440892098500626e-16, 0.0, 0.0, 3.9831835437900596, 0.0, 0.4437810430354105, 3.9174692398793294, 28.084267857445706, 0.28514042855120136, -0.003457280959032971, -0.0032553332681608538, 0.0, 0.0],
[0.0019331178811305971, 2.5e-323, 0.0, 1e-323, 0.0, 2.5e-323, 4.440892098500626e-16, 0.0, 0.0, 4.075653629067225, 0.0, 0.4290228241672928, 4.500010694479182, 28.676452261877625, 0.28807528440968566, 1.0307210845774328, -0.002530726962648444, 0.0, 0.0],
[2.298291697477861e-05, 1.5e-323, 0.0, 5e-324, 0.0, 3.5e-323, 4.440892098500626e-16, 0.0, 0.0, 3.864199963993683, 0.0, 0.4987502102370889, 4.11295840025559, 28.687021464162747, 1.0746236698818588, -0.004333591364492406, 5.094709976971826, 0.0, 0.0],
[0.0008602597935317702, 2e-323, 0.0, 5e-324, 0.0, 2.5e-323, 4.440892098500626e-16, 0.0, 0.0, 4.027094763882786, 0.0, 0.4123044742686323, 4.676661971401279, 28.681083079691597, 0.4516199729010939, 5.403449070913983, 5.094709104978317, 0.0, 0.0],
[0.0012823366853365048, 3e-323, 0.0, 5e-324, 0.0, 1.5e-323, 4.440892098500626e-16, 0.0, 0.0, 4.319244362343794, 0.0, 0.34920934349089067, 3.3743768482233825, 28.874379951180142, 0.24993804245461915, 5.403426660823161, 5.094704919481205, 0.0, 0.0],
[1.0951579625384511e-05, 0.0, 0.0, 6.802503e-318, 0.0, 0.0, 4.440892098500626e-16, 0.0, 0.0, 3.4238926306193025, 0.0, 0.4667134960815824, 3.4963142815663497, 28.849949665547918, 0.25772079115923585, 5.403425997392923, 5.094702866478905, 0.0, 0.0],
[0.0002373581335185504, 0.0, 0.0, 7.19686532e-315, 0.0, 0.0, 4.440892098500626e-16, 0.0, 0.0, 3.215216694460803, 0.0, 0.30785975806304006, 3.505049817444161, 28.68153604052622, 0.6479155364575657, 5.40342132198836, 5.09470470598862, 0.0, 0.0],
[0.00022414902044898266, 0.0, 0.0, 3.77e-321, 0.0, 0.0, 4.440892098500626e-16, 0.0, 0.0, 3.6365792143508253, 0.0, 0.4375788202516044, 3.9739778570192836, 28.850791388648243, 0.30110900313747635, 5.403428657803552, -0.0024966732472826436, 0.0, 0.0],
[0.0005044759545409505, 0.0, 0.0, 4.407e-321, 0.0, 0.0, 4.440892098500626e-16, 0.0, 0.0, 3.626694465917341, 0.0, 0.3408675826459338, 3.9983306253161044, 28.084469927382827, 0.7052005774463179, -0.0004246896366613129, 8.411474192538373, 0.0, 0.0],
[0.0014738647402097182, 0.0, 0.0, 0.0, 0.0, 0.0, 4.440892098500626e-16, 0.0, 0.0, 4.0643381225162285, 0.0, 0.36413278725955484, 4.157677907095506, 27.335917126847335, 0.198076685409319, 5.403423151461614, 5.094776149455721, 0.0, 0.0],
[0.0005501902640826506, 8.4e-323, 0.0, 0.0, 0.0, 0.0, 4.440892098500626e-16, 0.0, 0.0, 4.122535968565461, 0.0, 0.4884042689812471, 4.22666111700102, 28.6814186107025, 0.2710440752366421, 5.4034692138253915, 5.0947077467023005, 0.0, 0.0],
[0.0005696450041960954, 9.4e-323, 0.0, 0.0, 0.0, 0.0, 4.440892098500626e-16, 0.0, 0.0, 4.03052426136911, 0.0, 0.41641055028880775, 4.25459586676731, 28.068718722401314, 0.36675652429487293, -0.0042569888120009836, 5.0947141207795665, 0.0, 0.0],
[5.5314401843031824e-05, 1.1e-322, 0.0, 0.0, 0.0, 0.0, 4.440892098500626e-16, 0.0, 0.0, 4.448216097729073, 0.0, 0.34667006442732073, 4.522180058639878, 28.656457962023836, 0.015792248261249586, 5.403453612345337, 5.094706842512742, 0.0, 0.0],
[0.0016261183398194046, 3.5e-323, 0.0, 0.0, 0.0, 0.0, 4.440892098500626e-16, 0.0, 0.0, 3.9396436296290136, 0.0, 0.43917614003687727, 3.9048062876820735, 28.680463641058836, 0.16871004898641795, -0.0039850051224323835, 5.094711059925528, 0.0, 0.0],
[0.001038921286696101, 7e-323, 0.0, 0.0, 0.0, 0.0, 4.440892098500626e-16, 0.0, 0.0, 3.5487079656334375, 0.0, 0.542109467573078, 3.7221776578860903, 27.912412388582872, 0.2464085975274406, 5.403424167870786, 5.0947194589300935, 0.0, 0.0],
[0.002846017507085463, 6.023271931402107e-248, 0.0, 3.5589776409271527e-212, 0.0, 5.979896159910128e-249, 4.440892098500626e-16, 0.0, 0.0, 3.9359301986050435, 0.0, 0.33576052885030433, 3.0035599171798126, 28.819130417650882, 0.25398295164286644, 5.4034633392791, 5.095112814502079, 0.0, 0.0],
[0.0035350594571768035, 3.709514113819938e-250, 0.0, 7.281966388112616e-210, 0.0, 6.283224475174633e-249, 4.440892098500626e-16, 0.0, 0.0, 3.0139261134317366, 0.0, 0.4559507726843418, 3.7534405117608025, 28.85952103024694, 0.34946799687049523, 5.403452518238389, -0.002876280609880766, 0.0, 0.0],
[0.0002418074712573064, 1.2720745842527314e-249, 0.0, 6.26880357353789e-212, 0.0, 4.154381754951754e-248, 4.440892098500626e-16, 0.0, 0.0, 3.497017202346397, 0.0, 0.2669626459316487, 3.244848705494911, 28.640432602739576, 0.3067267268793623, 5.404098549807419, 5.0947553448202765, 0.0, 0.0],
[0.0009704042135655344, 1.0462834642316249e-246, 0.0, 5.623469137095443e-214, 0.0, 1.279330593579348e-247, 4.440892098500626e-16, 0.0, 0.0, 3.54450184510701, 0.0, 0.22785057639171355, 3.9936703402614837, 28.077820112937502, 0.43282522230478904, 5.403442100360422, 5.094712111542143, 0.0, 0.0],
[0.0010624287847015301, 3.7868994373239435e-247, 0.0, 6.525823719745201e-210, 0.0, 4.658091625255888e-246, 4.440892098500626e-16, 0.0, 0.0, 3.038880952684478, 0.0, 0.19781290237073829, 3.7455922963232218, 28.63129161188364, 0.5461635150632573, 5.403430863747143, 0.012349401488984668, 0.0, 0.0],
[0.00853280730923911, 6.494288943763214e-104, 4.388542998001554e-167, 1.348643181235583e-90, 0.0, 0.0, 4.440892098500626e-16, 0.0, 0.0, 3.5253914164462077, 0.0, 0.3381693596968721, 2.701965328350269, 28.627527271221574, 0.2552854972108509, -0.003283523893161444, 9.267912679765189, 0.0, 0.0],
[0.0007061581409891771, 0.0, 8.877904987962357e-164, 3.941587181813572e-92, 0.0, 0.0, 4.440892098500626e-16, 0.0, 0.0, 2.3056253091170755, 0.0, 0.33362484443141716, 3.199906022427483, 27.17107096724855, 0.36329686152688234, 5.1988136873922075, 5.094715679361006, 0.0, 0.0],
[0.0019132298209326715, 7.256907136498636e-106, 2.970202157534818e-160, 2.792067064995022e-88, 0.0, 0.0, 4.440892098500626e-16, 0.0, 0.0, 2.6551777604950946, 0.0, 0.2700949383880358, 2.527141314387078, 27.403851713519, 0.2924278966057132, 0.0074897305748340415, 5.094784327357494, 0.0, 0.0],
[0.0013951743661465565, 1.5157678535006588e-103, 4.176419551487725e-164, 2.1744158645568813e-88, 0.0, 0.0, 4.440892098500626e-16, 0.0, 0.0, 3.2271442335249207, 0.0, 0.18527727762340643, 3.291177028913865, 28.606801059365605, 0.6247351478568284, 5.403449132909725, 0.03742824653678056, 0.0, 0.0],
[0.0003995277702185054, 2.44307349680733e-104, 3.695979334321893e-162, 9.615084870477373e-88, 0.0, 5.5570989597774156e-105, 4.440892098500626e-16, 0.0, 0.0, 2.965723869102919, 0.0, 0.24312612756021462, 2.541162314908736, 27.192906059537503, 0.20745415095447184, 0.00966137836592651, 5.094870292317879, 0.0, 0.0],
[0.0015026269284387217, 1.5381734319693625e-191, 1.8830866671029797e-308, 8.502963218224098e-159, 0.0, 2.3422055430066635e-191, 4.440892098500626e-16, 0.0, 0.0, 2.8283707194916228, 0.0, 0.22722098670920107, 2.762653728752695, 27.837077403514776, 0.25931749599592946, 5.403484309030466, 0.0030331104301080103, 0.0, 0.0],
[0.004298036811693717, 1.0618598553621823e-190, 7.324361521064086e-304, 2.774850665390225e-163, 0.0, 4.850654530165434e-193, 4.440892098500626e-16, 0.0, 0.0, 3.3241435399978645, 0.0, 0.2277041798476641, 2.5133390368490436, 28.641072972891678, 0.3355059470269057, -0.0007186637568885601, 5.094763099803315, 0.0, 0.0],
[0.002632638147904043, 1.2965740482682552e-192, 6.754750930451452e-300, 2.4792546153004916e-167, 0.0, 1.4841624826354048e-191, 4.440892098500626e-16, 0.0, 0.0, 3.682638340278398, 0.0, 0.2549133540873405, 2.8179725645577407, 28.053000382924342, 0.47744350315094186, -0.003105572141398838, 5.094857980121407, 0.0, 0.0],
[0.0035215476379755195, 6.075956400844939e-191, 4.712783918483038e-299, 9.375517297849212e-163, 0.0, 3.8638692792293635e-195, 4.440892098500626e-16, 0.0, 0.0, 3.0249819367226753, 0.0, 0.2972580194742938, 4.0056859239077, 27.188995016826233, 0.24045980614768903, 3.1291555874237664, 9.338684003912672, 0.0, 0.0],
[0.00042924698624158264, 6.846841052173146e-194, 3.7644747859139097e-303, 5.005921784794084e-166, 0.0, 4.972800981942125e-195, 4.440892098500626e-16, 0.0, 0.0, 3.0059006701727253, 0.0, 0.7968045326953973, 2.8712815920146157, 28.830379240350577, 0.359762348035491, -0.0016834924495743309, 5.094714091594907, 0.0, 0.0],
[0.00410906934694133, 0.0, 8.742248451423057e-69, 1.347122779509625e-37, 0.0, 0.0, 4.440892098500626e-16, 0.0, 0.0, 3.250944135767222, 0.0, 0.34116822334258706, 3.4597323277844296, 28.897809794638665, 0.33371236367852175, 0.43682593028559147, 4.794384239193968, 0.0, 0.0],
[0.0005323632818340363, 0.0, 7.438275763047918e-74, 1.8615582897001877e-37, 0.0, 0.0, 4.440892098500626e-16, 0.0, 0.0, 3.4151789352600086, 0.0, 0.32649776950381176, 3.1032241452141416, 28.704481554276313, 0.668060991239201, 0.3034501157943357, 0.13957908344877978, 0.0, 0.0],
[0.0029911521089173165, 0.0, 3.5447288982493526e-69, 2.4243544940013585e-37, 0.0, 0.0, 4.440892098500626e-16, 0.0, 0.0, 4.001986026111407, 0.0, 0.2930634899304831, 3.9677138302991324, 28.84184021191168, 0.509092466933228, 0.021430629215201336, 0.10099686289056464, 0.0, 0.0],
[0.0004713566559968463, 0.0, 4.031016634657994e-66, 2.2506582063604317e-37, 0.0, 0.0, 4.440892098500626e-16, 0.0, 0.0, 2.7660304335537984, 0.0, 0.18684552419425182, 4.68713781914803, 28.676804249883645, 0.2361563133369673, 5.403908238901623, 0.22429028275524843, 0.0, 0.0],
[0.0004018273806416781, 0.0, 1.3701274654540611e-67, 2.04901431302458e-35, 0.0, 0.0, 4.440892098500626e-16, 0.0, 0.0, 3.434638448415685, 0.0, 0.4414206958642999, 3.8519329056352003, 28.898344265401708, 1.3302858844254404, 0.007210688003944199, 0.012244278341043469, 0.0, 1.1102230246251565e-16],
[0.0005085609976451799, 2.973790676399417e-125, 1.851425398480033e-191, 6.371193408824708e-105, 0.0, 4.062345902119536e-124, 4.440892098500626e-16, 0.0, 6.649469143307985e-117, 3.217612107395812, 0.0, 0.19042934163942377, 3.483502699447373, 27.133442334727285, 0.4974497408820806, 5.403495617719075, 5.094714374235751, 0.0, 0.0],
[0.0016076649521861164, 1.6435570020473288e-121, 6.708692654734707e-197, 8.931633400634932e-102, 0.0, 2.0842996083020096e-122, 4.440892098500626e-16, 0.0, 0.0, 3.1377571930518116, 0.0, 0.2875643530174453, 2.265090853987408, 27.247026544356647, 0.49245299754129146, 4.263571689443762, 5.094733672199769, 0.0, 0.0],
[0.0009104550136294076, 1.2540377995054058e-121, 8.217588261176581e-195, 1.386821997045224e-102, 0.0, 4.876020691056593e-122, 4.440892098500626e-16, 0.0, 0.0, 3.187321201122721, 0.0, 0.18560107304257537, 2.5530391755635655, 27.215000828217605, 0.42267017151963904, 5.4035341581884255, 5.094904435775485, 0.0, 0.0],
[0.0016623287016155341, 1.788471199630606e-122, 1.4185278093037325e-193, 2.440399822733111e-104, 0.0, 3.80533211439366e-127, 4.440892098500626e-16, 0.0, 0.0, 2.1738617695325977, 0.0, 0.3362032859080187, 2.606038955578197, 27.277260146724487, 0.22039261999128446, 9.041729973549003, 5.094793741111968, 0.0, 0.0],
[0.0017463193360391926, 1.0344168338080153e-124, 2.1187517779225535e-195, 2.6608159157084004e-109, 0.0, 4.0693492096560224e-120, 4.440892098500626e-16, 0.0, 0.0, 3.5203607834214807, 0.0, 0.3896297067024592, 2.581771885244345, 28.81606816709367, 0.6477178806206565, -0.002783859419619006, 0.0050626448382029565, 0.0, 0.0],
[0.006400075883278511, 2.8803554173178166e-111, 1.3864138104622465e-176, 4.946489732286267e-97, 0.0, 4.20927574198512e-112, 4.440892098500626e-16, 0.0, 0.0, 3.740916273240169, 0.0, 0.43314631396048714, 3.198546062358898, 28.641044238438617, 0.3044321873545104, -0.0009282162195116683, -0.0008791240438768, 0.0, 0.0],
[0.004554852403486076, 1.4601945891338872e-111, 5.321728189608324e-175, 3.251346291509811e-95, 0.0, 2.3589358175192888e-113, 4.440892098500626e-16, 0.0, 0.0, 3.3977190440222067, 0.0, 0.21661394329229125, 2.972581013177931, 27.204132005331477, 0.6019610698270839, 5.403511818367462, 2.132252069151779, 0.0, 0.0],
[0.001356258070277394, 1.881197413675384e-107, 6.559557906483577e-176, 3.2394686776595373e-96, 0.0, 3.594178618969922e-116, 4.440892098500626e-16, 0.0, 0.0, 3.474814919081507, 0.0, 0.2524328818975375, 2.412723381697516, 27.118025036618693, 0.48538145187809745, -0.003707432846201897, 0.008734781427357063, 0.0, 0.0],
[0.0037767465687335022, 2.4462706634317867e-117, 2.0607104988689513e-177, 2.228636543977622e-94, 0.0, 4.024246077811253e-111, 4.440892098500626e-16, 0.0, 0.0, 3.329693578127513, 0.0, 0.2548842557555554, 2.815449548091966, 28.843043397847744, 0.267995770468882, 5.403560750027909, 0.01452023894661636, 0.0, 0.0],
[0.0039815914062346636, 1.0734565336550684e-112, 1.1092145736341773e-175, 9.183846252148087e-96, 0.0, 1.9649494938253452e-112, 4.440892098500626e-16, 0.0, 0.0, 2.3515827314153763, 0.0, 0.3058321285298489, 3.0392306895670127, 28.016747038740235, 0.016123921323754953, 5.403508564785872, 3.8737086751438845, 0.0, 0.0],
[0.006306004612738669, 0.0, 3.5291061710929926e-49, 5.9408351880635414e-27, 0.0, 0.0, 4.440892098500626e-16, 0.0, 0.0, 4.026819087487253, 0.0, 0.3994864670609889, 3.0172616808123554, 28.096558058613496, 0.5121295547367333, 0.1754185559953978, 6.277765512895673, 0.0, 0.0],
[0.0013056375349430605, 0.0, 1.2601525143718134e-47, 7.646839349286426e-27, 0.0, 0.0, 4.440892098500626e-16, 0.0, 0.0, 3.9968328921782494, 0.0, 0.5016823424618188, 3.9970288941317764, 28.886708148880828, 0.022819923168200695, 3.734646991624512, 0.0936519215590721, 0.0, 0.0],
[0.003910759158659238, 0.0, 1.6093417567344657e-48, 2.067563204151032e-27, 0.0, 0.0, 4.440892098500626e-16, 0.0, 0.0, 4.022520311502848, 0.0, 0.4128649927772366, 3.9113153490231616, 28.648727801815205, 0.5491308676852023, 4.588891158272873, 0.7383835986576326, 0.0, 0.0],
[0.0003843358111235151, 0.0, 2.894958990794722e-48, 1.3781279637193055e-27, 0.0, 0.0, 4.440892098500626e-16, 0.0, 0.0, 5.355631713392593, 0.0, 0.41473183926698526, 3.708204738726815, 29.0, 0.3004182770997059, 0.13503075232621775, 0.08879462188992449, 0.0, 0.0],
[0.003308720691226252, 0.0, 3.8827251824117977e-50, 5.843506003994759e-26, 0.0, 0.0, 4.440892098500626e-16, 0.0, 0.0, 3.2790513463297124, 0.0, 0.45247401566844564, 5.005515101796093, 27.662442917442892, 0.14141781780210438, 0.09129685781839036, 6.1565127732507765, 0.0, 0.0],
[0.001999323730856095, 0.0, 8.139040247474266e-127, 3.5103730581137564e-68, 0.0, 0.0, 4.440892098500626e-16, 0.0, 0.0, 3.471010588024661, 0.0, 0.24414051655176444, 3.713829379448854, 28.82448134430757, 0.14703031749361362, 0.005203066860259398, 5.094710147726301, 0.0, 0.0],
[0.0013567613029581427, 0.0, 6.106269322769168e-127, 8.10959360561028e-68, 0.0, 0.0, 4.440892098500626e-16, 0.0, 0.0, 3.073694151385884, 0.0, 0.17585711889321315, 2.748000460167332, 27.247150884876437, 0.22633288821157027, 3.0831328771877073, 5.09475285085788, 0.0, 0.0],
[0.0006810604784204255, 0.0, 5.719999676616384e-132, 1.2990560088831672e-66, 0.0, 0.0, 4.440892098500626e-16, 0.0, 0.0, 2.425042505011065, 0.0, 0.15431726335762522, 2.4344000365733156, 28.830953508290435, 0.25688180333967425, 0.004644220425621626, 0.11611807372707617, 0.0, 0.0],
[0.0033180231961679185, 0.0, 3.2577381797880997e-131, 1.6354169112593423e-69, 0.0, 0.0, 4.440892098500626e-16, 0.0, 0.0, 2.8021481013454306, 0.0, 0.6962156807445858, 2.8975313027303127, 28.061072007064475, 0.2588078471945998, 7.0956792295931175, 0.0016717807460580048, 0.0, 0.0],
[0.004710983911277555, 0.0, 4.3302272622227683e-125, 4.917985789782079e-69, 0.0, 0.0, 4.440892098500626e-16, 0.0, 0.0, 2.4889232198778473, 0.0, 0.24643585872483495, 2.787663421245402, 28.833423418552417, 0.01651703401527982, 0.0020849380151179986, 9.267911306034783, 0.0, 0.0],
[0.0021514933921936708, 5.727791681164254e-129, 2.4954590863410103e-209, 5.000857808379846e-108, 0.0, 4.802296048616173e-127, 4.440892098500626e-16, 0.0, 0.0, 3.808616702433481, 0.0, 0.7088321914648635, 2.7615306407444518, 28.658811309974308, 0.524934881062455, 0.008929272761708518, 6.442819013477168, 0.0, 0.0],
[0.00047266252115374385, 4.08107845332156e-127, 9.405340031835285e-205, 3.7519209559967455e-111, 0.0, 1.47658513617848e-125, 4.440892098500626e-16, 0.0, 0.0, 3.588403322406192, 0.0, 0.3096694859405007, 2.3809456883496227, 28.823346069125215, 0.12110877360452221, -0.004291248271984571, 8.755929074435269, 0.0, 0.0],
[0.0017026221163578734, 3.167290851662648e-130, 5.409898149484475e-201, 4.5848126966633795e-109, 0.0, 1.133956923859279e-130, 4.440892098500626e-16, 0.0, 0.0, 3.4658832098525347, 0.0, 0.4116689673643814, 2.279235315114528, 28.81398502670448, 0.8480179121677107, 2.5975126924740666, 9.267909206219732, 0.0, 0.0],
[0.0005748198115033132, 8.919874687289892e-128, 4.035126126375532e-205, 4.0247001178739294e-110, 0.0, 3.439878042980433e-128, 4.440892098500626e-16, 0.0, 7.081969084148008e-119, 3.283456634171035, 0.0, 0.34316973951232127, 2.9595086620745708, 28.03177580585702, 0.29521421479021637, 0.007804681855251516, 8.415421434910602, 0.0, 0.0],
[0.004624931534730092, 1.2854653113322631e-129, 2.404243086565838e-205, 3.728007137711184e-110, 0.0, 7.718398666318674e-127, 4.440892098500626e-16, 0.0, 2.4927282352082745e-120, 3.0438933157999264, 0.0, 0.17693714515848794, 3.4969926013775643, 26.9094910934139, 0.3619314224359327, 0.0016545539251762165, 5.949106277278949, 0.0, 0.0],
[0.004692764197162358, 0.0, 6.548375965087478e-67, 9.631608757476891e-37, 0.0, 0.0, 4.440892098500626e-16, 0.0, 0.0, 2.7605590553231876, 0.0, 0.3141552600263798, 3.5747607287669845, 28.63665395664601, 0.1481096113053102, 0.07515565833378446, 0.2568572665435749, 0.0, 0.0],
[0.0036990802766238007, 0.0, 2.2232080530536965e-65, 6.473048334982693e-35, 0.0, 0.0, 4.440892098500626e-16, 0.0, 0.0, 3.4268784842852917, 0.0, 0.43327091481347946, 3.6859776257264505, 28.873939025464683, 0.16430374286387917, 0.04730948993977968, 5.1145527974524585, 0.0, 0.0],
[0.0011716889847046552, 0.0, 3.5684064956923925e-68, 4.635957297058711e-35, 0.0, 0.0, 4.440892098500626e-16, 0.0, 0.0, 3.2503460491632774, 0.0, 0.21626929458280875, 3.208460653725017, 28.671105967304488, 0.27706318643782435, 0.05173728074972317, 5.095184202517254, 0.0, 0.0],
[0.002528167404571624, 0.0, 3.532748744044855e-62, 1.2669433409675295e-36, 0.0, 0.0, 4.440892098500626e-16, 0.0, 0.0, 3.367454573291353, 0.0, 0.30968468933812504, 3.6132091789962932, 28.19639992784475, 0.21096172463506901, 0.5578436827204847, 0.37602693264992126, 0.0, 0.0],
[0.006994507069665756, 0.0, 8.287125139793209e-66, 3.027231785818716e-35, 0.0, 0.0, 4.440892098500626e-16, 0.0, 0.0, 3.5308708098636425, 0.0, 0.44271993709010543, 3.743406442948707, 28.5816837273925, 0.1859190241300528, 0.013567646449098802, 0.3544119368508465, 0.0, 0.0],
[0.0011324906632708132, 0.0, 2.997643137303976e-93, 5.5543065111484264e-49, 0.0, 0.0, 4.440892098500626e-16, 0.0, 0.0, 3.689733322991782, 0.0, 0.27003343172524763, 2.9259873021311553, 27.966507569488904, 0.2015393788568267, 0.015314202079718342, 5.099975776417529, 0.0, 0.0],
[0.002460378405598318, 0.0, 3.454306353558546e-92, 4.953376279441098e-50, 0.0, 0.0, 4.440892098500626e-16, 0.0, 0.0, 2.708698957365992, 0.0, 0.45988488933240973, 3.3794279089763397, 28.845981533786045, 0.2930570106032704, 5.40357787755495, 5.094952613797775, 0.0, 0.0],
[0.0038356340834161395, 0.0, 2.984497034303708e-88, 5.9184054440408605e-52, 0.0, 0.0, 4.440892098500626e-16, 0.0, 0.0, 2.985315000737332, 0.0, 0.19150082741871666, 3.490747944498573, 28.644542940980433, 0.4841349889376101, 1.4448397948285407, 0.3655423359219476, 0.0, 0.0],
[0.00064964422750724, 0.0, 4.42723239896592e-92, 2.040344204906956e-50, 0.0, 0.0, 4.440892098500626e-16, 0.0, 0.0, 3.426555900848266, 0.0, 0.21256989721465186, 3.255451967845857, 27.257714764718447, 0.28038435017071084, 5.6345016565545425, 0.008471619110638429, 0.0, 0.0],
[0.0010388234679355806, 0.0, 5.238432020418103e-92, 5.0364325290413714e-48, 0.0, 0.0, 4.440892098500626e-16, 0.0, 0.0, 2.4403664202194157, 0.0, 0.44904147668570965, 2.320295274475672, 28.102790342925022, 0.2735622743474919, 0.016576480259509196, 0.015023168524315622, 0.0, 1.1102230246251565e-16],
[0.0010224969052758449, 0.0, 2.957879034921993e-144, 7.396698664677707e-81, 0.0, 0.0, 4.440892098500626e-16, 0.0, 0.0, 2.973916076912293, 0.0, 0.3548391420701694, 3.4814775452509057, 28.070378342937236, 0.25974700038724263, 5.403489941828254, 0.001029549585043199, 0.0, 0.0],
[0.004091236083263832, 0.0, 4.2389858433683344e-148, 3.3261069055597983e-80, 0.0, 0.0, 4.440892098500626e-16, 0.0, 0.0, 2.8948599978082403, 0.0, 0.756677635708706, 2.2429429188546375, 28.01059935217724, 0.04738466645372563, 5.404396846389392, 5.09560039169605, 0.0, 0.0],
[0.0017123346446643595, 1.8270055957198982e-93, 8.034285845473118e-145, 1.2390207964074959e-76, 0.0, 0.0, 4.440892098500626e-16, 0.0, 0.0, 3.20833268631673, 0.0, 0.20159326541183833, 3.117428266204545, 27.244961834999522, 0.25972370534949807, -0.0027270158359780083, 5.094819101272357, 0.0, 0.0],
[0.00659977542946566, 0.0, 4.37208814978955e-143, 1.0588571309357448e-78, 0.0, 0.0, 4.440892098500626e-16, 0.0, 0.0, 3.2896155032760634, 0.0, 0.20011955086484448, 3.5291126735463596, 28.604635172956712, 0.2941122484815186, 3.2139774276270394, -0.0016109785930460419, 0.0, 0.0],
[0.00020999744670369136, 3.940756408619225e-91, 2.8004751997757155e-145, 7.202857024344397e-79, 0.0, 0.0, 4.440892098500626e-16, 0.0, 0.0, 3.1710070276588915, 0.0, 0.3250553269921979, 2.6861132757296415, 28.01668204083364, 0.6341762225607859, 0.017051962411693822, 5.095218866643159, 0.0, 0.0],
[0.001776899469679738, 0.0, 2.0173783355464664e-55, 9.873830443132604e-32, 0.0, 0.0, 4.440892098500626e-16, 0.0, 0.0, 2.6323301675797257, 0.0, 0.4973582432523372, 3.9483391742495493, 28.175949544224363, 0.017783918308129643, 0.0703177788697733, 0.051124218054344084, 0.0, 0.0],
[0.005663423301738191, 0.0, 8.089966754976105e-57, 1.6969053815025827e-31, 0.0, 0.0, 4.440892098500626e-16, 0.0, 0.0, 3.647644478324412, 0.0, 0.3787211428590606, 4.230013739456745, 28.981049451504397, 0.27904009150327136, 0.05150300604065983, 0.5722761222689527, 0.0, 0.0],
[0.0033210790890256803, 0.0, 1.3110604382108807e-56, 1.3264026499957032e-32, 0.0, 0.0, 4.440892098500626e-16, 0.0, 0.0, 3.3765183644950705, 0.0, 0.37775661422816054, 3.3198119548267053, 28.900835166321123, 0.14809541108625446, 0.19156132529260717, 3.4375953508955677, 0.0, 0.0],
[0.0035340925073288343, 0.0, 3.5478169617164244e-58, 7.426250864969185e-30, 0.0, 0.0, 4.440892098500626e-16, 0.0, 0.0, 2.7221116203271962, 0.0, 0.30322910468052855, 4.045208294801235, 27.775084270300994, 0.305133784595371, 0.15090911808457008, 0.16630659049950225, 0.0, 0.0],
[0.0029308472177975325, 0.0, 2.717000539236085e-54, 4.345602290422747e-31, 0.0, 0.0, 4.440892098500626e-16, 0.0, 0.0, 2.2945249155418805, 0.0, 0.4789455409584018, 3.087167713684007, 28.71584362902207, 0.24454892393305672, 0.03553715349891107, 7.376404997910102, 0.0, 1.1102230246251565e-16],
[0.0010513039745339015, 0.0, 1.2093316494838236e-108, 1.7334323116373587e-58, 0.0, 0.0, 4.440892098500626e-16, 0.0, 0.0, 4.369794568794567, 0.0, 0.35275286677904577, 3.5828511401107046, 28.049709391279407, 0.3062498604913535, 0.02041885069143312, 0.027429633825592603, 0.0, 0.0],
[0.0002483738730505447, 0.0, 1.1790253022075942e-110, 5.951339007130388e-61, 0.0, 0.0, 4.440892098500626e-16, 0.0, 0.0, 2.7893875723050887, 0.0, 0.3303385010612563, 3.5326557569629005, 28.067313686543695, 0.2072035683802147, 1.6020243114378836, 0.010748087412327578, 0.0, 0.0],
[4.9303526424737676e-05, 0.0, 1.5510705223821505e-110, 1.0164848818796313e-59, 0.0, 0.0, 4.440892098500626e-16, 0.0, 0.0, 2.754884910268987, 0.0, 0.3261368482864234, 2.9204265501170945, 27.052359840122957, 0.446119269949492, 0.014391753052622747, 5.094845559677463, 0.0, 0.0],
[0.0038359547234119934, 0.0, 1.5406582007782628e-108, 3.1923130843273693e-60, 0.0, 0.0, 4.440892098500626e-16, 0.0, 0.0, 3.5802168842194826, 0.0, 0.14301947423656805, 2.8661491467124147, 28.645287988683542, 0.4585305241412998, 0.07139748730605255, 4.251850414596389, 0.0, 0.0],
[0.007911316539068625, 0.0, 1.4076911203597295e-111, 7.106053272617367e-58, 0.0, 0.0, 4.440892098500626e-16, 0.0, 0.0, 2.9677952687375235, 0.0, 0.26128479551991207, 3.6296039658583714, 27.69562976251851, 0.0951518942771461, 5.403551295223716, 0.059913275178509195, 0.0, 1.1102230246251565e-16],
[0.008481293673149537, 0.0, 1.2476853778182368e-15, 1.2564022529342286e-08, 0.0, 0.0, 4.440892098500626e-16, 0.0, 0.0, 4.06707272049392, 0.0, 0.5892855877744223, 5.106674451038335, 28.710932872434803, 0.15482679902697782, 8.142338495033021, 7.526968651363655, 0.0, 1.3456671332789938e-09],
[0.0012587874556634349, 0.0, 2.4922703679742835e-16, 1.61956260530985e-09, 0.0, 0.0, 4.440892098500626e-16, 0.0, 0.0, 4.5160320619749745, 0.0, 0.6993863179523627, 4.198157517158116, 29.0, 0.03441204041009316, 0.5922326122154615, 0.30512561547953965, 0.0, 2.8215363379047176e-10],
[0.002905456651209559, 0.0, 1.5040899679194657e-15, 1.7718139825228256e-08, 0.0, 0.0, 4.440892098500626e-16, 0.0, 0.0, 5.004449879210669, 0.0, 1.611052450780233, 7.5, 28.904009250563313, 0.07106255088887226, 0.8311125767148919, 0.220877081814006, 0.0, 2.346469685221564e-10],
[0.0011805477803429332, 0.0, 5.614316796646644e-15, 8.20232363362743e-09, 0.0, 0.0, 4.440892098500626e-16, 0.0, 0.0, 4.475459233189997, 0.0, 0.7913587965186855, 5.017172478122843, 29.0, 0.24988456226971456, 3.062557028522133, 0.2514484442892524, 0.0, 8.888694225106519e-10],
[0.0052212295509114215, 0.0, 9.575086748851244e-16, 3.588950585706081e-08, 0.0, 0.0, 4.440892098500626e-16, 0.0, 0.0, 4.145705018957979, 0.0, 0.4372252902264389, 5.04439469773531, 28.755595469159854, 0.24402089492190138, 5.812599347092866, 0.08893390452645278, 0.0, 1.4900309874832374e-09],
[0.0026448225008089965, 0.0, 3.8015094968082136e-45, 1.925688431457168e-26, 0.0, 0.0, 4.440892098500626e-16, 0.0, 0.0, 3.5644254782142926, 0.0, 0.47288352466931527, 4.726709285426609, 29.0, 0.2229682693006616, 0.14009358238513947, 0.18602258473389988, 0.0, 1.1102230246251565e-16],
[0.006264545157216474, 0.0, 1.7333836896222176e-45, 6.057072995186095e-25, 0.0, 0.0, 4.440892098500626e-16, 0.0, 0.0, 3.8995993168033403, 0.0, 0.5689725935479493, 3.710650039009525, 29.0, 0.2745418105141617, 5.405080155970758, 0.12463933966015261, 0.0, 1.1102230246251565e-16],
[1.8311341465904185e-05, 0.0, 1.8565407889078214e-46, 3.4809620455144834e-26, 0.0, 0.0, 4.440892098500626e-16, 0.0, 0.0, 3.328248200787129, 0.0, 0.32784179539264535, 3.9352029744418586, 29.0, 0.41129393819202953, 0.3999314216647214, 0.026322996559946077, 0.0, 0.0],
[0.009672821109327656, 0.0, 1.9006590899308045e-50, 3.74215145568325e-26, 0.0, 0.0, 4.440892098500626e-16, 0.0, 0.0, 3.7067800238812847, 0.0, 0.40274923483825603, 3.978764123237067, 28.88691319300235, 0.166156372493806, 1.1160058861365378, 5.095771980737323, 0.0, 0.0],
[0.005189043530530291, 0.0, 6.382014561402028e-47, 1.4523718125200036e-25, 0.0, 0.0, 4.440892098500626e-16, 0.0, 0.0, 3.6359383231457274, 0.0, 0.3374156305471345, 3.431147417931309, 28.67359296962389, 0.16709145265675396, 0.02971561506286413, 6.830099237691055, 0.0, 0.0],
[1.791237810444546e-05, 0.0, 4.8737879352683614e-88, 4.126522920652338e-46, 0.0, 0.0, 4.440892098500626e-16, 0.0, 0.0, 2.9639803954707533, 0.0, 0.35195139150826266, 4.185028214488532, 28.031019798309682, 0.2061220230070524, 0.004373127598004345, 5.095168971927723, 0.0, 0.0],
[0.0005675173633270525, 0.0, 2.034315027270969e-87, 1.3027991245095437e-49, 0.0, 0.0, 4.440892098500626e-16, 0.0, 0.0, 3.0220668233830863, 0.0, 0.49384282245310424, 3.2779997661640747, 28.841069741796215, 0.12834778969390515, 0.029255901204185975, 0.024552684547181514, 0.0, 0.0],
[0.0005941571350496971, 0.0, 1.1143870482590617e-87, 1.3144689044493009e-48, 0.0, 0.0, 4.440892098500626e-16, 0.0, 0.0, 3.4816182768745763, 0.0, 0.7768353694762784, 3.864337081214846, 29.0, 0.1865180160552149, 0.23958000179785088, 5.096708459370579, 0.0, 0.0],
[0.0036520623118887485, 0.0, 6.687782405477859e-89, 4.362421210458668e-48, 0.0, 0.0, 4.440892098500626e-16, 0.0, 0.0, 3.081773830319662, 0.0, 0.3714950576384597, 2.144214637453296, 27.802981870855632, 0.1577269352969215, 0.03091266406511295, 0.015081721107069512, 0.0, 0.0],
[0.0074544844691569745, 0.0, 3.3250254427740075e-88, 1.898409323590111e-46, 0.0, 0.0, 4.440892098500626e-16, 0.0, 0.0, 3.054442953101695, 0.0, 0.16110587255828152, 3.366953651999475, 28.017731932103686, 0.01646869275699281, 0.1641724513617593, 0.5261841325249161, 0.0, 0.0],
[6.65857479738019e-05, 0.0, 5.0229811188551755e-20, 2.025675087980393e-11, 0.0, 0.0, 4.440892098500626e-16, 0.0, 0.0, 4.552205858657807, 0.0, 0.6312903199024247, 4.962182359580711, 28.89041254060304, 0.26976480762764643, 6.941415444770273, 4.8963328021689385, 0.0, 1.4344081478157023e-13],
[0.0016115417780191432, 0.0, 2.1937507047317596e-20, 7.996081503227177e-12, 0.0, 0.0, 4.440892098500626e-16, 0.0, 0.0, 4.413854048908128, 0.0, 0.3947268414568627, 7.326815003380818, 29.0, 0.024476089512510235, 0.19770488976072542, 1.1105545646283925, 0.0, 1.3183898417423734e-12],
[0.0030186865221495385, 0.0, 2.147312810993741e-19, 5.360305524266366e-12, 0.0, 0.0, 4.440892098500626e-16, 0.0, 0.0, 4.004368221560001, 0.0, 0.5211164436018042, 5.4251962196734365, 29.0, 0.1696113990692072, 0.539432143174027, 3.8836322301944346, 0.0, 1.0249578963339445e-12],
[0.047364076902181806, 0.0, 1.2631332679067876e-18, 2.7104247986330965e-11, 0.0, 0.0, 4.440892098500626e-16, 0.0, 0.0, 3.8434537066564625, 0.0, 0.4574327858981646, 6.190954103127469, 29.0, 0.22093479096547997, 2.2522629791415927, 7.60300077466788, 0.0, 1.0615952561465747e-12],
[0.00977284625987973, 0.0, 8.81484950119578e-20, 4.645872826366327e-12, 0.0, 0.0, 4.440892098500626e-16, 0.0, 0.0, 4.554614303189914, 0.0, 0.7915129613895455, 7.5, 29.0, 0.35579590648750425, 5.459419649307565, 2.5060991041930443, 0.0, 0.0],
[0.008015138773601748, 0.0, 1.5153489544111472e-60, 3.838263496467768e-36, 0.0, 0.0, 4.440892098500626e-16, 0.0, 0.0, 3.0728265681942553, 0.0, 0.24442363482218094, 4.136682264851983, 28.93973294380472, 0.9968192474606115, 0.0324109612752963, 0.36474875726438505, 0.0, 0.0],
[0.0011114544057585496, 0.0, 3.593260071461885e-58, 2.1736168320988207e-32, 0.0, 0.0, 4.440892098500626e-16, 0.0, 0.0, 3.952560913855529, 0.0, 0.297082690640012, 3.3073667373047506, 28.883968919997482, 0.6276837862501461, 0.028735582984875663, 0.0441134769728766, 0.0, 0.0],
[0.004784926717017579, 0.0, 1.694177860202549e-60, 1.0619623307427976e-33, 0.0, 0.0, 4.440892098500626e-16, 0.0, 0.0, 3.3155378839866625, 0.0, 0.15034971455009746, 4.326606923748668, 28.035645138447872, 0.04090798421416597, 0.03932675567920185, 0.07670695737394873, 0.0, 1.1102230246251565e-16],
[0.003235955563143156, 0.0, 2.3172717525593454e-57, 2.2242071632693952e-32, 0.0, 0.0, 4.440892098500626e-16, 0.0, 0.0, 3.35368580955641, 0.0, 0.8675898651540322, 3.258826820624581, 28.873090868842862, 0.13819075450598417, 0.028094616623722146, 0.01333314031947097, 0.0, 0.0],
[0.0016902347424124775, 0.0, 8.05531814106057e-61, 1.2698989532754656e-31, 0.0, 0.0, 4.440892098500626e-16, 0.0, 0.0, 3.7530867626590574, 0.0, 0.3109248284620389, 3.445288141180506, 28.136802889184143, 0.2422552563696967, 3.8363509589639033, 0.0779417145207848, 0.0, 0.0],
[0.0011840241108034277, 0.0, 2.6687172812122283e-95, 1.4729684100800525e-51, 0.0, 0.0, 4.440892098500626e-16, 0.0, 0.0, 2.8067832656115166, 0.0, 0.31384284625672737, 3.6113883000059754, 28.64904322394836, 0.261931543060971, 9.585588253439138, 0.10862806848121842, 0.0, 0.0],
[0.0020731145578412047, 0.0, 2.4180892858845663e-95, 2.811348551805953e-51, 0.0, 0.0, 4.440892098500626e-16, 0.0, 0.0, 3.816337687454266, 0.0, 0.3278080314686748, 3.108494646266536, 28.641161390987968, 0.25966093727655704, 5.404063535264522, 0.033772610594608565, 0.0, 0.0],
[0.002007464132994304, 0.0, 1.0820508374255624e-96, 3.4844369271147826e-52, 0.0, 0.0, 4.440892098500626e-16, 0.0, 0.0, 3.1518944598444274, 0.0, 0.32334993345808394, 2.669884346668316, 28.62707468496734, 0.01656573071741896, 0.037719758134324266, 0.11782706226254192, 0.0, 0.0],
[0.0020320434886170746, 0.0, 1.5986727105842086e-95, 6.4888860559663315e-52, 0.0, 0.0, 4.440892098500626e-16, 0.0, 0.0, 2.97281853414584, 0.0, 0.4184329532311619, 3.0662247709599226, 28.890082974533307, 0.25033297773540397, 5.40354820339337, 0.014380278179583428, 0.0, 0.0],
[0.001207196411951994, 0.0, 3.306643996225067e-101, 1.1121026470497518e-50, 0.0, 0.0, 4.440892098500626e-16, 0.0, 0.0, 3.019524493530243, 0.0, 0.23774874485876893, 3.108854070172984, 28.641757677469194, 0.2568092869467118, 0.016340769064044025, 0.023210858537417778, 0.0, 0.0]]

ARV=[]
m=(np.array(out1)).min(0)
j=0
for i in range(25):
    s=0
    for ra in range(j,j+5):
        print(ra)
        for la in range(len(out1[ra])):
            s+=(out1[ra][la]-m[la])
    print(s)
    ARV.append(s/(5*19))
    j+=5
print(ARV)
