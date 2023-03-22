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

def VGWO(Positions,objf, lb, ub, dim, SearchAgents_no, Max_iter):
    #F = 0.6  # 缩放因子
    T = 1
    T0 = 1
    K = 0.9
    EPS = 0.0001
    ccc = []
    ddd = []
    for i in Positions:
        ccc.append(i[0])
        ddd.append(i[1])
    # print(f'c={ccc};')
    # print(f'd={ddd};')
    Positions=re_gene(Positions,objf,SearchAgents_no)
    Convergence_curve_1 = []
    best_ans=objf(Positions[0])
    #迭代寻优
    l=0
    while T > EPS:
        # Positions=numpy.array(sorted(Positions,key=lambda x:objf(x)))
        Alpha_score=objf(Positions[0])
        Alpha_pos =Positions[0]
        Beta_pos =Positions[1]
        Delta_pos =Positions[2]
        # 以上的循环里，Alpha、Beta、Delta
        Gamma=math.log(EPS/T0,K)
        a = 2 - l * ((2) / Gamma)  #   a从2线性减少到0
        # a = 2-(math.log(1+1.3*math.tan((l/Gamma)**3),2))**6
        # print(a)
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
                # temp=((2 / 3) - (l / Gamma)) * X1 + (1 / 3) * X2 + (l / Gamma) * X3
                temp = (2 / 3)*(l / Gamma) * X1 + (1 / 3) * X2 + ((2 / 3) - (2 / 3)*(l / Gamma)) * X3
                if temp < lb:
                    temp = (ub + lb) / 2 - (-temp) % ((ub - lb) / 2)
                elif temp > ub:
                    temp = (ub + lb) / 2 + (temp) % ((ub - lb) / 2)
                Positions[i, j] = temp
                #Positions[i, j] = (X1 + X2 + X3) / 3  # 候选狼的位置更新为根据Alpha、Beta、Delta得出的下一代灰狼地址。
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
        F=2*(T0-T)/(T0-EPS)
        v_list = add(Positions[r1], multiply(F, substract(Positions[r2], Positions[r3],lb,ub),lb,ub),lb,ub)
        v_list_ans = objf(numpy.array(v_list))
        # Positions=numpy.array(sorted(Positions, key=lambda x: objf(x)))
        ant = 0
        for key in range(SearchAgents_no):
            if objf(Positions[key]) > v_list_ans:
                ant += 1
        # p=1/(1+math.exp((-ant)*T))#退火算子
        p = math.exp(((ant / SearchAgents_no) - 1))
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
    # print(Positions[0])
    aaa=[]
    bbb=[]
    # for i in Positions:
    #     aaa.append(i[0])
    #     bbb.append(i[1])
    # print(f'a={aaa};')
    # print(f'b={bbb};')
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


tracemalloc.start()
#主程序
func_details = [[F1, -100, 100, 30],[F2, -100, 100, 30],[F3, -100, 100, 30],
                [F4, -100, 100, 30],[F5, -100, 100, 30],[F6, -100, 100, 30],
                [F7, -32, 32, 30],[F8, -100, 100, 30],[F9, -100, 100, 30],
                [F10, -50, 50, 30],[F11, -5.12, 5.12, 30],[F12, -50, 50, 30],
                [F13, -100, 100, 30],[F14, -30, 30, 30],[F15, 0, 1, 6],
                [F16, 0, 10, 4],[F17, 0, 10, 4],[F18, -100, 100, 2],
                [F19, -10, 10, 30]]
for fun in range(len(func_details)):
    Fx = func_details[fun][0]
    output_path = 'result of Calculating st^la.txt'
    with open(output_path, 'a', encoding='utf-8') as file1:
        print(f'F{fun + 1}',file=file1)
    print(f'F{fun+1}')
    Max_iter = 90#迭代次数
    lb = func_details[fun][1]#下界-10000000000000000000000
    ub = func_details[fun][2]#上届10000000000000000000000
    dim = func_details[fun][3]#狼的寻值范围30
    SearchAgents_no = 50#寻值的狼的数量
    positions=init()
    tn=15


    Y=[]
    for i in range(tn):
        begin_time = time()
        positions_2 = positions.copy()
        y = GWO(positions_2,Fx, lb, ub, dim, SearchAgents_no, Max_iter)#经典灰狼
        end_time = time()
        run_time = end_time - begin_time
        Y.append(y[len(y)-1])
        output_path = 'result of Calculating st^la.txt'
        with open(output_path, 'a', encoding='utf-8') as file1:
            print(y[len(y)-1],end=" ", file=file1)
            print(run_time, file=file1)
        print(y[len(y)-1],end=" ")
        print(run_time)
    print(y)
    print(f'传统灰狼平均值{np.mean(Y)}标准差{np.std(Y,ddof=1)}')
    output_path = 'result of Calculating st^la.txt'
    with open(output_path, 'a', encoding='utf-8') as file1:
        print(file=file1)
    print()

    CMA=[]
    for i in range(tn):
        begin_time = time()
        positions_CMAGWO = beta_init()
        cma = CMAGWO(positions_CMAGWO,Fx, lb, ub, dim, SearchAgents_no, Max_iter)#经典灰狼
        end_time = time()
        run_time = end_time - begin_time
        CMA.append(cma[len(cma)-1])
        output_path = 'result of Calculating st^la.txt'
        with open(output_path, 'a', encoding='utf-8') as file1:
            print(cma[len(cma) - 1], end=" ", file=file1)
            print(run_time, file=file1)
        print(cma[len(cma) - 1], end=" ")
        print(run_time)
    # print(cma)
    # print(f'传统灰狼平均值{np.mean(CMA)}标准差{np.std(CMA,ddof=1)}')
    output_path = 'result of Calculating st^la.txt'
    with open(output_path, 'a', encoding='utf-8') as file1:
        print(file=file1)
    print()

    Z=[]
    for i in range(tn):
        begin_time = time()
        positions_3 = PSO_GWO_init()
        z = PSO_GWO(positions_3,Fx, lb, ub, dim, SearchAgents_no, Max_iter)#经典灰狼
        end_time = time()
        run_time = end_time - begin_time
        Z.append(z[len(z)-1])
        output_path = 'result of Calculating st^la.txt'
        with open(output_path, 'a', encoding='utf-8') as file1:
            print(z[len(z) - 1], end=" ", file=file1)
            print(run_time, file=file1)
        print(z[len(z) - 1], end=" ")
        print(run_time)
    # print(z)
    # print(f'传统灰狼平均值{np.mean(Z)}标准差{np.std(Z,ddof=1)}')
    output_path = 'result of Calculating st^la.txt'
    with open(output_path, 'a', encoding='utf-8') as file1:
        print(file=file1)
    print()



    VM=[]
    for i in range(tn):
        begin_time = time()
        positions_5 = positions.copy()
        vw = vw_GWO(positions_5,Fx, lb, ub, dim, SearchAgents_no, Max_iter)
        end_time = time()
        run_time = end_time - begin_time
        VM.append(vw[len(vw)-1])
        output_path = 'result of Calculating st^la.txt'
        with open(output_path, 'a', encoding='utf-8') as file1:
            print(vw[len(vw) - 1], end=" ", file=file1)
            print(run_time, file=file1)
        print(vw[len(vw) - 1], end=" ")
        print(run_time)
    # print(vm)
    # print(f'变权灰狼平均值{np.mean(VM)}标准差{np.std(VM,ddof=1)}')
    output_path = 'result of Calculating st^la.txt'
    with open(output_path, 'a', encoding='utf-8') as file1:
        print(file=file1)
    print()