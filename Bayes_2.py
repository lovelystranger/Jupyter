import numpy as np
import math
from sklearn.datasets import load_iris

iris = load_iris()  # 加载数据
y = iris.target
x = iris.data


def naive_bayes(x, y, predict):
    unique_y = list(set(y))
    label_num = len(unique_y)
    sample_num, dim = x.shape
    joint_p = [1] * label_num
    for (label_index, label) in enumerate(unique_y):
    	p_c = len(y[y == label]) / sample_num
    	for (feature_index, x_i) in enumerate(predict):
    		tmp = x[y == label]
    		bayes_mean = np.mean(tmp,axis = 0)
    		bayes_var = np.var(tmp,axis = 0)
    		a = 1/(math.sqrt(2*math.pi)*bayes_var[feature_index])
    		b = -((x_i-bayes_mean[feature_index])**2)/(2*(bayes_var[feature_index])**2)
    		joint_p[label_index] = a*math.exp(b)
    	joint_p[label_index] *=p_c

    tmp = joint_p[0]
    max_index = 0
    for (i, p) in enumerate(joint_p):
        if tmp < p:
            tmp = p
            max_index = i

    return unique_y[max_index]
#测试所用的数据为iris数据集中的第46-50，96-100，146-150条数据，类别分别为0，1，2
out1 = naive_bayes(x, y, np.array([4.8,3.0,1.4,0.3]))
out2 = naive_bayes(x, y, np.array([5.1,3.8,1.6,0.2]))
out3 = naive_bayes(x, y, np.array([4.6,3.2,1.4,0.2]))
out4 = naive_bayes(x, y, np.array([5.3,3.7,1.5,0.2]))
out5 = naive_bayes(x, y, np.array([5.0,3.3,1.4,0.2]))
out6 = naive_bayes(x, y, np.array([5.7,3.0,4.2,1.2]))
out7 = naive_bayes(x, y, np.array([5.7,2.9,4.2,1.3]))
out8 = naive_bayes(x, y, np.array([6.2,2.9,4.3,1.3]))
out9 = naive_bayes(x, y, np.array([5.1,2.55,3.0,1.1]))
out10 = naive_bayes(x, y, np.array([5.7,2.8,4.1,1.3]))
out11 = naive_bayes(x, y, np.array([7.7,3.0,6.1,2.3]))
out12 = naive_bayes(x, y, np.array([6.3,3.4,5.6,2.4]))
out13 = naive_bayes(x, y, np.array([6.4,3.1,5.5,1.8]))
out14 = naive_bayes(x, y, np.array([6.0,3.0,4.8,1.8]))
out15 = naive_bayes(x, y, np.array([6.9,3.1,5.4,2.1]))

print(out1,out2,out3,out4,out5,out6,out7,out8,out9,out10,out11,out12,out13,out14,out15)