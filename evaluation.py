# 下一步工作：写一些性能验证函数(可视化等等）
import numpy as np

def confusion_mat(true,est):
    #confusion行是真值，列是估计值
    if np.size(true,0)==0:
        print('row==0没有数据')
        pass

    confusion = np.mat(np.zeros((np.size(true,1),np.size(true,1))))
    total = np.zeros((np.size(true,1)))
    k = np.size(true,0)
    for i in range(k):
        if np.sum(true[i]) == 1:
            n = np.where(true[i] == 1)
            '''
            print(n)
            print(n[0])
            print(n[0][0])
            print(est[i,:])
            print(confusion[n,:])'''
            confusion[n[1][0],:] += est[i,:]
            total[n[1][0]] += 1
    confusion = np.true_divide(confusion,np.transpose(np.mat(total)))

    return confusion

if __name__ == '__main__':
    a = np.mat([[1,0,0],[1,0,0],[0,1,0],[0,0,1]])
    b = np.mat([[0,1,0],[0,1,1],[1,0,0],[0,1,0]])
    c = confusion_mat(a,b)