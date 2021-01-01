# 下一步工作：写一些性能验证函数(可视化等等）
import numpy as np


def confusion_matrix(labels, predictions):
    assert type(labels) is np.ndarray
    confusion = []
    num_items, num_classes = labels.shape
    classes = np.zeros(num_classes).astype(np.bool)
    labels = labels.astype(np.bool)
    predictions = predictions.astype(np.bool)
    for i in range(num_classes):
        cur_class = classes
        cur_class[i] = True
        class_i_only = np.where(np.bitwise_not(np.any(np.bitwise_xor(cur_class, labels), axis=1)))
        predictions_for_i = predictions[class_i_only, :]
        epsilon = 1e-5
        num_class_i = np.size(class_i_only, 0)
        confusion_i = np.sum(predictions_for_i, axis=1) + epsilon / num_class_i + epsilon * num_classes
        confusion.append(confusion_i)
    return np.array(confusion)



def confusion_mat(true, est):
    # confusion行是真值，列是估计值
    if np.size(true, 0) == 0:
        print('row==0没有数据')
        pass

    confusion = np.mat(np.zeros((np.size(true, 1), np.size(true, 1))))
    total = np.zeros((np.size(true, 1)))
    k = np.size(true, 0)
    for i in range(k):
        if np.sum(true[i]) == 1:
            n = np.where(true[i] == 1)
            confusion[n[1][0], :] += est[i, :]
            total[n[1][0]] += 1
    confusion = np.true_divide(confusion, np.transpose(np.mat(total)))

    return confusion


if __name__ == '__main__':
    a = np.mat([[1, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]).astype(np.bool)
    b = np.mat([[0, 1, 0], [0, 1, 1], [1, 0, 0], [0, 1, 0]]).astype(np.bool)
    c = confusion_mat(a, b)
    print(c)
