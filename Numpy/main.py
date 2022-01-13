import numpy as np

def main():
    # result = 0
    # for i in range(100):
    #     result += i
    # print('sum(%d) is %d' % (100, result))
    l = list(range(10))
    L2 = [str(c) for c in l]
    print(L2)
    arr = np.mat(np.zeros((3, 5)))
    arr[0,1] = 2
    print(arr[:, 0])
    #print(l)

main()