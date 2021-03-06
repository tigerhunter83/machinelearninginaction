from numpy import *
from numpy import linalg as la

def test1():
    U, Sigma, VT = linalg.svd([[1, 1], [7, 7]])
    print(U)
    print(Sigma)
    print(VT)

def loadExData():
    return [[0, 0, 0, 2, 2],
            [0, 0, 0, 3, 3],
            [0, 0, 0, 1, 1],
            [1, 1, 1, 0, 0],
            [2, 2, 2, 0, 0],
            [5, 5, 5, 0, 0],
            [1, 1, 1, 0, 0]]
        
def ecluSim(inA, inB):
    return 1.0/(1.0 + la.norm(inA - inB))

def pearsSim(inA, inB):
    if len(inA) < 3: return 1.0

    return 0.5 + 0.5*corrcoef(inA, inB, rowvar = 0)[0][1]

def cosSim(inA, inB):
    num = float(inA.T*inB)
    denom = la.norm(inA)*la.norm(inB)
    return 0.5 + 0.5 * (num/denom)

def standEst(dataMat, user, simMeas, item):
    n = shape(dataMat)[1]
    simTotal = 0.0; ratSimTotal = 0.0
    for j in range(n):
        userRating = dataMat[user, j]
        if userRating == 0: continue
        overLap = nonzero(logic_and(dataMat[:, item].A > 0, dataMat[:, j].A > 0))
        if len(overLap) == 0: similarity = 0
        else: similarity = simMeas(dataMat[overLap, item], dataMat[overLap, j])
        # print('the %d and %d similarity is : %f' % (item, j, similarity))
        simTotal += similarity
        ratSimTotal += similarity * userRating
    if simTotal == 0: return 0
    else: return ratSimTotal/simTotal

def recomend(dataMat, user, N = 3, simMeas = cosSim, estMethod = standEst):
    unratedItems = nonzero(dataMat[user, :].A == 0)[1]
    if len(unratedItems) == 0: return 'you rated everything'
    itemScores = []
    for item in unratedItems:
        estimatedScore = estMethod(dataMat, user, simMeas, item)
        itemScores.append((item, estimatedScore))

    return sorted(itemScores, key = lambda jj: jj[1], reverse=True)[:N]

def svdEst(dataMat, user, simMeas, item):
    n = shape(dataMat)[1]
    simTotal =0.0; ratSimTotal = 0.0
    U, Sigma, VT =la.svd(dataMat)
    Sig4 = mat(eye(4)* Sigma[:4])
    xformedItems = dataMat.T * U[:, :4]* Sig4.I
    for j in range(n):
        userRating = dataMat[user, j]
        if userRating == 0 or j == item : continue
        similarity = simMeas(xformedItems[item, :].T, xformedItems[j, :].T)
        print ('the %d and %d similarity is: %f' % (item, j, similarity))
        simTotal += similarity
        ratSimTotal += similarity * userRating
    if simTotal == 0: return 0
    else:
        return ratSimTotal/simTotal

def printMat(inMat, thresh = 0.8):
    for i in range(32):
        for k in range(32):
            if float(inMat[i, k]) > thresh:
                print(1, end=' ')
            else:
                print(0, end = ' ')
        print('\n')


def imgCompress(numSV = 3, thresh =0.8):
    myl = []
    for line in open('0_5.txt').readlines():
        newRow = []
        for i in range(32):
            newRow.append(int(line[i]))
        myl.append(newRow)
    myMat = mat(myl)
    print('****original matrix*******')
    printMat(myMat, thresh)
    U, Sigma, VT = la.svd(myMat)
    SigRecon = mat(zeros((numSV, numSV)))
    for k in range(numSV):
        SigRecon[k, k] = Sigma[k]
    reconMat = U[:, :numSV]* SigRecon*VT[:numSV, :]
    print('****reconstructed matrix using %d singular values*****' % numSV)
    printMat(reconMat, thresh)





def test2():
    U, Sigma, VT = linalg.svd(loadExData())
    print(U)
    print(Sigma)
    print(VT)
        
def main():
    #test1()
    #test2()
    imgCompress(2)

main()