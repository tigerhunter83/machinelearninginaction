import dtRegression
import tkinter as tk
from numpy import *

# matplotlib.use('TkAgg')  # 设置后端TkAgg
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure


def reDraw(tolS, tolN):
    reDraw.f.clf()  # 清空之前的图像
    reDraw.a = reDraw.f.add_subplot(111)  # 重新添加新图
    if chkBtnVar.get():  # 检查选框model tree是否被选中
        if tolN < 2: tolN = 2
        myTree = dtRegression.createTree(reDraw.rawDat, dtRegression.modelLeaf, dtRegression.modelErr, (tolS, tolN))
        yHat = dtRegression.createForeCast(myTree, reDraw.testDat, dtRegression.modelTreeEval)
    else:
        myTree = dtRegression.createTree(reDraw.rawDat, ops=(tolS, tolN))
        yHat = dtRegression.createForeCast(myTree, reDraw.testDat)
    reDraw.a.scatter(reDraw.rawDat[:, 0].tolist(), reDraw.rawDat[:, 1].tolist(), s=5)  # 绘制真实值
    reDraw.a.plot(reDraw.testDat, yHat, linewidth=2.0)  # 绘制预测值
    reDraw.canvas.draw()


def getInputs():  # 获取输入
    try:  # 期望输入是整数
        tolN = int(tolNentry.get())
    except:  # 清楚错误用默认值替换
        tolN = 10
        print("enter Integer for tolN")
        tolNentry.delete(0, END)
        tolNentry.insert(0, '10')
    try:  # 期望输入是浮点数
        tolS = float(tolSentry.get())
    except:
        tolS = 1.0
        print("enter Float for tolS")
        tolSentry.delete(0, END)
        tolSentry.insert(0, '1.0')
    return tolN, tolS


def drawNewTree():
    tolN, tolS = getInputs()  # 从输入文本框中获取参数
    reDraw(tolS, tolN)  # 绘制图


root = tk.Tk()

reDraw.f = Figure(figsize=(5, 4), dpi=100)  # 创建画布
reDraw.canvas = FigureCanvasTkAgg(reDraw.f, master=root)
reDraw.canvas.draw()
reDraw.canvas.get_tk_widget().grid(row=0, columnspan=3)

tk.Label(root, text="tolN").grid(row=1, column=0)
tolNentry = tk.Entry(root)
tolNentry.grid(row=1, column=1)
tolNentry.insert(0, '10')
tk.Label(root, text="tolS").grid(row=2, column=0)
tolSentry = tk.Entry(root)
tolSentry.grid(row=2, column=1)
tolSentry.insert(0, '1.0')
tk.Button(root, text="ReDraw", command=drawNewTree).grid(row=1, column=2, rowspan=3)
chkBtnVar = tk.IntVar()
chkBtn = tk.Checkbutton(root, text="Model Tree", variable=chkBtnVar)
chkBtn.grid(row=3, column=0, columnspan=2)

reDraw.rawDat = mat(dtRegression.loadDataSet('sine.txt'))
reDraw.testDat = arange(min(reDraw.rawDat[:, 0]), max(reDraw.rawDat[:, 0]), 0.01)
reDraw(1.0, 10)

root.mainloop()
