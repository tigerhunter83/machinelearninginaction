line = input("请输入a：")
a = float(line)
line = input("请输入b：")
b = float(line)

line = input("请输入运算：1 加法 2 减法 3 乘法 4 除法\n")
choice = int(line)

if choice == 1:
    c = a + b
elif choice == 2:
    c = a - b
elif choice == 3:
    c = a * b
elif choice == 4:
    c = a / b 
operList = ["+", "-", "*", "/"]
print("%f %s %f = %f" %(a, operList[choice -1], b, c))