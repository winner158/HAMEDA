import pickle

filename = "data/lecs/lecs_40_seed24610.pkl"
# 打开pkl文件
with open(filename, 'rb') as f:
    data = pickle.load(f)

# 现在可以对data变量进行操作，‌比如打印出来查看内容
print(data)
