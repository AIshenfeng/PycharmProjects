#coding=utf-8
import os
str='/home/shenfeng/下载/江苏省科研创新实践大赛-行人检测数据集/BMPImages'
f=os.listdir(str)
fw=open('imagine_index.txt','w')
for file in f:
    print(file)
    temp=file.split('.')[0]
    print(temp)
    fw.write('%s\n'%temp)
fw.close()