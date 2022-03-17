from itertools import count
import os
import pandas as pd

# from pyrsistent import T

file_path ="E:\JDT_repo\code"

dirs= os.listdir(file_path)
j=0
tag = 0
for dir in dirs:
    # First to get its tag
    csv_file = "D:/JDT/ISSTA21-JIT-DP/Data_Extraction/git_base/datasets/jdt/30k/jdt_k_feature.csv"
    csvpd = pd.read_csv(csv_file)
    for i in range(len(csvpd)):
        # print(len(csvpd))
        # print(type(dir))
        # print(type(csvpd['_id'][i]))
        if csvpd['_id'][i] == dir:
            tag = csvpd['bug'][i]
        # input("test")
    j +=1 
    temp = []
    if os.path.exists(file_path+"\\"+dir+"\\NewMatrix.txt"):
        # print(dir)
        file1 = open(file_path+"\\"+dir+"\\NewMatrix.txt")
        file2 = open(file_path+"\\"+dir+"\\OldMatrix.txt")
        
        line = file1.readline()
        line = file1.readline()
        line = line.replace('], ','\n')
        line = line.replace(',','')
        line = line.replace(']]','\n')
        line = line.replace('[','')
        line = line.replace(']','')

        line2 = file2.readline()
        line2 = file2.readline()
        line2 = line2.replace('], ','\n')
        line2 = line2.replace(',','')
        line2 = line2.replace(']]','\n')
        line2 = line2.replace('[','')
        line2 = line2.replace(']','')
        if int(j/20) <17 :
            train1 = "./my_data/new/train/train_{counter}.txt".format(counter=int(j/20)) 
            train2 = './my_data/old/train/train_{counter}.txt'.format(counter=int(j/20))
        else:
            train1 = "./my_data/new/test/test_{counter}.txt".format(counter=int(j/20)) 
            train2 = './my_data/old/test/test_{counter}.txt'.format(counter=int(j/20))
        
        # line = line.replace(' [','')
        if line != '':
            with open(train1,'a') as f:
                f.write(line)
                tag = str(tag)
                f.write("? "+tag+" "+tag+" "+tag)
                f.write('\n')
                f.write('\n')
        if line2 != '':
            with open(train2,'a') as f2:
                f2.write(line2)
                tag=str(tag)
                f2.write("? "+tag+" "+tag+" "+tag)
                f2.write('\n')
                f2.write('\n')
        
        # print(temp)
        # print(temp[1])
        # print(line)
        # for item in temp:
        #     print(item)
        # input("test")