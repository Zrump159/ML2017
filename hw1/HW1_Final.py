import csv
import sys
import numpy as np



def Predict_Ans( point, Weight, Bias ) :
    temp = np.array([[]*9]*18)
    temp = np.multiply(Weight,point[...,:9])
    Sum = np.sum(temp)
    Ans = Sum + Bias
    return Ans
def Gradient( point, Weight, P_Ans ) :
    Gradient_matrix = np.array([[]*9]*18)
    Gradient_matrix = np.multiply(2.0*(point[9][9] - P_Ans),-point[...,:9])
    return Gradient_matrix




#------------------------------------------------------------------------
#main function
#total 4320/18 = 240*24hour data
#240*24hour/9 = 640 trainset



i = 0
data = [[]]*18
T_data = [[]]*18
trainset = [[[]*10]*18]*1000
temp = [[]]*18


with open(sys.argv[1],encoding='Big5',newline = '') as csvfile :
    reader = csv.reader(csvfile, delimiter= ',')
    csvfile.readline()
    for row in reader :
        i = i % 18
        data[i] = data[i] + [x if x != 'NR' else 0.0 for x in row[3:26]]
        i += 1

csvfile.close()
data = np.array(data,dtype=np.float64)

for i in range(1000) :
    slice = [data[x][0+5*i:10+5*i] for x in range(18)]
    trainset[i] = slice

trainset = np.array(trainset,dtype=np.float64)

#----------------------------------------------------------------------------
#linear regression

#my feature [3] CO 2 [5]NO 2 [6]NO2 2 [7]NOx 2 [8]O3 1 [9]PM10 5 [10]PM2.5 9 [12]RH 3 [14]THC 2
my_feature = np.array([[0.0]*9]*18)
my_feature[2][8] = 1.0
my_feature[4][7:9] = 1.0
my_feature[5][8] = 1.0
my_feature[6][7:9] = 1.0
my_feature[7][8] = 1.0
my_feature[8][4:9] = 1.0
my_feature[9][4:9] = 1.0
my_feature[11][6:9] =1.0
my_feature[13][7:9] = 1.0

#1 0.5 14 5000 38
#0.01 0.5 14 5000 33
#0.002 0.5 14 5000 30
#0.005 0.5 14 5000 27
#0.1 0.5 14 5000 24.1

Weight = np.array([[0.0]*9]*18)
Weight[2][8] = 0.04
Weight[4][7] = 0.0138
Weight[4][8] = 0.1
Weight[5][8] = 0.098
Weight[6][7] = -0.025
Weight[6][8] = 0.0536
Weight[7][8] = 0.0492
Weight[8][4] = -0.0412
Weight[8][5] = -0.0323
Weight[8][6] = -0.0249
Weight[8][7] =-0.0514
Weight[8][8] = 0.114
Weight[9][4] = -0.0412
Weight[9][5] = 0.3
Weight[9][6] = -0.3909
Weight[9][7] =-0.0429
Weight[9][8] = 1.033
Weight[11][6] = 0.1166
Weight[11][7] = -0.048
Weight[11][8] = -0.06
Weight[13][7] = 0.2275
Weight[13][8] = 0.231
g_Weight = np.array([[0.0]*9]*18)
wb=np.array([[0.0]*9]*18)
Target_Error = 0.0
Bias = -0.396
g_Bias = 0
gb=0.0
learning_rate = 0.1
Ac_rate = 0.5
Ac_restrict = 14.0
Ans = 0.0
g_Weight_past = np.array([[1.0]*9]*18)
g_Bias_past = 0.0
w_p = np.array([[1.0]*9]*18)
WTF = np.array([[1.0]*9]*18)
b_p = 1.0
iteration = 1900
Min_Error = 100.0
Min_Error_position = 0
filter_ratio = 0.5

for i in range(iteration) :
    Error = 0.0
    go = 0
    for j in range(1000) :
        P_Ans = Predict_Ans( trainset[j], Weight*my_feature, Bias )
        if abs(trainset[j][9][9] - P_Ans )/P_Ans > filter_ratio :
            go += 1
            continue
        g_Weight += my_feature*Gradient( trainset[j], Weight*my_feature, P_Ans )
        g_Bias += -2.0*(trainset[j][9][9] - P_Ans)
        Error += (trainset[j][9][9] - P_Ans)**2
    Error = Error/(1000-go)
    print("Error[",i,"]:",Error)
    if Error < Target_Error :
        break
    if Error < Min_Error :
        Min_Error = Error
        Min_Error_position = i

    wb += np.sum(g_Weight**2)
    gb += (g_Bias ** 2)


    for y in range(9) :
        for x in range(18) :
            g_Weight_now = np.divide(g_Weight[x][y], np.absolute(g_Weight[x][y]))
            if g_Weight_past[x][y] * g_Weight_now == 1 :
                if w_p[x][y] < Ac_restrict :
                    w_p[x][y] += Ac_rate
            else :
                w_p[x][y] = 1.0


    if g_Bias_past*np.divide(g_Bias, np.absolute(g_Bias)) == 1 :
        if b_p < Ac_restrict :
            b_p += Ac_rate
    else :
        b_p = 1.0

    g_Weight_past = np.divide(g_Weight, np.absolute(g_Weight))
    g_Bias_past = np.divide(g_Bias, np.absolute(g_Bias))

    g_Weight_ac = g_Weight*w_p

    Weight = Weight - learning_rate/1000 * (1. / (wb ** 0.5)) * g_Weight_ac
    Bias = Bias - learning_rate/1000 * (1. / (gb ** 0.5)) * g_Bias * b_p

print("Min_Error in [",Min_Error_position,"]:",Min_Error,'\nWeight\n',Weight,'\n',"Bias",Bias)

#-------------------------------------------------------------------------
#write csv
test_data = [[]]*18
testset = [[[]*9]*18]*240

with open(sys.argv[2],encoding='Big5',newline = '') as csvfile :
    reader = csv.reader(csvfile, delimiter= ',')
    i = 0
    for row in reader :
        i = i % 18
        test_data[i] = test_data[i] + [x if x != 'NR' else '0' for x in row[2:11]]
        i += 1

test_data = np.array(test_data,dtype=np.float64)



for i in range(240) :
    slice = [test_data[x][0+9*i:9+9*i] for x in range(18)]
    testset[i] = slice

testset = np.array(testset,dtype=np.float64)


csvfile.close()


data = np.array(data,dtype=np.float64)
with open(sys.argv[3], 'w') as outfile :
    writer = csv.writer(outfile , delimiter=',')
    output = [['id','value']]
    writer.writerows(output)

    for i in range(240) :
        my_Ans = Predict_Ans( testset[i], Weight*my_feature, Bias )
        if my_Ans < 0 :
            my_Ans = 0.0
        output = [['id_%d' %i,'%f' %my_Ans]]
        writer.writerows(output)

outfile.close()








