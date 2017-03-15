import csv
import numpy as np
import sys



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

#my feature [3] CO 2 [7]NOx 3 [8]O3 1 [9]PM10 5 [10]PM2.5 9 [12]RH 3 [14]THC 1
my_feature = np.array([[0.0]*9]*18)
my_feature[2][8] = 1.0
my_feature[6][6:9] = 1.0
my_feature[7][8] = 1.0
my_feature[8][4:9] = 1.0
my_feature[9][0:9] = 1.0
my_feature[11][6:9] =1.0
my_feature[13][8] = 1.0

#0.02 0.5 20 5000 110
#0.1 0.5 20 5000 56
#0.05 0.5 20 5000 50
#0.1 0.5 16 5000 47
#0.05 0.5 16 5000 47.5
#0.005 0.5 10 5000 36.5
#0.005 0.5 14 5000 30
#0.002 0.5 14 10000 29
#0.5 0.5 14 4208 23.6
#0.01 0.5 14 2148 24.5
#0.1 0.5 14 4926 0.5 23.4
#0.001 0.5 14 5000 0.25 15
#0.005 0.5 10 5000 0.25 13.5

Weight = np.array([[0.0]*9]*18)
Weight[2][8] = 0.03461
Weight[6][6] = -0.02383217
Weight[6][7] = 0.029049
Weight[6][8] = 0.13954667
Weight[7][8] = 0.03626642
Weight[8][4] = -0.014644
Weight[8][5] = -0.0317656
Weight[8][6] = 0.001204
Weight[8][7] =-0.038192
Weight[8][8] = 0.0682514
Weight[9][0] = 0.039375
Weight[9][1] = -0.06743533
Weight[9][2] = 0.136419
Weight[9][3] = -0.10733
Weight[9][4] = -0.07999
Weight[9][5] = 0.3273
Weight[9][6] = -0.40363928
Weight[9][7] =-0.05221217
Weight[9][8] = 1.01638
Weight[11][6] = 0.12653
Weight[11][7] = -0.091377
Weight[11][8] = -0.068541
Weight[13][8] = 0.23027
g_Weight = np.array([[0.0]*9]*18)
wb=np.array([[0.0]*9]*18)
Target_Error = 0.0
Bias = -0.45068
g_Bias = 0
gb=0.0
learning_rate = 0.005
Ac_rate = 0.5
Ac_restrict = 10.0
Ans = 0.0
g_Weight_past = np.array([[1.0]*9]*18)
g_Bias_past = 0.0
w_p = np.array([[1.0]*9]*18)
WTF = np.array([[1.0]*9]*18)
b_p = 1.0
iteration = 4578
Min_Error = 100.0
Min_Error_position = 0
filter_ratio = 0.25

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








