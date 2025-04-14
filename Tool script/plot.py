import pandas as pd
import matplotlib.pyplot as plt

data1R = pd.read_csv('ResNet_one_layer.txt', sep=r'\s+',header=None)
"""data2R = pd.read_csv('two_layers_rightResNet.txt', sep=r'\s+',header=None)
data2nR = pd.read_csv('two_layers_withoutResNet.txt', sep=r'\s+',header=None)
datad1_001 = pd.read_csv('output_0.01.txt', sep=r'\s+',header=None)
datad1 = pd.read_csv('output_d1.txt', sep=r'\s+',header=None)"""
#datad1_dis = pd.read_csv('output_d1_distanct.txt', sep=r'\s+',header=None)

columns = []
datad1_ct = pd.DataFrame(columns=columns)
#datad1_dis_ct = pd.DataFrame(columns=columns)
data1R_ct = pd.DataFrame(columns=columns)
data2R_ct = pd.DataFrame(columns=columns)
data2nR_ct = pd.DataFrame(columns=columns)


#d1_001
for i in range(1, 13):
    globals()[f'datad1_001{i}'] = pd.DataFrame(columns=columns)
for i in range(0,4):
    for j in range(0,4):
        for k in range(0,3):
            data= datad1_001.iloc[k+3*j+12*i]
            x = k + 3*i + 1
            globals()[f'datad1_001{x}'] = pd.concat([globals()[f'datad1_001{x}'], data.to_frame().T], ignore_index=True)




#d1
for z in range(0,16):
    for i in range(0,30):
        if i == 24:
            data = datad1.iloc[i+30*z:i+30*z+6,:]
            datad1_ct = pd.concat([datad1_ct, data], ignore_index=True)


for i in range(1, 25):
    globals()[f'datad1_ct{i}'] = pd.DataFrame(columns=columns)

for i in range(0,4):
    for j in range(0,4):
        for z in range(0,6):
            data = datad1_ct.iloc[z+6*j+24*i]
            x=z+6*i+1
            globals()[f'datad1_ct{x}'] = pd.concat([globals()[f'datad1_ct{x}'], data.to_frame().T], ignore_index=True)

#distanct_d1
"""for z in range(0,16):
    for i in range(0,30):
        if i == 24:
            data = datad1_dis.iloc[i+30*z:i+30*z+6,:]
            datad1_dis_ct = pd.concat([datad1_dis_ct, data], ignore_index=True)


for i in range(1, 25):
    globals()[f'datad1_dis_ct{i}'] = pd.DataFrame(columns=columns)

for i in range(0,4):
    for j in range(0,4):
        for z in range(0,6):
            data = datad1_dis_ct.iloc[z+6*j+24*i]
            x=z+6*i+1
            globals()[f'datad1_dis_ct{x}'] = pd.concat([globals()[f'datad1_dis_ct{x}'], data.to_frame().T], ignore_index=True)"""

#1 layer
for i in range(0,len(data1R)):
    data = data1R.iloc[i]
    if data[1] == 50000:
        data1R_ct = pd.concat([data1R_ct, data.to_frame().T], ignore_index=True)

#2 layer
for i in range(0,len(data2R)):
    data = data2R.iloc[i]
    if data[2] == 50000:
        data2R_ct = pd.concat([data2R_ct, data.to_frame().T], ignore_index=True)


"""for i in range(0,len(data2nR)):
    data = data2nR.iloc[i]
    if data[2] == 50000:
        data2nR_ct = pd.concat([data2nR_ct, data.to_frame().T], ignore_index=True)"""

for i in range(1, 11):
    globals()[f'data2R_ct{i}'] = pd.DataFrame(columns=columns)



"""for i in range(1, 11):
    globals()[f'data2nR_ct{i}'] = pd.DataFrame(columns=columns)"""


for i in range(1,11):
    for j in range(0,10):
        data = data2R_ct.iloc[10*(i-1)+j]
        globals()[f'data2R_ct{i}'] = pd.concat([globals()[f'data2R_ct{i}'], data.to_frame().T], ignore_index=True)

"""for i in range(1,11):
    for j in range(0,10):
        data = data2nR_ct.iloc[10*(i-1)+j]
        globals()[f'data2nR_ct{i}'] = pd.concat([globals()[f'data2nR_ct{i}'], data.to_frame().T], ignore_index=True)"""


plt.figure(figsize=(10, 6))

#1 layer with Res
plt.errorbar(data1R_ct[0], data1R_ct[2], yerr=data1R_ct[3], fmt='o-', color=(1.0, 0.0, 0.0),
             ecolor='lightgray', elinewidth=2, capsize=4, label='1R_ct')#red

#2 layer with Res
"""plt.errorbar(data2R_ct1[1], data2R_ct1[3], yerr=data2R_ct1[4], fmt='o-', color=(1.0, 0.5, 0.0),
             ecolor='lightgray', elinewidth=2, capsize=4, label='2R_ct1')#Orange
plt.errorbar(data2R_ct2[1], data2R_ct2[3], yerr=data2R_ct2[4], fmt='o-', color=(1.0, 1.0, 0.0),
             ecolor='lightgray', elinewidth=2, capsize=4, label='2R_ct2')#Yellow
plt.errorbar(data2R_ct3[1], data2R_ct3[3], yerr=data2R_ct3[4], fmt='o-', color=(0.5, 1.0, 0.0),
             ecolor='lightgray', elinewidth=2, capsize=4, label='2R_ct3')#Light Green
plt.errorbar(data2R_ct4[1], data2R_ct4[3], yerr=data2R_ct4[4], fmt='o-', color=(0.0, 1.0, 0.0),
             ecolor='lightgray', elinewidth=2, capsize=4, label='2R_ct4')#Green
plt.errorbar(data2R_ct5[1], data2R_ct5[3], yerr=data2R_ct5[4], fmt='o-', color=(0.0, 1.0, 0.5),
             ecolor='lightgray', elinewidth=2, capsize=4, label='2R_ct5')#Aqua Green
plt.errorbar(data2R_ct6[1], data2R_ct6[3], yerr=data2R_ct6[4], fmt='o-', color=(0.0, 1.0, 1.0),
             ecolor='lightgray', elinewidth=2, capsize=4, label='2R_ct6')#Cyan
plt.errorbar(data2R_ct7[1], data2R_ct7[3], yerr=data2R_ct7[4], fmt='o-', color=(0.0, 0.5, 1.0),
             ecolor='lightgray', elinewidth=2, capsize=4, label='2R_ct7')#Light Blue
plt.errorbar(data2R_ct8[1], data2R_ct8[3], yerr=data2R_ct8[4], fmt='o-', color=(0.0, 0.0, 1.0),
             ecolor='lightgray', elinewidth=2, capsize=4, label='2R_ct8')#Blue
plt.errorbar(data2R_ct9[1], data2R_ct9[3], yerr=data2R_ct9[4], fmt='o-', color=(0.5, 0.0, 1.0),
             ecolor='lightgray', elinewidth=2, capsize=4, label='2R_ct9')#Indigo
plt.errorbar(data2R_ct10[1], data2R_ct10[3], yerr=data2R_ct10[4], fmt='o-', color=(1.0, 0.0, 1.0),
             ecolor='lightgray', elinewidth=2, capsize=4, label='2R_ct10')"""#Magenta

#2 layer withour Res
"""plt.errorbar(data2nR_ct1[1], data2nR_ct1[3], yerr=data2nR_ct1[4], fmt='o-', color=(1.0, 0.0, 0.5),
             ecolor='lightgray', elinewidth=2, capsize=4, label='2nR_ct1')#Pink
plt.errorbar(data2nR_ct2[1], data2nR_ct2[3], yerr=data2nR_ct2[4], fmt='o-', color=(1.0, 0.75, 0.75),
             ecolor='lightgray', elinewidth=2, capsize=4, label='2nR_ct2')#Light Pink
plt.errorbar(data2nR_ct3[1], data2nR_ct3[3], yerr=data2nR_ct3[4], fmt='o-', color=(0.75, 0.75, 0.75),
             ecolor='lightgray', elinewidth=2, capsize=4, label='2nR_ct3')#Light Gray
plt.errorbar(data2nR_ct4[1], data2nR_ct4[3], yerr=data2nR_ct4[4], fmt='o-', color=(0.5, 0.5, 0.5),
             ecolor='lightgray', elinewidth=2, capsize=4, label='2nR_ct4')#Gray
plt.errorbar(data2nR_ct5[1], data2nR_ct5[3], yerr=data2nR_ct5[4], fmt='o-', color=(0.25, 0.25, 0.25),
             ecolor='lightgray', elinewidth=2, capsize=4, label='2nR_ct5')#Dark Gray
plt.errorbar(data2nR_ct6[1], data2nR_ct6[3], yerr=data2nR_ct6[4], fmt='o-', color=(0.5, 0.25, 0.0),
             ecolor='lightgray', elinewidth=2, capsize=4, label='2nR_ct6')#Brown
plt.errorbar(data2nR_ct7[1], data2nR_ct7[3], yerr=data2nR_ct7[4], fmt='o-', color=(0.75, 0.5, 0.0),
             ecolor='lightgray', elinewidth=2, capsize=4, label='2nR_ct7')#Tan
plt.errorbar(data2nR_ct8[1], data2nR_ct8[3], yerr=data2nR_ct8[4], fmt='o-', color=(0.5, 0.0, 0.5),
             ecolor='lightgray', elinewidth=2, capsize=4, label='2nR_ct8')#Purple
plt.errorbar(data2nR_ct9[1], data2nR_ct9[3], yerr=data2nR_ct9[4], fmt='o-', color=(0.75, 0.0, 0.75),
             ecolor='lightgray', elinewidth=2, capsize=4, label='2nR_ct9')#Lavender
plt.errorbar(data2nR_ct10[1], data2nR_ct10[3], yerr=data2nR_ct10[4], fmt='o-', color=(1.0, 0.5, 0.5),
             ecolor='lightgray', elinewidth=2, capsize=4, label='2nR_ct10')#Salmon"""

#d1
"""plt.errorbar(datad1_ct1[1], datad1_ct1[3], yerr=datad1_ct1[4], fmt='o-', color=(0.5, 0.0, 0.0),
             ecolor='lightgray', elinewidth=2, capsize=4, label='datad1_ct0.2_100')#Dark red
plt.errorbar(datad1_ct2[1], datad1_ct2[3], yerr=datad1_ct2[4], fmt='o-', color=(0.5, 0.5, 0.0),
             ecolor='lightgray', elinewidth=2, capsize=4, label='datad1_ct0.35_100')#Olive Green
plt.errorbar(datad1_ct3[1], datad1_ct3[3], yerr=datad1_ct3[4], fmt='o-', color=(0.0, 0.0, 0.5),
             ecolor='lightgray', elinewidth=2, capsize=4, label='datad1_ct0.5_100')#Dark Blue
plt.errorbar(datad1_ct4[1], datad1_ct4[3], yerr=datad1_ct4[4], fmt='o-', color=(1.0, 0.85, 0.0),
             ecolor='lightgray', elinewidth=2, capsize=4, label='datad1_ct0.65_100')#Gold
plt.errorbar(datad1_ct5[1], datad1_ct5[3], yerr=datad1_ct5[4], fmt='o-', color=(0.0, 0.5, 0.5),
             ecolor='lightgray', elinewidth=2, capsize=4, label='datad1_ct0.8_100')#Teal
plt.errorbar(datad1_ct6[1], datad1_ct6[3], yerr=datad1_ct6[4], fmt='o-', color=(1.0, 0.5, 0.31),
             ecolor='lightgray', elinewidth=2, capsize=4, label='datad1_ct0.95_100')#Coral
plt.errorbar(datad1_ct7[1], datad1_ct7[3], yerr=datad1_ct7[4], fmt='o-', color=(0.42, 0.35, 0.80),
             ecolor='lightgray', elinewidth=2, capsize=4, label='datad1_ct0.2_400')#Slate Blue
plt.errorbar(datad1_ct8[1], datad1_ct8[3], yerr=datad1_ct8[4], fmt='o-', color=(0.18, 0.55, 0.34),
             ecolor='lightgray', elinewidth=2, capsize=4, label='datad1_ct0.35_400')#Sea Green
plt.errorbar(datad1_ct9[1], datad1_ct9[3], yerr=datad1_ct9[4], fmt='o-', color=(0.85, 0.65, 0.13) ,
             ecolor='lightgray', elinewidth=2, capsize=4, label='datad1_ct0.5_400')#Goldenrod
plt.errorbar(datad1_ct10[1], datad1_ct10[3], yerr=datad1_ct10[4], fmt='o-', color=(0.86, 0.08, 0.24),
             ecolor='lightgray', elinewidth=2, capsize=4, label='datad1_ct0.65_400')#Crimson
plt.errorbar(datad1_ct11[1], datad1_ct11[3], yerr=datad1_ct11[4], fmt='o-', color=(0.58, 0.44, 0.86),
             ecolor='lightgray', elinewidth=2, capsize=4, label='datad1_ct0.8_400')#Medium Purple
plt.errorbar(datad1_ct12[1], datad1_ct12[3], yerr=datad1_ct12[4], fmt='o-', color=(0.76, 0.61, 0.42),
             ecolor='lightgray', elinewidth=2, capsize=4, label='datad1_ct0.95_400')#Sandy Brown
plt.errorbar(datad1_ct13[1], datad1_ct13[3], yerr=datad1_ct13[4], fmt='o-', color=(0.33, 0.42, 0.18),
             ecolor='lightgray', elinewidth=2, capsize=4, label='datad1_ct0.2_700')#Dark Olive Green
plt.errorbar(datad1_ct14[1], datad1_ct14[3], yerr=datad1_ct14[4], fmt='o-', color=(0.24, 0.70, 0.44),
             ecolor='lightgray', elinewidth=2, capsize=4, label='datad1_ct0.35_700')#Medium Sea Green
plt.errorbar(datad1_ct15[1], datad1_ct15[3], yerr=datad1_ct15[4], fmt='o-', color=(0.27, 0.51, 0.71) ,
             ecolor='lightgray', elinewidth=2, capsize=4, label='datad1_ct0.5_700')#Steel Blue
plt.errorbar(datad1_ct16[1], datad1_ct16[3], yerr=datad1_ct16[4], fmt='o-', color=(0.74, 0.56, 0.56),
             ecolor='lightgray', elinewidth=2, capsize=4, label='datad1_ct0.65_700')#Rosy Brown
plt.errorbar(datad1_ct17[1], datad1_ct17[3], yerr=datad1_ct17[4], fmt='o-', color=(0.75, 0.75, 0.75),
             ecolor='lightgray', elinewidth=2, capsize=4, label='datad1_ct0.8_700')#Light Gray
plt.errorbar(datad1_ct18[1], datad1_ct18[3], yerr=datad1_ct18[4], fmt='o-', color=(0.5, 0.5, 0.5),
             ecolor='lightgray', elinewidth=2, capsize=4, label='datad1_ct0.95_700')#Gray
plt.errorbar(datad1_ct19[1], datad1_ct19[3], yerr=datad1_ct19[4], fmt='o-', color=(0.25, 0.25, 0.25),
             ecolor='lightgray', elinewidth=2, capsize=4, label='datad1_ct0.2_1000')#Dark Gray
plt.errorbar(datad1_ct20[1], datad1_ct20[3], yerr=datad1_ct20[4], fmt='o-', color=(0.5, 0.25, 0.0),
             ecolor='lightgray', elinewidth=2, capsize=4, label='datad1_ct0.35_1000')#Brown
plt.errorbar(datad1_ct21[1], datad1_ct21[3], yerr=datad1_ct21[4], fmt='o-', color=(0.75, 0.5, 0.0),
             ecolor='lightgray', elinewidth=2, capsize=4, label='datad1_ct0.5_1000')#Tan
plt.errorbar(datad1_ct22[1], datad1_ct22[3], yerr=datad1_ct22[4], fmt='o-', color=(0.5, 0.0, 0.5),
             ecolor='lightgray', elinewidth=2, capsize=4, label='datad1_ct0.65_1000')#Purple
plt.errorbar(datad1_ct23[1], datad1_ct23[3], yerr=datad1_ct23[4], fmt='o-', color=(0.75, 0.0, 0.75),
             ecolor='lightgray', elinewidth=2, capsize=4, label='datad1_ct0.8_1000')#Lavender
plt.errorbar(datad1_ct24[1], datad1_ct24[3], yerr=datad1_ct24[4], fmt='o-', color=(1.0, 0.5, 0.5),
             ecolor='lightgray', elinewidth=2, capsize=4, label='datad1_ct0.95_1000')"""#Salmon

#distanct_d1
"""plt.errorbar(datad1_dis_ct1[1], datad1_dis_ct1[3], yerr=datad1_dis_ct1[4], fmt='o-', color=(0.5, 0.0, 0.0),
             ecolor='lightgray', elinewidth=2, capsize=4, label='d1_dis_ct0.2_100')#Dark red
plt.errorbar(datad1_dis_ct2[1], datad1_dis_ct2[3], yerr=datad1_dis_ct2[4], fmt='o-', color=(0.5, 0.5, 0.0),
             ecolor='lightgray', elinewidth=2, capsize=4, label='d1_dis_ct0.35_100')#Olive Green
plt.errorbar(datad1_dis_ct3[1], datad1_dis_ct3[3], yerr=datad1_dis_ct3[4], fmt='o-', color=(0.0, 0.0, 0.5),
             ecolor='lightgray', elinewidth=2, capsize=4, label='d1_dis_ct0.5_100')#Dark Blue
plt.errorbar(datad1_dis_ct4[1], datad1_dis_ct4[3], yerr=datad1_dis_ct4[4], fmt='o-', color=(1.0, 0.85, 0.0),
             ecolor='lightgray', elinewidth=2, capsize=4, label='d1_dis_ct0.65_100')#Gold
plt.errorbar(datad1_dis_ct5[1], datad1_dis_ct5[3], yerr=datad1_dis_ct5[4], fmt='o-', color=(0.0, 0.5, 0.5),
             ecolor='lightgray', elinewidth=2, capsize=4, label='d1_dis_ct0.8_100')#Teal
plt.errorbar(datad1_dis_ct6[1], datad1_dis_ct6[3], yerr=datad1_dis_ct6[4], fmt='o-', color=(1.0, 0.5, 0.31),
             ecolor='lightgray', elinewidth=2, capsize=4, label='d1_dis_ct0.95_100')#Coral
plt.errorbar(datad1_dis_ct7[1], datad1_dis_ct7[3], yerr=datad1_dis_ct7[4], fmt='o-', color=(0.42, 0.35, 0.80),
             ecolor='lightgray', elinewidth=2, capsize=4, label='d1_dis_ct0.2_400')#Slate Blue
plt.errorbar(datad1_dis_ct8[1], datad1_dis_ct8[3], yerr=datad1_dis_ct8[4], fmt='o-', color=(0.18, 0.55, 0.34),
             ecolor='lightgray', elinewidth=2, capsize=4, label='d1_dis_ct0.35_400')#Sea Green
plt.errorbar(datad1_dis_ct9[1], datad1_dis_ct9[3], yerr=datad1_dis_ct9[4], fmt='o-', color=(0.85, 0.65, 0.13) ,
             ecolor='lightgray', elinewidth=2, capsize=4, label='d1_dis_ct0.5_400')#Goldenrod
plt.errorbar(datad1_dis_ct10[1], datad1_dis_ct10[3], yerr=datad1_dis_ct10[4], fmt='o-', color=(0.86, 0.08, 0.24),
             ecolor='lightgray', elinewidth=2, capsize=4, label='d1_dis_ct0.65_400')#Crimson
plt.errorbar(datad1_dis_ct11[1], datad1_dis_ct11[3], yerr=datad1_dis_ct11[4], fmt='o-', color=(0.58, 0.44, 0.86),
             ecolor='lightgray', elinewidth=2, capsize=4, label='d1_dis_ct0.8_400')#Medium Purple
plt.errorbar(datad1_dis_ct12[1], datad1_dis_ct12[3], yerr=datad1_dis_ct12[4], fmt='o-', color=(0.76, 0.61, 0.42),
             ecolor='lightgray', elinewidth=2, capsize=4, label='d1_dis_ct0.95_400')#Sandy Brown
plt.errorbar(datad1_dis_ct13[1], datad1_dis_ct13[3], yerr=datad1_dis_ct13[4], fmt='o-', color=(0.33, 0.42, 0.18),
             ecolor='lightgray', elinewidth=2, capsize=4, label='d1_dis_ct0.2_700')#Dark Olive Green
plt.errorbar(datad1_dis_ct14[1], datad1_dis_ct14[3], yerr=datad1_dis_ct14[4], fmt='o-', color=(0.24, 0.70, 0.44),
             ecolor='lightgray', elinewidth=2, capsize=4, label='d1_dis_ct0.35_700')#Medium Sea Green
plt.errorbar(datad1_dis_ct15[1], datad1_dis_ct15[3], yerr=datad1_dis_ct15[4], fmt='o-', color=(0.27, 0.51, 0.71) ,
             ecolor='lightgray', elinewidth=2, capsize=4, label='d1_dis_ct0.5_700')#Steel Blue
plt.errorbar(datad1_dis_ct16[1], datad1_dis_ct16[3], yerr=datad1_dis_ct16[4], fmt='o-', color=(0.74, 0.56, 0.56),
             ecolor='lightgray', elinewidth=2, capsize=4, label='d1_dis_ct0.65_700')#Rosy Brown
plt.errorbar(datad1_dis_ct17[1], datad1_dis_ct17[3], yerr=datad1_dis_ct17[4], fmt='o-', color=(0.75, 0.75, 0.75),
             ecolor='lightgray', elinewidth=2, capsize=4, label='d1_dis_ct0.8_700')#Light Gray
plt.errorbar(datad1_dis_ct18[1], datad1_dis_ct18[3], yerr=datad1_dis_ct18[4], fmt='o-', color=(0.5, 0.5, 0.5),
             ecolor='lightgray', elinewidth=2, capsize=4, label='d1_dis_ct0.95_700')#Gray
plt.errorbar(datad1_dis_ct19[1], datad1_dis_ct19[3], yerr=datad1_dis_ct19[4], fmt='o-', color=(0.25, 0.25, 0.25),
             ecolor='lightgray', elinewidth=2, capsize=4, label='d1_dis_ct0.2_1000')#Dark Gray
plt.errorbar(datad1_dis_ct20[1], datad1_dis_ct20[3], yerr=datad1_dis_ct20[4], fmt='o-', color=(0.5, 0.25, 0.0),
             ecolor='lightgray', elinewidth=2, capsize=4, label='d1_dis_ct0.35_1000')#Brown
plt.errorbar(datad1_dis_ct21[1], datad1_dis_ct21[3], yerr=datad1_dis_ct21[4], fmt='o-', color=(0.75, 0.5, 0.0),
             ecolor='lightgray', elinewidth=2, capsize=4, label='d1_dis_ct0.5_1000')#Tan
plt.errorbar(datad1_dis_ct22[1], datad1_dis_ct22[3], yerr=datad1_dis_ct22[4], fmt='o-', color=(0.5, 0.0, 0.5),
             ecolor='lightgray', elinewidth=2, capsize=4, label='d1_dis_ct0.65_1000')#Purple
plt.errorbar(datad1_dis_ct23[1], datad1_dis_ct23[3], yerr=datad1_dis_ct23[4], fmt='o-', color=(0.75, 0.0, 0.75),
             ecolor='lightgray', elinewidth=2, capsize=4, label='d1_dis_ct0.8_1000')#Lavender
plt.errorbar(datad1_dis_ct24[1], datad1_dis_ct24[3], yerr=datad1_dis_ct24[4], fmt='o-', color=(1.0, 0.5, 0.5),
             ecolor='lightgray', elinewidth=2, capsize=4, label='d1_dis_ct0.95_1000')"""#Salmon

#d1_001
"""plt.errorbar(datad1_0011[1], datad1_0011[3], yerr=datad1_0011[4], fmt='o-', color=(1.0, 0.5, 0.0),
             ecolor='lightgray', elinewidth=2, capsize=4, label='datad1_ct0.01_100')#Orange
plt.errorbar(datad1_0012[1], datad1_0012[3], yerr=datad1_0012[4], fmt='o-', color=(1.0, 1.0, 0.0),
             ecolor='lightgray', elinewidth=2, capsize=4, label='datad1_ct0.05_100')#Yellow
plt.errorbar(datad1_0013[1], datad1_0013[3], yerr=datad1_0013[4], fmt='o-', color=(0.5, 1.0, 0.0),
             ecolor='lightgray', elinewidth=2, capsize=4, label='datad1_ct0.1_100')#Light Green
plt.errorbar(datad1_0014[1], datad1_0014[3], yerr=datad1_0014[4], fmt='o-', color=(0.0, 1.0, 0.0),
             ecolor='lightgray', elinewidth=2, capsize=4, label='datad1_ct0.01_400')#Green
plt.errorbar(datad1_0015[1], datad1_0015[3], yerr=datad1_0015[4], fmt='o-', color=(0.0, 1.0, 0.5),
             ecolor='lightgray', elinewidth=2, capsize=4, label='datad1_ct0.05_400')#Aqua Green
plt.errorbar(datad1_0016[1], datad1_0016[3], yerr=datad1_0016[4], fmt='o-', color=(0.0, 1.0, 1.0),
             ecolor='lightgray', elinewidth=2, capsize=4, label='datad1_ct0.1_400')#Cyan
plt.errorbar(datad1_0017[1], datad1_0017[3], yerr=datad1_0017[4], fmt='o-', color=(0.0, 0.5, 1.0),
             ecolor='lightgray', elinewidth=2, capsize=4, label='datad1_ct0.01_700')#Light Blue
plt.errorbar(datad1_0018[1], datad1_0018[3], yerr=datad1_0018[4], fmt='o-', color=(0.0, 0.0, 1.0),
             ecolor='lightgray', elinewidth=2, capsize=4, label='datad1_ct0.05_700')#Blue
plt.errorbar(datad1_0019[1], datad1_0019[3], yerr=datad1_0019[4], fmt='o-', color=(0.5, 0.0, 1.0),
             ecolor='lightgray', elinewidth=2, capsize=4, label='datad1_ct0.1_700')#Indigo
plt.errorbar(datad1_00110[1], datad1_00110[3], yerr=datad1_00110[4], fmt='o-', color=(1.0, 0.0, 1.0),
             ecolor='lightgray', elinewidth=2, capsize=4, label='datad1_ct0.01_1000')
plt.errorbar(datad1_00111[1], datad1_00111[3], yerr=datad1_00111[4], fmt='o-', color=(1.0, 0.0, 0.5),
             ecolor='lightgray', elinewidth=2, capsize=4, label='datad1_ct0.05_1000')#Pink
plt.errorbar(datad1_00112[1], datad1_00112[3], yerr=datad1_00112[4], fmt='o-', color=(1.0, 0.75, 0.75),
             ecolor='lightgray', elinewidth=2, capsize=4, label='datad1_ct0.1_1000')"""#Light Pink



plt.title('Accuracy vs Number of Neurons', fontsize=16)
plt.xlabel('Number of Neurons(second layer)', fontsize=14)
plt.ylabel('Accuracy', fontsize=14)
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.grid()
plt.savefig('Accuracy vs Number of Neurons.png', dpi=300, bbox_inches='tight')
plt.show()