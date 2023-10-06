import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep
hep.style.use('LHCb2')

df = pd.read_csv('C:/Users/yt21461/CR code/CR_MSci_Lab/Run0001.csv')

for ch in ['A','B','C','D','E','F','G','H']:
    h, bins = np.histogram(df[f'pulse_height_{ch}'],range=[0,1],bins=100)
    hep.histplot(h,bins,yerr=np.sqrt(h),histtype='step',label=f'Channel {ch}')

plt.legend()
plt.xlabel('Peak of channel (V))',fontsize=32)
plt.ylabel('Counts',fontsize=32)
plt.savefig('peak_distbn.pdf')

thA = 0.1
thB = 0.1
thC = 0.1
thD = 0.01
thE = 0.1
thF = 0.1
thG = 0.1
thH = 0.01


df_BDF = df.query(f"pulse_height_B>{thB} & pulse_height_D>{thD} & pulse_height_F>{thF} & pulse_height_H<{thH}")
df_BDH = df.query(f"pulse_height_B>{thB} & pulse_height_D>{thD} & pulse_height_H>{thH} & pulse_height_F<{thF}")
df_BFH = df.query(f"pulse_height_B>{thB} & pulse_height_F>{thF} & pulse_height_H>{thH} & pulse_height_D<{thD}")
df_DFH = df.query(f"pulse_height_D>{thD} & pulse_height_F>{thF} & pulse_height_H>{thH} & pulse_height_B<{thB}")
df_BDFH = df.query(f"pulse_height_D>{thD} & pulse_height_F>{thF} & pulse_height_H>{thH} & pulse_height_B>{thB}")

print(f"Number of BDF triggers and !H: {df_BDF.shape[0]}")
print(f"Number of BDH triggers and !F: {df_BDH.shape[0]}")
print(f"Number of BFH triggers and !D: {df_BFH.shape[0]}")
print(f"Number of DFH triggers and !B: {df_DFH.shape[0]}")
print(f"Number of BDFH triggers: {df_BDFH.shape[0]}")
print(f"Total number of triggers: {df.shape[0]}")