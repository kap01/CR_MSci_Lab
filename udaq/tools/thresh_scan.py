import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep
import os
hep.style.use('LHCb2')



os.sys('python .\udaq\src\udaq.py .\udaq\tools\udaq_single.config')
df = pd.read_csv('C:/Users/yt21461/CR code/CR_MSci_Lab/Run0001.csv')

