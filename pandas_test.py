import matplotlib.pyplot as plt
import pandas as pd

df = pd.DataFrame()
plt.figure();
df.ix[5].plot.bar();
plt.axhline(0, color='k')