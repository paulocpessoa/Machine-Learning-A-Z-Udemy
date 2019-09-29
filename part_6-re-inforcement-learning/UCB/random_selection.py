# Random Selection

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('C:\git\Machine-Learning-A-Z-Udemy\data_files\Ads_CTR_Optimisation.csv')

# Implementing Random Selection
import random
N = 10000
d = 10
ads_selected = []
total_reward = 0

for n in range(N):
    random_ad = random.randrange(d)
    ads_selected.append(random_ad)
    reward = dataset.values[n, random_ad]
    total_reward += reward


print(total_reward)
# Visualising the results
plt.hist(ads_selected)
plt.title('Histogram of ads selections')
plt.xlabel('Ads')
plt.ylabel('Number of times each ad was selected')
plt.show()