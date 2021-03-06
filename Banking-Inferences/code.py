# --------------
import pandas as pd
import scipy.stats as stats
import math
import numpy as np
import warnings

warnings.filterwarnings('ignore')
#Sample_Size
sample_size=2000

#Z_Critical Score
z_critical = stats.norm.ppf(q = 0.95)  


# path        [File location variable]
data=pd.read_csv(path)
#Code starts here
data_sample=data.sample(n=sample_size,random_state=0)
print(data_sample.head())
sample_mean=data_sample['installment'].mean()

sample_std=data_sample['installment'].std()

margin_of_error=z_critical*(sample_std/math.sqrt(sample_size))
print(margin_of_error)

confidence_interval=(sample_mean-margin_of_error),(sample_mean+margin_of_error)
print(confidence_interval)

true_mean=data['installment'].mean()
print(true_mean)
if true_mean>confidence_interval[0] and true_mean<confidence_interval[1]:
    print('It falls in the range of confidence interval')
else:
    print('it does not falls in the range of confidnce interval')



# --------------
import matplotlib.pyplot as plt
import numpy as np

#Different sample sizes to take
sample_size=np.array([20,50,100])

#Code starts here
fig,axes=plt.subplots(3,1)


for i in range(len(sample_size)):
    m=[]
    for j in range(0,1000):
        m.append(data.sample(n=sample_size[i])['installment'].mean())
    mean_series=pd.Series(m)
    axes[i].hist(mean_series)



# --------------
#Importing header files

from statsmodels.stats.weightstats import ztest

#Code starts here
#data.rename(columns=lambda x: x.replace('.', '_'), inplace=True)
data['int.rate']=[float(x.strip('%')) for x in data['int.rate']] 
data['int.rate']=data['int.rate']/100
x1=data[data['purpose']=='small_business']['int.rate']
value=data['int.rate'].mean()
z_statistic,p_value=ztest(x1,value=value,alternative='larger')
if p_value<0.05:
    print('Reject Null Hypothesis')
elif p_value>0.05:
    print('Cannot Reject Null Hypothesis')



# --------------
#Importing header files
from statsmodels.stats.weightstats import ztest

#Code starts here
z_statistic,p_value=ztest(x1=data[data['paid.back.loan']=='No']['installment'],x2=data[data['paid.back.loan']=='Yes']['installment'])
if p_value<0.05:
    inference='Reject'
    print(inference)
else:
    inference='Accept'
    print(inference)


# --------------
#Importing header files
from scipy.stats import chi2_contingency

#Critical value 
critical_value = stats.chi2.ppf(q = 0.95, # Find the critical value for 95% confidence*
                      df = 6)   # Df = number of variable categories(in purpose) - 1

#Code starts here
yes=data[data['paid.back.loan']=='Yes']['purpose'].value_counts()
no=data[data['paid.back.loan']=='No']['purpose'].value_counts()
observed=pd.concat([yes.transpose(),no.transpose()],axis=1,keys=['Yes','No'])
print(observed)
chi2,p,dof,ex=chi2_contingency(observed)
if chi2>critical_value:
    inference='Reject'
    print(inference)
else:
    inference='Accept'
    print(inference)


