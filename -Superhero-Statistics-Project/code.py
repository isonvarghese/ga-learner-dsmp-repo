# --------------
#Header files
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#path of the data file- path
data=pd.read_csv(path)
print(data.head())

#Code starts here 

data['Gender'].replace('-','Agender',inplace=True)
#print(data[data['Gender']=='Agender'])

gender_count=data['Gender'].value_counts()

plt.bar(gender_count,height=500,width=20)
plt.xlabel('Gender Counts')
plt.ylabel('Frequencies')
plt.title('Bar plot for Gender count')



# --------------
#Code starts here
alignment=data['Alignment'].value_counts()
print(alignment)
labels=['Good','Bad','Neutral']
plt.pie(alignment,labels=labels)
plt.xlabel('Character Alignment')


# --------------
#Code starts here
#for Strength and Combat
sc_df=data[['Strength','Combat']]
a=sc_df.cov()
sc_covariance=a.iloc[0,1]
print(sc_covariance)
sc_strength=sc_df['Strength'].std()

sc_combat=sc_df['Combat'].std()

sc_pearson=sc_covariance/(sc_strength*sc_combat)
print(sc_pearson)
#for Intelligence and Combat
ic_df=data[['Intelligence','Combat']]

b=ic_df.cov()
ic_covariance=b.iloc[0,1]
ic_intelligence=ic_df['Intelligence'].std()

ic_combat=ic_df['Combat'].std()

ic_pearson=ic_covariance/(ic_intelligence*ic_combat)
print(ic_pearson)



# --------------
#Code starts here
total_high=data['Total'].quantile(0.99)
print(total_high)

super_best=data[data['Total']>total_high]
print(super_best)

super_best_names=list(super_best['Name'])
print(super_best_names)


# --------------
#Code starts here
fig,(ax_1,ax_2,ax_3)=plt.subplots(3,1,figsize=(20,20))

ax_1.boxplot(data.Intelligence)

ax_2.boxplot(data.Speed)

ax_3.boxplot(data.Power)






