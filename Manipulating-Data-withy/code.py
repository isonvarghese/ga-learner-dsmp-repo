# --------------
#Code starts here

import numpy as np
age=census[:,0]
max_age=np.max(age)
min_age=np.min(age)
age_mean=np.sum(age)/1001
age_std=np.std(age)


# --------------
# Importing header files
import numpy as np

# Path of the file has been stored in variable called 'path'
data=np.genfromtxt(path,delimiter=",",skip_header=1)
print(data)
print(type(data))
#New record
new_record=[[50,  9,  4,  1,  0,  0, 40,  0]]
census=np.concatenate((data,new_record),axis=0)
print(census)
#Code starts here



# --------------
#Code starts here
import numpy as np
senior_citizens=census[census[:,0]>60]

working_hours_sum=senior_citizens[:,6].sum()

senior_citizens_len=len(senior_citizens)

avg_working_hours=working_hours_sum/senior_citizens_len

print(avg_working_hours)


# --------------
#Code starts here
high=census[census[:,1]>10]
low=census[census[:,1]<=10]

avg_pay_high=high[:,7].mean()
avg_pay_low=low[:,7].mean()
avg_pay_low




# --------------
#Code starts here
import numpy as np
race=census[:,2]
race_0=race[np.array(np.where(race==0))]
race_1=race[np.array(np.where(race==1))]
race_2=race[np.array(np.where(race==2))]
race_3=race[np.array(np.where(race==3))]
race_4=race[np.array(np.where(race==4))]

#length of above created
len_0=race_0.size
len_1=race_1.size
len_2=race_2.size
len_3=race_3.size
len_4=race_4.size

minority_race=len_3/2
print(minority_race)


