# --------------
#Code starts here
import matplotlib.pyplot as plt

education_and_loan=data.groupby(['Education','Loan_Status'])
education_and_loan=education_and_loan.size().unstack()

education_and_loan.plot(kind='bar',stacked=True,figsize=(15,10))

plt.xlabel('Education Status')
plt.ylabel('Loan Status')
plt.xticks(rotation=45)


# --------------
#Code starts here
import matplotlib.pyplot as plt

fig ,(ax_1,ax_2,ax_3)=plt.subplots(nrows=3,ncols=1)
ax_1=plt.scatter(data['ApplicantIncome'],data['LoanAmount'])
plt.title('Applicant Income')

ax_2=plt.scatter(data['CoapplicantIncome'],data['LoanAmount'])
plt.title('Coapplicant Income')

data['TotalIncome']=data.ApplicantIncome+data.CoapplicantIncome

ax_3=plt.scatter(data['ApplicantIncome'],data['LoanAmount'])
plt.title('Total Income')





# --------------
#Importing header files
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt




#Code starts here

data=pd.read_csv(path)
print(data.head())

loan_status=data['Loan_Status'].value_counts()
print(loan_status)
#plt.bar(loan_status)


# --------------
#Code starts here
import matplotlib.pyplot as plt

property_and_loan=data.groupby(['Property_Area','Loan_Status'])

property_and_loan=property_and_loan.size().unstack()

property_and_loan.plot(kind='bar',stacked=False,figsize=(15,10))

plt.xlabel('Property Area')
plt.ylabel('Loan Status')
plt.xticks(rotation=45)


# --------------
#Code starts here
import pandas as pd
graduate=data[data['Education']=='Graduate']
print(type(graduate))
not_graduate=data[data['Education']=='Not Graduate']

graduate.plot(kind='density',label='Graduate')

not_graduate.plot(kind='density',label='Not Graduate')












#Code ends here

#For automatic legend display
plt.legend()


