
# coding: utf-8

# In[43]:

from numpy import *
import pandas as pd


# In[76]:

#load data file
data_file = pd.read_excel(r'C:\Users\WRIGHP10\Desktop\Test_ASAP_data.xlsx')
print(data_file)
type(data_file)
print(data_file.ix[6,'Temp/ºC'])


# In[77]:

#data processing
#http://stackoverflow.com/questions/15315452/selecting-with-complex-criteria-from-pandas-dataframe
Initial_deg_criterion = data_file.loc[(data_file['Duration']=='Initial')] # locate all initial data
non_t0_data_file = data_file.loc[(data_file['Duration']!='Initial')] # need to create to not have initial in the time column
Time = non_t0_data_file['Duration']#all non_initial data
Temperature = non_t0_data_file['Temp/ºC']#all non_initial data
Recipricol_temperature = (1/(Temperature+273.15))
Humidity = non_t0_data_file['%RH']#all non_initial data
Degradation = non_t0_data_file['% Deg']#all non_initial data
Average_initial_deg = np.mean(Initial_deg_criterion['% Deg'])

#zero order
Normalized_degradation = Degradation - Average_initial_deg
Degradation_rate = Normalized_degradation/Time
Natural_log_deg_rate = np.log(Degradation_rate.astype('float64')) # need to convert dataframe to numeric to .log
print(Natural_log_deg_rate)


# In[132]:

#multiple linear regression section
df = pd.DataFrame({"A": Natural_log_deg_rate, "B": Recipricol_temperature.astype('float64'), "C": Humidity.astype('float64')})
print(df)
import statsmodels.formula.api as sm
import statsmodels.api as sd
result = sm.ols(formula="A ~ B + C", data=df).fit()
Intercept = result.params[0]
Activation_Energy = (result.params[1]*-8.314)/1000
B = result.params[2]
R_squared = result.rsquared
Results = pd.DataFrame({"Ea (KJ/mol)": Activation_Energy , "ln(A) Intercept": Intercept , "Humidity_coefficient": B,"R^2": [R_squared]})
print(Results)
Predicted_Deg = np.exp(result.predict())*Time + Average_initial_deg

#standard parameter errors
print('Standard errors: ', result.bse)



#dropping an observation: http://statsmodels.sourceforge.net/devel/examples/notebooks/generated/ols.html (bottom)
infl = result.get_influence()
print(infl.summary_frame().filter(regex="dfb"))

#plot data
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
plt.scatter(Degradation,Predicted_Deg)
plt.title('Observed deg vs. Predicted deg')
#plt.bar( 1,2 ,yerr=result.bse)
plt.xlabel('Observed deg')
plt.ylabel('Predicted deg')
plt.show()



    
#from statsmodels.sandbox.regression.predstd import wls_prediction_std
#import matplotlib.pyplot as plt
#fig, ax = plt.(Natural_log_deg_rate, result.predict)


#from sklearn import linear_model
#http://stackoverflow.com/questions/19991445/run-an-ols-regression-with-pandas-data-frame
#result = linear_model.LinearRegression()
#result.fit(df[['B', 'C']], df['A'])
#Activation_Energy = (result.coef_[0]*-8.314)/1000
#B = result.coef_[1]
#A = result.intercept_
#Results = pd.DataFrame({"Ea (KJ/mol)": Activation_Energy , "ln(A) Intercept": [A] , "Humidity_coefficient": B})
#print(Results)
#Results.to_csv(r'C:\Users\WRIGHP10\Documents\Python Scripts\Results.csv')
#type(result.intercept_)


# In[ ]:



