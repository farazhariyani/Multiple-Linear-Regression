import pandas as pd
import numpy as np

# loading the data
data = pd.read_csv("ToyotaCorolla.csv")
data.describe()

data_new = data.filter(['Price','Age_08_04','KM','HP','cc','Doors','Gears','Quarterly_Tax','Weight'], axis = 1)
data_new.columns = "Price","Age","KM","HP","cc","Doors","Gears","QuarterlyTax","Weight"
data_new.describe()

#Graphical Representation
import matplotlib.pyplot as plt

# Price
data_new.Price.plot.bar()
plt.hist(data_new.Price) #histogram
plt.boxplot(data_new.Price) #boxplot
# 1 peak, many outliers

# HP
data_new.HP.plot.bar()
plt.hist(data_new.HP) #histogram
plt.boxplot(data_new.HP) #boxplot
# 1 peak, some outliers

# Jointplot
import seaborn as sns

# Price, Age
sns.jointplot(x=data_new['Price'], y=data_new['Age'])
# direction = -ve, strength = moderate, linearity = non linear

# Countplot - 1 figure, 16 = height, 10 = width
plt.figure(1, figsize=(16, 10))
sns.countplot(data_new['Age'])

# Q-Q Plot
from scipy import stats
import pylab
stats.probplot(data_new.Price, dist = "norm", plot = pylab)
plt.show()
# non linear

# Scatter plot between the variables along with histograms
import seaborn as sns
sns.pairplot(data_new.iloc[:, :])

# Correlation matrix 
data_new.corr()
# weight-QuarterlyTax = high colinearity

# preparing model considering all the variables 
import statsmodels.formula.api as smf # for regression model

ml1 = smf.ols('Price ~ Age + KM + HP + cc + Doors + Gears + QuarterlyTax + Weight', data = data_new).fit() 
ml1.summary()
# r-squared = 0.86, f value = significant, cc, doors insignificant > 0.05

# Checking whether data has any influential values 
# Influence Index Plots
import statsmodels.api as sm

sm.graphics.influence_plot(ml1)
# 80 most influential observation

# Studentized Residuals = Residual/standard deviation of residuals
#drop influential observation
data_new = data_new.drop(data_new.index[[80]])

# Preparing model                  
ml_new = smf.ols('Price ~ Age + KM + HP + cc + Doors + Gears + QuarterlyTax + Weight', data = data_new).fit()    
ml_new.summary()
# r-squared = 0.86, f value = significant, doors insignificant > 0.05

# Check for Colinearity to decide to remove a variable using VIF
# Assumption: VIF > 10 = colinearity
# calculating VIF's values of independent variables
#figure which variable has highest vif value
rsq_1 = smf.ols('Price ~ Age + KM + HP + cc + Doors + Gears + QuarterlyTax + Weight', data = data_new).fit().rsquared  
vif_1 = 1/(1 - rsq_1) 

rsq_2 = smf.ols('KM ~ Age + Price + HP + cc + Doors + Gears + QuarterlyTax + Weight', data = data_new).fit().rsquared  
vif_2 = 1/(1 - rsq_2)

rsq_3 = smf.ols('HP ~ Age + KM + Price + cc + Doors + Gears + QuarterlyTax + Weight', data = data_new).fit().rsquared  
vif_3 = 1/(1 - rsq_3) 

rsq_4 = smf.ols('cc ~ Age + KM + HP + Price + Doors + Gears + QuarterlyTax + Weight', data = data_new).fit().rsquared  
vif_4 = 1/(1 - rsq_4) 

rsq_5 = smf.ols('Doors ~ Age + KM + HP + cc + Price + Gears + QuarterlyTax + Weight', data = data_new).fit().rsquared  
vif_5 = 1/(1 - rsq_5) 

rsq_6 = smf.ols('Gears ~ Age + KM + HP + cc + Doors + Price + QuarterlyTax + Weight', data = data_new).fit().rsquared  
vif_6 = 1/(1 - rsq_6) 

rsq_7 = smf.ols('QuarterlyTax ~ Age + KM + HP + cc + Doors + Gears + Price + Weight', data = data_new).fit().rsquared  
vif_7 = 1/(1 - rsq_7) 

rsq_8 = smf.ols('Weight ~ Age + KM + HP + cc + Doors + Gears + QuarterlyTax + Price', data = data_new).fit().rsquared  
vif_8 = 1/(1 - rsq_8) 

rsq_9 = smf.ols('Age ~ Price + KM + HP + cc + Doors + Gears + QuarterlyTax + Weight', data = data_new).fit().rsquared  
vif_9 = 1/(1 - rsq_9) 

# Storing vif values in a data frame
d1 = {'Variables':['Price', 'KM', 'HP', 'cc', 'Doors', 'Gears', 'QuarterlyTax', 'Weight', 'Age'], 'VIF':[vif_1, vif_2, vif_3, vif_4, vif_5, vif_6, vif_7, vif_8, vif_9]}
Vif_frame = pd.DataFrame(d1)  
Vif_frame
# Price highest vif
# we are going to drop the column having highest vif from the prediction model

# Final model
final_ml = smf.ols('QuarterlyTax ~ KM + HP + cc + Doors + Gears + Age + Weight', data = data_new).fit()
final_ml.summary() 
# r-squared = 0.65, f value = significant

# Prediction
pred = final_ml.predict(data_new)

# Q-Q plot
res = final_ml.resid

# Q-Q plot
stats.probplot(res, dist = "norm", plot = pylab)
plt.show()
# not normally distributed

# Residuals vs Fitted plot
sns.residplot(x = pred, y = data_new.Price, lowess = True)
plt.xlabel('Fitted')
plt.ylabel('Residual')
plt.title('Fitted vs Residual')
plt.show()

sm.graphics.influence_plot(final_ml)
# 221 most influential observation

# Splitting the data into train and test data 
from sklearn.model_selection import train_test_split
data_train, data_test = train_test_split(data_new, test_size = 0.2) 

# preparing the model on train data 
# QuarterlyTax ~ KM + HP + cc + Doors + Gears + Age + Weight
model_train = smf.ols("QuarterlyTax ~ KM + HP + cc + Doors + Gears + Age + Weight", data = data_train).fit()

# prediction on test data set 
test_pred = model_train.predict(data_test)
# test residual values 
test_resid = test_pred - data_test.QuarterlyTax
# RMSE value for test data 
test_rmse = np.sqrt(np.mean(test_resid * test_resid))
test_rmse

# train_data prediction
train_pred = model_train.predict(data_train)

# train residual values 
train_resid  = train_pred - data_train.QuarterlyTax
# RMSE value for train data 
train_rmse = np.sqrt(np.mean(train_resid * train_resid))
train_rmse
# training = 23.43,testing = 27.93