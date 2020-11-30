# Load the dataset
# ToyotaCorolla.csv
data <- read.csv(file.choose())
View(data)

library(dplyr)
data <- data[,c("Price","Age_08_04","KM","HP","cc","Doors","Gears","Quarterly_Tax","Weight")]

#rename columns not for computer_data.csv
data <- data %>% rename(price = Price, age = Age_08_04, km = KM, hp = HP, cc= cc, doors = Doors, gears = Gears, qtax = Quarterly_Tax, weight = Weight) 

attach(data)

# price
# Normal distribution
qqnorm(price)
qqline(price)
# direction = +ve, strength = moderate, linearity = non linear

summary(data)
# Scatter plot | Plot relation ships between each X with Y
# age, price | qtax, price

plot(age, price) 
# direction = -ve, strength = moderate, linearity = non linear

plot(qtax, price)
# direction = +ve, strength = moderate, linearity = non linear

# Or make a combined plot | Scatter plot for all pairs of variables
pairs(data)   
plot(data)

cor(age, price)
# correlation matrix

cor(data) 
# weight-QuarterlyTax = high colinearity

# The Linear Model of interest | lm(Y ~ X)
model.data <- lm(price ~ age + km + hp + cc + doors + gears + qtax + weight, data = data)
summary(model.data)
# r-squared = 0.86, f value = significant, cc, doors insignificant > 0.05

# Scatter plot matrix with Correlations inserted in graph
library(GGally)
ggpairs(data)

# Partial Correlation matrix
library(corpcor)
cor(data)
cor2pcor(cor(data))

# Diagnostic Plots
library(car)

#Residual Plots, QQ-Plot, Std. Residuals vs Fitted, Cook's distance
plot(model.data)

#QQ plots of studentized residuals, helps identify outliers
qqPlot(model.data, id.n = 5)

# Deletion Diagnostics for identifying influential obseravations
# Index Plots of the influence measures
# A user friendly representation of the above

influenceIndexPlot(model.data, id.n = 3) 
influencePlot(model.data, id.n = 3) 
# 81 most influential observation

# Regression after deleting the observation
model.data1 <- lm(price ~ age + km + hp + cc + doors + gears + qtax + weight , data = data[-81, ])
summary(model.data1)
# r-squared = 0.86, f value = significant, doors insignificant > 0.05

# Variance Inflation Factors | VIF is > 10 => collinearity
vif(model.data1)   

# Regression model to check R^2 on Independent variales
VIF1 <- lm(price ~ age + km + hp + cc + doors + gears + qtax + weight )
VIF2 <- lm(age ~ price + km + hp + cc + doors + gears + qtax + weight )
VIF3 <- lm(km ~ age + price + hp + cc + doors + gears + qtax + weight )
VIF4 <- lm(hp ~ age + km + price + cc + doors + gears + qtax + weight )
VIF5 <- lm(cc ~ age + km + hp + price + doors + gears + qtax + weight )
VIF6 <- lm(doors ~ age + km + hp + cc + price + gears + qtax + weight )
VIF7 <- lm(gears ~ age + km + hp + cc + doors + price + qtax + weight )
VIF8 <- lm(qtax ~ age + km + hp + cc + doors + gears + price + weight )
VIF9 <- lm(weight ~ age + km + hp + cc + doors + gears + qtax + price )

summary(VIF1)
summary(VIF2)
summary(VIF3)
summary(VIF4)
summary(VIF5)
summary(VIF6)
summary(VIF7)
summary(VIF8)
summary(VIF9)

vif_rtable <- matrix(c(summary(VIF1)$r.squared,summary(VIF2)$r.squared,summary(VIF3)$r.squared,summary(VIF4)$r.squared,summary(VIF5)$r.squared,summary(VIF6)$r.squared,summary(VIF7)$r.squared,summary(VIF8)$r.squared,summary(VIF9)$r.squared),ncol=1,byrow=TRUE)
colnames(vif_rtable) <- c("R-squared")
rownames(vif_rtable) <- c("VIF1","VIF2","VIF3","VIF4","VIF5","VIF6","VIF7","VIF8","VIF9")
vif_rtable <- as.table(vif_rtable)
vif_rtable

# Added Variable Plots 
avPlots(model.data, id.n = 2, id.cex = 0.8, col = "red")

# Linear Model 
model.final <- lm(price ~ age + km + hp + cc + doors + gears + qtax + weight, data = data)
summary(model.final)
# r-squared = 0.86, f value = significant, cc, doors insignificant > 0.05

# Linear model without influential observation
model.final1 <- lm(price ~ age + km + hp + cc + gears + qtax + weight, data = data[-81, ])
summary(model.final1)
# r-squared = 0.86, f value = significant

rtable <- matrix(c(summary(model.data)$r.squared,summary(model.data1)$r.squared,summary(model.final)$r.squared,summary(model.final1)$r.squared),ncol=1,byrow=TRUE)
colnames(rtable) <- c("R-squared")
rownames(rtable) <- c("model1","model2","model3","model4")
rtable <- as.table(rtable)
rtable

# Added Variable Plots
avPlots(model.final1, id.n = 2, id.cex = 0.8, col = "red")

# Variance Influence Plot
vif(model.final1)

# Evaluation Model Assumptions
plot(model.final1)
plot(model.final1$fitted.values, model.final1$residuals)

qqnorm(model.final1$residuals)
qqline(model.final1$residuals)

library(leaps)
lm_best <- regsubsets(price ~ ., data = data, nvmax = 15)
summary(lm_best)
summary(lm_best)$adjr2
which.max(summary(lm_best)$adjr2)
coef(lm_best, 3)

lm_forward <- regsubsets(price ~ ., data = data, nvmax = 15, method = "forward")
summary(lm_forward)

# Data Partitioning
n <- nrow(data)
n1 <- n * 0.8
n2 <- n - n1
train <- sample(1:n, n1)
test <- data[-train, ]

# Model Training
# price ~ age + km + hp + cc + gears + qtax + weight
model <- lm(price ~ age + km + hp + cc + gears + qtax + weight, data[train, ])
summary(model)

pred <- predict(model, newdata = test)
actual <- test$price
error <- actual - pred

test.rmse <- sqrt(mean(error**2))
test.rmse

train.rmse <- sqrt(mean(model$residuals**2))
train.rmse
# train = 1305.96, test = 1471.66