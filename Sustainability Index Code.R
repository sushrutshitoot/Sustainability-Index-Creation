#- Project: Sustainability Index -#
#- Date: 24th April 2023 -#
#- Author: Sushrut Shitoot -#


#- Free up space in the memory -#
rm (list = ls(all = TRUE))
gc ()

#- Set the working directory -#
setwd ("D:/Freelance/Sustainability Index/Working")
getwd ()

#- Import the dataset into the RStudio environment -#
Data <- read.csv(file = "Input_Scaled.csv", header = TRUE)
Data_Backup <- Data
str(Data)


#- Change the data type / class of the variables -#
Data$ID <- as.factor(Data$ID)
Data$var1 <- as.numeric(Data$var1)
Data$var2 <- as.factor(Data$var2)
Data$var3 <- as.factor(Data$var3)
Data$var4 <- as.factor(Data$var4)
Data$var5 <- as.factor(Data$var5)
Data$var6 <- as.factor(Data$var6)
Data$var7 <- as.numeric(Data$var7)
Data$var8 <- as.factor(Data$var8)
Data$var9 <- as.factor(Data$var9)
Data$var10 <- as.factor(Data$var10)
Data$var11 <- as.numeric(Data$var11)
Data$var12 <- as.numeric(Data$var12)
Data$var13 <- as.numeric(Data$var13)
Data$var14 <- as.numeric(Data$var14)
Data$var15 <- as.factor(Data$var15)
Data$var16 <- as.factor(Data$var16)
Data$var17 <- as.factor(Data$var17)
Data$var18 <- as.numeric(Data$var18)
Data$var19 <- as.numeric(Data$var19)
Data$var20 <- as.numeric(Data$var20)
Data$var21 <- as.factor(Data$var21)
Data$var22 <- as.numeric(Data$var22)
Data$var23 <- as.numeric(Data$var23)
Data$var24 <- as.numeric(Data$var24)
Data$var25 <- as.numeric(Data$var25)
Data$var26 <- as.numeric(Data$var26)
Data$var27 <- as.numeric(Data$var27)
Data$var28 <- as.numeric(Data$var28)
Data$var29 <- as.numeric(Data$var29)
Data$var30 <- as.numeric(Data$var30)
Data$Y <- as.numeric(Data$Y)

str(Data)


#- Univariate Analysis -#
#- install.packages("funModeling")
library(funModeling)

#- Look at the number and percentage of zeroes and missing values in the data -#
Univariate1 <- df_status(Data)

#- 1.2 Checking mean, min, max, median, sd of variables -#
Univariate2 <- do.call (data.frame,
                        list (mean = vapply(Data, function(x) mean(x[!(is.na(x))]), numeric(1)),
                              median = apply(Data, 2, median, na.rm = TRUE),
                              min = apply(Data, 2, min, na.rm = TRUE),
                              max = apply(Data, 2, max, na.rm = TRUE),
                              sd = apply(Data, 2, sd, na.rm = TRUE)
                        ))


#- Writing the Univariate analysis (1.1 + 1.2) for future analysis and reference
Univariate <- cbind (
  var = Univariate1$variable,
  type = Univariate1$type,
  q_zeros = Univariate1$q_zeros,
  p_zeros = Univariate1$p_zeros,
  q_na = Univariate1$q_na,
  p_na = Univariate1$p_na,
  mean = Univariate2$mean,
  median = Univariate2$median,
  min = Univariate2$min,
  max = Univariate2$max,
  sd = Univariate2$sd
)

write.csv (Univariate, file = "Univariate Analysis.csv", row.names = TRUE)

#- Calculate a flag that tests the variance of the variable and determines whether it is zero or near zero -#
#- This is important since the variable with zero or near zero variance do not contribute a lot of the explainability of the model and dependent variable -#
#- install.packages("caret")
library (caret)
nzv_details = nearZeroVar(Data, freqCut = 95/5, uniqueCut = 10, saveMetrics = TRUE)
write.csv (nzv_details, file = "Univariate_NearZerovariance.csv", row.names = TRUE)

library(dplyr)
library(ggplot2)

#- Exploratory Data Analysis -#
#- Correlation between independent variables and the dependent variable -#
#install.packages("corrplot")
library(corrplot)

#- Create a subset of the dataset to include only the numeric variables for correlation -#
Numeric_variables <- unlist(lapply(Data, is.numeric))

Data_for_correlation <- Data[ , Numeric_variables]   

CorrelationMatrix <- cor(Data_for_correlation)

#- Plot the correlation matrix with aesthetic tweaking for best look -#
corrplot(CorrelationMatrix, method = "square", main = "Correlation Matrix", mar = c(0,0,2,0), outline = T, addgrid.col = "darkgrey", addrect = 2, rect.col = "black", cl.pos = "r", tl.col = "indianred4", tl.cex = 1, cl.cex = 0.75, tl.srt = 45, col = colorRampPalette(c("darkred", "white","midnightblue"))(100))
#mtext("Correlation Matrix", at=11, line=0.0005, cex=1.2)

write.csv(CorrelationMatrix, "Correlation Matrix.csv", row.names = TRUE)

#
str(Data)

#- Feature Selection using Boruta -#
library (Boruta)
library(caret)
library(randomForest)

#- Implement the Boruta algorithm to identify the most important variables -#
set.seed(123)
Boruta.train <- Boruta(Y ~. -ID, data = Data, doTrace = 1, maxRuns = 200)
print(Boruta.train)

#- Plot the variable importance and decision together on one chart -#
plot(Boruta.train, main = "Variable Importance")

Boruta.Dataframe <- attStats(Boruta.train)
print(Boruta.Dataframe)
write.csv(Boruta.Dataframe, "Boruta Output.csv", row.names = TRUE)

Boruta_ConfirmedVariables <- getSelectedAttributes(Boruta.train, withTentative = FALSE)
write.csv(Boruta_ConfirmedVariables, "Boruta_ConfirmedVariables.csv", row.names = TRUE)

Boruta_Confirmed_TentativeVariables <- getSelectedAttributes(Boruta.train, withTentative = TRUE)
write.csv(Boruta_Confirmed_TentativeVariables, "Boruta_Confirmed_TentativeVariables.csv", row.names = TRUE)

#- Extracting the list of final predictors from Boruta that can be used as input for building the model -#
Predictor.names_Boruta <- getSelectedAttributes(Boruta.train, withTentative = FALSE)
Predictor.names_Boruta

#- Feature Formula -#
Predictor.names_Boruta
Predictors.formula.Boruta <- formula(paste('Y ~', paste(Predictor.names_Boruta, collapse = '+'), sep = ''))
Predictors.formula.Boruta


#- Model Building -#

library(caret)
library(Matrix)



#- Building a linear regression model with confirmed features from Boruta -#
set.seed(1234)
LinearModel <- lm(Predictors.formula.Boruta, data = Data)

summary(LinearModel)

#- Extract coefficients from the linear model -#
LinearModelCoefficients <- as.data.frame(LinearModel$coefficients)
write.csv(LinearModelCoefficients, file = "Linear Model Coefficients.csv", row.names = TRUE)


#- Build a Random Forest model to compare findings -#
SearchGrid <- expand.grid(mtry = c(3:15))


#- Define the parameters for cross-validation -#
fitControl <- trainControl(
  method = "loocv",
  verboseIter = TRUE,
  classProbs = FALSE,
  savePredictions = TRUE)


Predictors.formula.Boruta

#- Use the following command to understand what command to use for which modeling technique -#
#- names(getModelInfo())

set.seed(1234)
Model.fit <- train(
  Predictors.formula.Boruta, #Individual variable names can also be used for better iteration purposes at a later step
  data = Data,
  method = "rf",
  trControl = fitControl,
  verbose = TRUE,
  maximize = FALSE,
  tuneGrid = SearchGrid,
  metric = "RMSE")

Model.fit
summary(Model.fit)
VariableImportance <- varImp(Model.fit)
plot(VariableImportance, main = "Variable Importance RF")

write.csv(VariableImportance$importance, file = "Variable Importance RF.csv", row.names = TRUE)

plot(Model.fit)

#

