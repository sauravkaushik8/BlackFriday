library('plyr')
library('dummies')
library('scales')
library('caret')

train<-read.csv("train.csv",stringsAsFactors = F)
test<-read.csv("test-comb.csv",stringsAsFactors = F)


X_train <- subset(train, !Product_Category_1 %in% c(19,20))
X_test <- test

X_train <- dummy.data.frame(X_train, names=c("City_Category"), sep="_")
X_test <- dummy.data.frame(X_test, names=c("City_Category"), sep="_")

X_train$Age[X_train$Age == "0-17"] <- "15"
X_train$Age[X_train$Age == "18-25"] <- "21"
X_train$Age[X_train$Age == "26-35"] <- "30"
X_train$Age[X_train$Age == "36-45"] <- "40"
X_train$Age[X_train$Age == "46-50"] <- "48"
X_train$Age[X_train$Age == "51-55"] <- "53"
X_train$Age[X_train$Age == "55+"] <- "60"

X_test$Age[X_test$Age == "0-17"] <- "15"
X_test$Age[X_test$Age == "18-25"] <- "21"
X_test$Age[X_test$Age == "26-35"] <- "30"
X_test$Age[X_test$Age == "36-45"] <- "40"
X_test$Age[X_test$Age == "46-50"] <- "48"
X_test$Age[X_test$Age == "51-55"] <- "53"
X_test$Age[X_test$Age == "55+"] <- "60"

X_train$Age <- as.integer(X_train$Age)
X_test$Age <- as.integer(X_test$Age)

X_train$Stay_In_Current_City_Years[X_train$Stay_In_Current_City_Years == "4+"] <- "4"
X_test$Stay_In_Current_City_Years[X_test$Stay_In_Current_City_Years == "4+"] <- "4"

X_train$Stay_In_Current_City_Years <- as.integer(X_train$Stay_In_Current_City_Years)
X_test$Stay_In_Current_City_Years <- as.integer(X_test$Stay_In_Current_City_Years)

X_train$Gender <- ifelse(X_train$Gender == "F", 1, 0)
X_test$Gender <- ifelse(X_test$Gender == "F", 1, 0)

user_count <- ddply(X_train, .(User_ID), nrow)
names(user_count)[2] <- "User_Count"
X_train <- merge(X_train, user_count, by="User_ID")
X_test <- merge(X_test, user_count, all.x=T, by="User_ID")

product_count <- ddply(X_train, .(Product_ID), nrow)
names(product_count)[2] <- "Product_Count"
X_train <- merge(X_train, product_count, by="Product_ID")
X_test <- merge(X_test, product_count, all.x=T, by="Product_ID")
X_test$Product_Count[is.na(X_test$Product_Count)] <- 0

product_mean <- ddply(X_train, .(Product_ID), summarize, Product_Mean=mean(Purchase))
X_train <- merge(X_train, product_mean, by="Product_ID")
X_test <- merge(X_test, product_mean, all.x=T, by="Product_ID")
X_test$Product_Mean[is.na(X_test$Product_Mean)] <- mean(X_train$Purchase)

X_train$flag_high <- ifelse(X_train$Purchase > X_train$Product_Mean,1,0)
user_high <- ddply(X_train, .(User_ID), summarize, User_High=mean(flag_high))
X_train <- merge(X_train, user_high, by="User_ID")
X_test <- merge(X_test, user_high, by="User_ID")

submit <- X_test[,c("User_ID","Product_ID")]

y <- X_train$Purchase

X_train <- subset(X_train, select=-c(Purchase,Product_ID,flag_high))
X_test <- subset(X_test, select=c(colnames(X_train)))


bst <- xgboost(data = as.matrix(X_train),
               label = y,
               objective="reg:linear",nrounds=500,max.depth=10,eta=0.1,colsample_bytree=0.5,seed=235,metric="rmse",importance=1,missing='NA')



pred <- predict(bst, as.matrix(X_test), outputmargin=TRUE,missing='NA')


submit$Purchase <- pred
submit$Purchase[submit$Purchase < 185] <- 185
submit$Purchase[submit$Purchase > 23961] <- 23961

submit$Purchase<-squish(submit$Purchase, round(quantile(submit$Purchase, c(.005, .995))))


write.csv(submit, "submit.csv", row.names=F)
