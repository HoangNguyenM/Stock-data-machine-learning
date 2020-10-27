# get stock price

require(rpart)
require(quantmod)

aapl<- getSymbols("TSLA", src = "yahoo", from = "2015-01-20", to = "2020-08-01", auto.assign = FALSE)


plot(aapl[,6])

# find trading points
# n is the number of trades allowed in the specified period of time

# assuming 1 buy + 1 sell per week
n = floor(length(aapl[,6])/5)

price <- aapl[,6]
price_change <- as.numeric(diff(price))

pos = 2

while (pos < length(price_change) - 1)
{
  if (price_change[pos] * price_change[pos + 1] > 0)
  {
    price_change[pos + 1] = price_change[pos + 1] + price_change[pos]
    price_change[pos] = 0
  }
  pos = pos + 1
}

# find optimal sell points
count = 1
index_sell <- rep(1,n)

while (count < n + 1)
{
  temp = max(price_change[-index_sell])
  for (i in 2:length(price_change))
  {
    if (temp == price_change[i])
    {
      index_sell[count] = i
    }
  }
  count = count + 1
}

# find optimal buy points
# define price_change again since it was modified in the previous step
index_buy <- rep(1,n)
price_change <- as.numeric(diff(price))


for (i in 1:n)
{
  j = index_sell[i] - 1
  while (price_change[index_sell[i]] * price_change[j] > 0 && j > 1)
  {
    j = j - 1
  }
  index_buy[i] = j
}

# results
#index_buy
#price[index_buy]
#index_sell
#price[index_sell]


# print the labels

labels = rep(0,length(aapl[,6]))
labels[index_buy] = 1
labels[index_sell] = -1

# compute the profit
price_change <- as.numeric(diff(price))

cum_profit = NULL

switch = labels[1]
profit = 0
for (i in 2:length(price_change))
{
  if (switch == 1)
  {
    profit = profit + price_change[i]
  }
  cum_profit = c(cum_profit,profit)
  switch = switch + labels[i]
}
cum_profit

# compute RSI

aapl1 <- getSymbols("AAPL", src = "yahoo", from = "2014-10-01", to = "2020-08-01", auto.assign = FALSE)


rel_str <- TTR::RSI(aapl1[,6], n = 14)
rel_str = rel_str[(length(rel_str[,1])-length(price)+1):length(rel_str[,1])]

length(rel_str)
length(price)

# compute MACD

aapl1 <- getSymbols("AAPL", src = "yahoo", from = "2014-10-01", to = "2020-08-01", auto.assign = FALSE)
mov_avg = MACD(aapl1[,6], nFast = 12, nSlow = 26, nSig = 9, percent = TRUE)
mov_avg = mov_avg[(length(mov_avg[,1])-length(price)+1):length(mov_avg[,1])]

length(mov_avg[,1])

# compute Bollinger Bands

aapl1 <- getSymbols("AAPL", src = "yahoo", from = "2014-10-01", to = "2020-08-01", auto.assign = FALSE)
aapl1 <- cbind(aapl1[,2], aapl1[,3], aapl1[,4])

band = TTR::BBands(aapl1, n = 20, sd = 2)
band = band[(length(band[,1])-length(price)+1):length(band[,1])]

length(band[,1])

# format the training data set
price = aapl[,6]
ptg_chg = diff(log(price))
volume = as.numeric(aapl[,5])
ptg_chg[1] = 0

trainset = cbind(ptg_chg, volume, rel_str, mov_avg, band)
trainset

write.csv(trainset, "D:/Dropbox/Code/traindata.csv")
write.csv(labels, "D:/Dropbox/Code/trainlabels.csv")

trainset = cbind(trainset, labels)

write.csv(trainset, "D:/Dropbox/Code/aapl.csv")

# Logistic regression
trainset <- as.data.frame(trainset)
length(labels)
length(trainset[,1])
mylogit <- glm(labels ~., data = trainset, family = "binomial")
result <- predict(mylogit, trainset, type = "response")

sum((round(result) - labels)^2)/length(price)


# Binary trees
tree = rpart(formula = labels~., data = trainset)
result <- predict(tree, trainset)
sum((round(result) - labels)^2)/length(price)

# make the test data
#aapl_test<- getSymbols("AAPL", src = "yahoo", from = "2020-08-01", to = "2020-09-01", auto.assign = FALSE)

#price_test = aapl_test[,6]
#ptg_chg_test = diff(log(price_test))
#volume_test = as.numeric(aapl_test[,5])
#ptg_chg_test[1] = 0

#testset = cbind(ptg_chg_test, volume)
