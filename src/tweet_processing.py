import os
import pandas as pd

data = pd.read_csv("data/stock_data.csv")

#View dataset loaded
#print(data)

#View each tweet/headline
#for row in  data["Text"].values:
    #print(row, "\n")

#Convert text to lowercase
data["Text"] = data["Text"].str.lower()

#Quit '
data["Text"] = data["Text"].str.replace("'", "", regex=False) #look_for = "'"

#Quit :
data["Text"] = data["Text"].str.replace(": ", " ", regex=False) #look_for = ": "

#Quit ,
data["Text"] = data["Text"].str.replace(", "," ", regex=False) #look_for = ", "

#Changhe %
data["Text"] = data["Text"].str.replace("%", " percent ", regex=False) #look_for = ", "

#Quit , in numbers.  e.g:(1,500 = 1500)
data["Text"] = data["Text"].str.replace(r'([0-9]),([0-9])', r'\1\2',regex=True)

#Quit , in sentences not properly written. e.g(hi,man= hi man)
data["Text"] = data["Text"].str.replace(r'([a-z]),([a-z])', r'\1 \2', regex=True)

#Quit , in sentences not properly written. e.g(hi,.....= hi .....)
data["Text"] = data["Text"].str.replace(r'([a-z]),([0-9\.])', r'\1 \2', regex=True)

#Quit . in sentences not properly written. e.g(.....hi= hi)
data["Text"] = data["Text"].str.replace("[\.]{2,}", " ", regex=True)

#Quit double whiteblanks
data["Text"] = data["Text"].str.replace(" +", " ", regex=True) #look_for = " "

#Quit
data["Text"] = data["Text"].str.strip(" .?!,():;-")


look_for = "3.08"

#View each tweet/headline
for row in  data.loc[data["Text"].str.contains(look_for, regex=False)].values:
    print(row, "\n")


