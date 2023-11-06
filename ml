#ML1
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

uber = pd.read_csv('uber.csv')

uber.head()

uber.info()

uber.isnull().sum()

uber.corr()

uber.dropna(inplace=True)

plt.boxplot(uber['fare_amount'])

from sklearn.model_selection import train_test_split

x = uber.drop("fare_amount",axis=1)
y = uber['fare_amount']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 1)

from sklearn.linear_model import LinearRegression

x['pickup_datetime'] = pd.to_numeric(pd.to_datetime(x['pickup_datetime']))
x = x.loc[:, x.columns.str.contains('^Unnamed')]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 1)

from sklearn.linear_model import LinearRegression

lrmodel = LinearRegression()
lrmodel.fit(x_train, y_train)

#ML2
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn import metrics

df=pd.read_csv('emails.csv')

df.head()

df.columns

df.isnull().sum()

df.dropna(inplace = True)

df.drop(['Email No.'],axis=1,inplace=True)
X = df.drop(['Prediction'],axis = 1)
y = df['Prediction']

from sklearn.preprocessing import scale
X = scale(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=7)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

print("Prediction",y_pred)

print("KNN accuracy = ",metrics.accuracy_score(y_test,y_pred))

print("Confusion matrix",metrics.confusion_matrix(y_test,y_pred))

#ML3
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt #Importing the libraries

df = pd.read_csv("Churn_Modelling.csv")

df.head()

df.shape

df.describe()

df.isnull()

df.isnull().sum()

df.info()

df.dtypes

df.columns

df = df.drop(['RowNumber', 'Surname', 'CustomerId'], axis= 1) #Dropping the unnecessary columns

df.head()

def visualization(x, y, xlabel):
    plt.figure(figsize=(10,5))
    plt.hist([x, y], color=['red', 'green'], label = ['exit', 'not_exit'])
    plt.xlabel(xlabel,fontsize=20)
    plt.ylabel("No. of customers", fontsize=20)
    plt.legend()

df_churn_exited = df[df['Exited']==1]['Tenure']
df_churn_not_exited = df[df['Exited']==0]['Tenure']

visualization(df_churn_exited, df_churn_not_exited, "Tenure")

df_churn_exited2 = df[df['Exited']==1]['Age']
df_churn_not_exited2 = df[df['Exited']==0]['Age']

visualization(df_churn_exited2, df_churn_not_exited2, "Age")

#ML4
import numpy as np
import pandas as pd

data = pd.read_csv('./diabetes.csv')
data.head()

data.isnull().sum()

for column in data.columns[1:-3]:
    data[column].replace(0, np.NaN, inplace = True)
    data[column].fillna(round(data[column].mean(skipna=True)), inplace = True)
data.head(10)

X = data.iloc[:, :8] 
Y = data.iloc[:, 8:] 

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn_fit = knn.fit(X_train, Y_train.values.ravel())
knn_pred = knn_fit.predict(X_test)

from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
print("Confusion Matrix")
print(confusion_matrix(Y_test, knn_pred))
print("Accuracy Score:", accuracy_score(Y_test, knn_pred))
print("Reacal Score:", recall_score(Y_test, knn_pred))
print("F1 Score:", f1_score(Y_test, knn_pred))
print("Precision Score:",precision_score(Y_test, knn_pred))

#-ML5
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans, k_means #For clustering
from sklearn.decomposition import PCA #Linear Dimensionality reduction.

df = pd.read_csv("sales_data_sample.csv", sep=",", encoding='Latin-1') #Loading the dataset.

df.head()

df.shape

df.describe()

df.info()

df.isnull().sum()

df.dtypes

df_drop  = ['ADDRESSLINE1', 'ADDRESSLINE2', 'STATUS','POSTALCODE', 'CITY', 'TERRITORY', 'PHONE', 'STATE', 'CONTACTFIRSTNAME', 'CONTACTLASTNAME', 'CUSTOMERNAME', 'ORDERNUMBER']
df = df.drop(df_drop, axis=1) #Dropping the categorical uneccessary columns along with columns having null values. Can't fill the null values are there are alot of null values.

df.isnull().sum()

df.dtypes

df['COUNTRY'].unique()

df['PRODUCTLINE'].unique()

df['DEALSIZE'].unique()

productline = pd.get_dummies(df['PRODUCTLINE']) #Converting the categorical columns.
Dealsize = pd.get_dummies(df['DEALSIZE'])

df = pd.concat([df,productline,Dealsize], axis = 1)

df_drop  = ['COUNTRY','PRODUCTLINE','DEALSIZE'] #Dropping Country too as there are alot of countries.
df = df.drop(df_drop, axis=1)

df['PRODUCTCODE'] = pd.Categorical(df['PRODUCTCODE']).codes #Converting the datatype.

df.drop('ORDERDATE', axis=1, inplace=True) #Dropping the Orderdate as Month is already included.

df.dtypes #All the datatypes are converted into numeric

distortions = [] # Within Cluster Sum of Squares from the centroid
K = range(1,10)
for k in K:
    kmeanModel = KMeans(n_clusters=k)
    kmeanModel.fit(df)
    distortions.append(kmeanModel.inertia_)   #Appeding the intertia to the Distortions

plt.figure(figsize=(16,8))
plt.plot(K, distortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Distortion')
plt.title('The Elbow Method showing the optimal k')
plt.show()

-----DAA------
#DAA1
COUNT = 0
x = int(input("Enter Number of Terms: "))
first = 0
sec = 1
c = 0

if x < 0:
    print("Enter a valid input..")
elif x == 0:
    print(0)
elif x == 1:
    print("Fibonacci series up to", x, "is", first)
else:
    while c < x:
        print(first)
        COUNT = COUNT + 1
        nth = first + sec
        COUNT = COUNT + 1
        first = sec
        COUNT = COUNT + 1
        sec = nth
        COUNT = COUNT + 1
        c += 1
        COUNT = COUNT + 1

    print("Steps required using Counter:", COUNT)

#DAA1
COUNT = 0
def recur_fibo(n):
    global COUNT
    COUNT = COUNT + 1

    if n <= 1:
        return n
    else:
        return recur_fibo(n - 1) + recur_fibo(n - 2)

nterms = int(input("How many terms? "))

if nterms <= 0:
    print("Please enter a positive integer")
else:
    print("Fibonacci sequence:")
    for i in range(nterms):
        print(recur_fibo(i))

    print("Steps required using Counter:", COUNT)

#DAA3
def knapSack(W, wt, val, n):
    dp = [0 for i in range(W+1)]
    
    for i in range(1, n+1):
        for w in range(W, 0, -1):
            if wt[i-1] <= w:
                dp[w] = max(dp[w], dp[w - wt[i-1]] + val[i-1])
    
    return dp[W]  

val = [60, 100, 120]
wt = [10, 20, 30]
W = 50
n = len(val)
print(knapSack(W, wt, val, n))

#DAA4
def n_queens(n):
    col = set()
    posDiag=set() # (r+c)
    negDiag=set() # (r-c)

    res=[]

    board = [["0"]*n for i in range(n) ]
    def backtrack(r):
        if r==n:
            copy = [" ".join(row) for row in board]
            res.append(copy)
            return

        for c in range(n):
            if c in col or (r+c) in posDiag or (r-c) in negDiag:
                continue

            col.add(c)
            posDiag.add(r+c)
            negDiag.add(r-c)
            board[r][c]="1"

            backtrack(r+1)

            col.remove(c)
            posDiag.remove(r+c)
            negDiag.remove(r-c)
            board[r][c]="0"
    backtrack(0)
    for sol in res:
        for row in sol:
            print(row)
        print()
    
if __name__=="__main__":
    n_queens(8)
    
#DAA5
import random

def quick_sort(arr):
    if len(arr) <= 1:
        return arr

    pivot = random.choice(arr)
    lesser = [x for x in arr if x < pivot]
    equal = [x for x in arr if x == pivot]
    greater = [x for x in arr if x > pivot]

    return quick_sort(lesser) + equal + quick_sort(greater)

if __name__ == "__main__":
    arr = [3, 6, 8, 10, 1, 2, 1]

    sorted_arr = quick_sort(arr.copy())
    print("Quick Sort:")
    print(sorted_arr)

#huffman
import heapq

class node:
    def __init__(self,freq,symbol,left=None,right=None):
        self.freq=freq
        self.symbol=symbol
        self.left=left 
        self.right=right 
        self.huff= ''

    def __lt__(self,nxt): 
        return self.freq<nxt.freq

def printnodes(node,val=''):
    newval=val+str(node.huff)
    if node.left: 
        printnodes(node.left,newval)
    if node.right: 
        printnodes(node.right,newval)
    if not node.left and not node.right:
        print("{} -> {}".format(node.symbol,newval))

if __name__=="__main__":
    chars = ['a', 'b', 'c', 'd', 'e', 'f']
    freq = [ 5, 9, 12, 13, 16, 45]
    nodes=[]    

    for i in range(len(chars)): 
        heapq.heappush(nodes, node(freq[i],chars[i]))
    while len(nodes)>1:
        left=heapq.heappop(nodes)
        right=heapq.heappop(nodes)
        left.huff = 0
        right.huff = 1
        newnode = node(left.freq + right.freq , left.symbol + right.symbol , left , right)
        heapq.heappush(nodes, newnode)
    printnodes(nodes[0])
    
----BT------
#bank.sol

//SPDX-License-Identifier: MIT 

pragma solidity ^0.8.0;

contract bank{
    mapping(address => uint) public balances;
    function deposit() public payable{
        balances[msg.sender] += msg.value; 
    }
    function withdraw(uint _amount) public{
        require(balances[msg.sender]>= _amount, "Not enough ether");
        balances[msg.sender] -= _amount*1000000000000000000;
        (bool sent,) = msg.sender.call{value: _amount*1000000000000000000}("Sent");
        require(sent, "failed to send ETH");    
    }
    function getBal() public view returns(uint){
        return address(this).balance/1000000000000000000;
    }

}

#student.sol
// SPDX-License-Identifier: MIT   
pragma solidity >= 0.8.7;

contract MarksManagmtSys
{
	
	struct Student
	{
		int ID;
		string fName;
		string lName;
		int marks;
	}

	address owner;
	int public stdCount = 0;
	mapping(int => Student) public stdRecords;

	modifier onlyOwner
	{
		require(owner == msg.sender);
		_;
	}
	constructor()
	{
		owner=msg.sender;
	}

	function addNewRecords(int _ID,
						string memory _fName,
						string memory _lName,
						int _marks) public onlyOwner
	{
		
		stdCount = stdCount + 1;

		stdRecords[stdCount] = Student(_ID, _fName,
									_lName, _marks);
	}
	function bonusMarks(int _bonus) public onlyOwner
	{
		stdRecords[stdCount].marks =
					stdRecords[stdCount].marks + _bonus;
	}
}




