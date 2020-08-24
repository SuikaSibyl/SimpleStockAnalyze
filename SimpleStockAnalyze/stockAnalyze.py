import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sklearn
from sklearn.preprocessing import scale
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn import neighbors
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler


def calc_macd(df, fastperiod=12, slowperiod=26, signalperiod=9):
    ewma12 = df['Adj Close'].ewm(span=fastperiod,adjust=False).mean()
    ewma26 = df['Adj Close'].ewm(span=slowperiod,adjust=False).mean()
    dif = ewma12-ewma26
    dea = dif.ewm(span=signalperiod,adjust=False).mean()
    df['macd'] = (dif-dea)*2
    return df


# load data
df_goog = pd.read_csv("data/GOOG.csv", sep=",")
df_msft = pd.read_csv("data/MSFT.csv", sep=",")

# 1- Check for missing values
# -----------------------------------------
print("# ------------------------------------------------------------")
print("# 1- Check for missing values\n")
# Check missing values
count_nan = df_goog.isnull().sum()
print("NA values in Google:\n------------------")
print(count_nan)
count_nan = df_msft.isnull().sum()
print("\nNA values in Microsoft:\n------------------")
print(count_nan)
print("\nNo NA values found in dataframe \n")

# 2- Pre-process of the data
# -----------------------------------------
print("# ------------------------------------------------------------")
print("# 2- Pre-process of the data\n")
# set Date as index
df_goog['Date'] = pd.to_datetime(df_goog['Date'])
df_msft['Date'] = pd.to_datetime(df_msft['Date'])
df_goog.set_index('Date', inplace=True)
df_msft.set_index('Date', inplace=True)
# Do min-max-scaler
scaler = MinMaxScaler(feature_range=(0, 1))
# calculate MA5,20,60
ma_list = [5, 20, 60]
for ma in ma_list:
    df_goog['MA_' + str(ma)] = df_goog['Adj Close'].rolling(ma).mean()
    df_msft['MA_' + str(ma)] = df_msft['Adj Close'].rolling(ma).mean()
calc_macd(df_goog)
calc_macd(df_msft)

# Create Prediction column
forecast_time = 30
df_goog['Prediction'] = df_goog['Adj Close'].shift(-forecast_time)
df_msft['Prediction'] = df_msft['Adj Close'].shift(-forecast_time)

# 3- Do basic visualization
# -----------------------------------------
print("# ------------------------------------------------------------")
print("# 3- Do basic visualization\n")
# Draw plot with plot & MAs
df_goog['Adj Close'].plot(label="GOOG")
df_goog['MA_5'].plot(label="GOOG:MA5")
df_goog['MA_20'].plot(label="GOOG:MA20")
df_goog['MA_60'].plot(label="GOOG:MA6O")
df_msft['Adj Close'].plot(label="MSFT")
df_msft['MA_5'].plot(label="GOOG:MA5")
df_msft['MA_20'].plot(label="GOOG:MA20")
df_msft['MA_60'].plot(label="GOOG:MA6O")
plt.legend()
plt.xlabel("Time")
plt.ylabel("Adj close")
plt.show()

df_goog['macd'].plot(label="GOOG")
df_msft['macd'].plot(label="MSFT")
plt.show()

# 4- Do linear regression
# -----------------------------------------
print("# ------------------------------------------------------------")
print("# 4- Do linear regression\n")
df_list = [df_goog, df_msft]
for df in df_list:
    X = df[60:-forecast_time].drop(['Prediction'],axis=1)
    y = df[60:-forecast_time]['Prediction']
    # split  train and test
    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=0)
    # regression model
    lr = LinearRegression()
    lr.fit(x_train, y_train)
    print('R^2', lr.score(x_train, y_train))
    print('Model Coffe.', lr.coef_)
    print('Model Intercept', lr.intercept_)
    # Predict the future 30 days
    x_forecast = np.array(df.drop(['Prediction'],1))[-forecast_time:]
    lr_prediction = lr.predict(x_forecast)
    print(lr_prediction)
    print('Max Value:', max(lr_prediction))
    print('Predicated earning rate:', max(lr_prediction)/df['Adj Close'].iloc[-1:].values)
    # plot the prediction situation
    linear_goog = df[60:].drop(['Prediction'],axis=1)
    linear_goog['Predicted'] = lr.predict(linear_goog)
    linear_goog[-forecast_time:]['Predicted'].plot(label='Predicted future')
    linear_goog[:-forecast_time]['Predicted'].plot(label='Regressed past')
    linear_goog['Adj Close'].shift(-forecast_time).plot(label='Ground truth')
    plt.legend()
    plt.xlabel("Time")
    plt.ylabel("Close")
    plt.show()

# 5- Do KNN regression
# -----------------------------------------
print("# ------------------------------------------------------------")
print("# 5- Do KNN clustering predict\n")
for df in df_list:
    X = df[60:-forecast_time].drop(['Prediction'],axis=1)
    y = df[60:-forecast_time]['Prediction']
    # split  train and test
    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)
    print(x_train)
    print(y_train)
    # regression model
    knn = neighbors.KNeighborsRegressor(n_neighbors=10, weights='distance')
    knn.fit(x_train, y_train)
    print('R^2', knn.score(x_train, y_train))
    # Predict the future 30 days
    x_forecast = np.array(df.drop(['Prediction'],1))[-forecast_time:]
    lr_prediction = knn.predict(x_forecast)
    print(lr_prediction)
    print('Max Value:', max(lr_prediction))
    print('Predicated earning rate:', max(lr_prediction)/df['Adj Close'].iloc[-1:].values)
    # plot the prediction situation
    linear_goog = df[60:].drop(['Prediction'],axis=1)
    linear_goog['Predicted'] = knn.predict(linear_goog)
    linear_goog[-forecast_time:]['Predicted'].plot(label='Predicted future')
    linear_goog[:-forecast_time]['Predicted'].plot(label='Regressed past')
    linear_goog['Adj Close'].shift(-forecast_time).plot(label='Ground truth')
    plt.legend()
    plt.xlabel("Time")
    plt.ylabel("Close")
    plt.show()

# 6- Do KNN clustering
# -----------------------------------------
print("# ------------------------------------------------------------")
print("# 6- Do KNN clustering\n")
from sklearn.cluster import KMeans
for df in df_list:
    df = df[60:-forecast_time]
    k = 2
    kmeans = KMeans(n_clusters=k, random_state=42).fit(df)
    print("\kmeans\n", kmeans)
    print(np.unique(kmeans.labels_, return_counts=True))
    print("\nkmeans cluster centers")
    print(kmeans.cluster_centers_)
    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_
    kmeans_3 = pd.DataFrame(labels, columns=["cluster"])
    print("\nkmeans3\n")
    print(kmeans_3.head())

    st = np.array(df)
    for i in range(k):
        ds = st[np.where(labels == i)]
        print(ds)
        plt.plot(ds[:, 0], ds[:, -1], 'o', markersize=7)
        plt.plot(ds[:, 0], ds[:, 0], 'o', markersize=1)
        lines = plt.plot(centroids[i, 0], centroids[i, 3], 'kx')
        plt.setp(lines, ms=15.0)
        plt.setp(lines, mew=4.0)
        plt.title('Cluster datra scatter plot')
        plt.xlabel('x-data')
        plt.ylabel('y-data')
        plt.legend(['cluster data', 'cluster centroid'])
    plt.show()

# 7- Do Decision Tree regression
# -----------------------------------------
print("# ------------------------------------------------------------")
print("# 7- Do KNN clustering predict\n")
from sklearn.tree import DecisionTreeRegressor
for df in df_list:
    X = df[60:-forecast_time].drop(['Prediction'], axis=1)
    y = df[60:-forecast_time]['Prediction']
    # split  train and test
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    # regression model
    regressor = DecisionTreeRegressor(random_state=0, max_depth=15)
    regressor.fit(x_train, y_train)
    print('R^2', regressor.score(x_train, y_train))
    # Predict the future 30 days
    x_forecast = np.array(df.drop(['Prediction'], 1))[-forecast_time:]
    lr_prediction = regressor.predict(x_forecast)
    print(lr_prediction)
    print('Max Value:', max(lr_prediction))
    print('Predicated earning rate:', max(lr_prediction) / df['Adj Close'].iloc[-1:].values)
    # plot the prediction situation
    linear_goog = df[60:].drop(['Prediction'], axis=1)
    linear_goog['Predicted'] = regressor.predict(linear_goog)
    linear_goog[-forecast_time:]['Predicted'].plot(label='Predicted future')
    linear_goog[:-forecast_time]['Predicted'].plot(label='Regressed past')
    linear_goog['Adj Close'].shift(-forecast_time).plot(label='Ground truth')
    plt.legend()
    plt.xlabel("Time")
    plt.ylabel("Close")
    plt.show()

# 7- Do Decision Tree regression
# -----------------------------------------
print("# ------------------------------------------------------------")
print("# 7- Do KNN clustering predict\n")
from sklearn.tree import DecisionTreeRegressor
for df in df_list:
    X = df[60:-forecast_time].drop(['Prediction'], axis=1)
    y = df[60:-forecast_time]['Prediction'] > df[60:-forecast_time]['Adj Close']
    print(y)
    # split  train and test
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    # Using the decision tree classifier
    from sklearn.tree import DecisionTreeClassifier
    classifier = DecisionTreeClassifier(max_depth=6)
    model = classifier.fit(x_train, y_train)
    print("The model is: ", model)
    # predict
    y_pred = classifier.predict(x_test)
    # evaluate
    from sklearn.metrics import classification_report, confusion_matrix
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    # score
    from sklearn import metrics
    print(metrics.accuracy_score(y_test, y_pred))
    # Decision Tree Visualization
    from sklearn import tree
    feature_cols = ['Variance', 'Skewness', 'Curtosis', 'Entropy']
    fig = plt.figure(figsize=(20, 5))
    myTree = tree.plot_tree(classifier, filled=True, node_ids=True)
    plt.title("Bill ")
    plt.axis("tight")
    plt.show()
    # predict future
    x_predict = df[-forecast_time:].drop(['Prediction'], axis=1)
    y_pred = classifier.predict(x_predict)
    print(y_pred)
    print('True number:', np.sum(y_pred != 0))