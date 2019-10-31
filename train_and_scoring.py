import pickle
import numpy as np
import pandas as pd


from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegressionCV
from sklearn.linear_model import RidgeClassifierCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC

from sklearn.metrics import accuracy_score


# This converts the coefficients into a more natural range between 0 to 100 
def score(x,a=1):
    return (1/(1+np.exp(-a*(x))))*100



if __name__ == '__main__':
    
    # Get dataframe
    final_df = pickle.load(open("final_df.p", "rb"))
    final_df.set_index('MatchID',inplace=True)


    # Split into features vs target
    X = final_df.drop("BlueWin",axis=1)
    y = final_df.BlueWin.astype('int')

    # Get train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=10000, random_state=42)

    # Set random seed
    np.random.seed(42)

    # Logistic Regression CV
    print("Working on logistic regression...")
    lr = LogisticRegressionCV(cv=5, random_state=0).fit(X_train, y_train)


    # Ridge Regression CV
    print("Working on ridge classifier...")
    rc = RidgeClassifierCV(cv=5).fit(X_train, y_train)

    # Parameters for Max Depth for SVM
    print("Working on SVM..")
    parameters_svm = {'C':np.logspace(-3,3,5)}
    svm = GridSearchCV(LinearSVC(), parameters_svm, n_jobs=-1).fit(X_train,y_train).best_estimator_


    # Parameters for Max Depth for Decision Tree
    print("Working on decision tree...")
    parameters_dt = {'max_depth':range(20,100,5)}
    dt = GridSearchCV(DecisionTreeClassifier(), parameters_dt, n_jobs=-1).fit(X_train,y_train).best_estimator_

    classifiers = [lr,rc,svm,dt]
    names = ["Logistic Regression","Ridge Regression","Support Vector Classifier","Decision Tree"]
    for i in range(len(names)):
        print("{}: train accuracy was: {:.4f} and test accuracy was {:.4f}".format(names[i],accuracy_score(y_train,classifiers[i].predict(X_train)),accuracy_score(y_test,classifiers[i].predict(X_test))))

    # pickle fitted  logistic regression object
    pickle.dump(lr ,open( "logistic_regression.p", "wb" ))

    # Create champion ranking list from the coefficients
    scores = pd.DataFrame({'Champion':X_train.columns,'Score':score(lr.coef_.flatten(),7)}).sort_values('Score',ascending=False).reset_index(drop=True)
    # Create image that will go straight to HTML
    scores["url"] = scores.Champion.apply(lambda x: '<div class="picture"> <img src="http://ddragon.leagueoflegends.com/cdn/9.20.1/img/champion/' + str(x) + '.png"></div>')


    scores.Score = scores.Score.round(2) # Round the scores
    scores.reset_index(inplace=True)  # Reset index
    scores.columns = ["Rank","Champion","Score"," "] # Rename column
    scores.Rank  += 1 # Rank should start at 0

    pickle.dump(scores ,open( "scores.p", "wb" )) # save as pickle