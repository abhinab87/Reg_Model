import pandas
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
import pickle
import joblib

# save the model to disk
pkl_filename = 'finalized_model.pkl'
joblib_filename = 'finalized_model.sav'

def linear_regression_model():
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
    names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
    dataframe = pandas.read_csv(url, names=names)
    array = dataframe.values
    X = array[:,0:8]
    Y = array[:,8]
    test_size = 0.33
    seed = 7
    X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=test_size, random_state=seed)
    x_test = X_test
    y_test = Y_test

    # Fit the model on training set
    model = LogisticRegression()
    model.fit(X_train, Y_train)
    pickle.dump(model, open(pkl_filename, 'wb'))
    joblib.dump(model, joblib_filename)
    return (X_test, Y_test)

# some time later...

if __name__ == "__main__":
    X_test, Y_test = linear_regression_model()
    # load the model from pkl to disk
    loaded_model = pickle.load(open(pkl_filename, 'rb'))
    X_test
    result = loaded_model.score(X_test, Y_test)
    print(result)

    # load the model from joblib to disk
    loaded_model = joblib.load(joblib_filename)
    result = loaded_model.score(X_test, Y_test)
    print(result)
