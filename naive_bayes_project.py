
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import metrics

from sklearn.datasets import load_breast_cancer
breast_cancer = load_breast_cancer()

X_train, X_test, y_train, y_test = train_test_split(breast_cancer.data, 
                                                    breast_cancer.target, 
                                                    test_size = 0.3, 
                                                    random_state = 42)


from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb_model = nb.fit(X_train, y_train)
print(nb_model)

print(nb.predict_proba(X_test))

y_pred = nb.predict(X_test)

print("olması gereken: ", y_test)
print("tahmin sonucu: ", y_pred)

print("accuracy score: ", accuracy_score(y_test, y_pred))