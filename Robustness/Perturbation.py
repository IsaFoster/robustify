from sklearn.metrics import accuracy_score

def changeInput(df, percent, changeUp):
    if changeUp:
        return df.transform(lambda x: x + x*percent/100)
    return df.transform(lambda x: x - x*percent/100)

def pertubate(model, X_train, y_train, X_test, y_test, percentage):
    accuracy_scores = []

    X_train_up = changeInput(X_train, 10, True)
    y_train = y_train.values.ravel()
    model.fit(X_train_up, y_train)
    pred_up = model.predict(X_test)
    accuracy_scores.append(accuracy_score(pred_up, y_test))

    X_train_down = changeInput(X_train, 10, False)
    model.fit(X_train_down, y_train)
    pred_down = model.predict(X_test)
    accuracy_scores.append(accuracy_score(pred_down, y_test))

    return accuracy_scores