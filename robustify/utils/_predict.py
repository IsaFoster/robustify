from ._transform import convert_to_numpy
import tensorflow as tf
import keras

def is_keras_model(model):
    return isinstance(model, (tf.keras.Model, keras.Model, tf.estimator.Estimator))

def get_prediction(model, X, custom_predict):
    if custom_predict != None:
        try:
            y_pred = custom_predict(model, X)
            verify_predictions(X, y_pred, custom_predict)
            return convert_to_numpy(y_pred)
        except Exception:
            raise
    else:
        try:
            if is_keras_model(model):
                y_pred = model.predict(X.values, verbose=0)
            else:
                y_pred = model.predict(X.values)
            return convert_to_numpy(y_pred)
        except Exception:
            raise

def verify_predictions(X, y_pred,custom_predict):
    if y_pred is None:
        raise TypeError("No predictions produced by method {}".format(custom_predict.__name__))
    if len(X) != len(y_pred):
        raise ValueError("Length of predictions by method {} must match length of input".format(custom_predict.__name__))
