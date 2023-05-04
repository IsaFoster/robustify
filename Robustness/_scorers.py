from sklearn import metrics

def get_scorer(metric):
    if isinstance(metric, str):
        return get_scorer_sckit_learn(metric)
    elif hasattr(metric, '__call__'):
        return get_custom_scorer(metric)
def get_scorer_sckit_learn(metric):
    # from https://scikit-learn.org/stable/modules/model_evaluation.html
    try: 
        return metrics.get_scorer(metric)
    except:
        print("not recognized")

def get_custom_scorer(metric):
    try:
        print("custom scorer!")
        return metric
    except: raise Exception ("soemthing went wrong")