from sklearn import metrics

def get_scorer_sckit_learn(metric):
    # from https://scikit-learn.org/stable/modules/model_evaluation.html
    try: 
        return metrics.get_scorer(metric)
    except:
        print("not recognized")