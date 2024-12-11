ENTRY_NUMBER_LAST_DIGIT = 1 # change with yours
ENTRY_NUMBER = '2021MT10241'
PRE_PROCESSING_CONFIG ={
    "hard_margin_linear" : {
        "use_pca" : None,
    },

    "hard_margin_rbf" : {
        "use_pca" : None,
    },

    "soft_margin_linear" : {
        "use_pca" : None,
    },

    "soft_margin_rbf" : {
        "use_pca" : None,
    },

    "AdaBoost" : {
        "use_pca" : None,
    },

    "RandomForest" : {
        "use_pca" : True,
    }
}

SVM_CONFIG = {
    "hard_margin_linear" : {
        "C" : 1e9,
        "kernel" : 'linear',
        "val_score" : 0.45455093408543934, # add the validation score you get on val set for the set hyperparams.
                         # Diff in your and our calculated score will results in severe penalites
        # add implementation specific hyperparams below (with one line explanation)
    },
    "hard_margin_rbf" : {
        "C" : 1e9,
        "kernel" : 'rbf',
        "val_score" :0.4294550945212839, # add the validation score you get on val set for the set hyperparams.
                         # Diff in your and our calculated score will results in severe penalites
        # add implementation specific hyperparams below (with one line explanation)
    },

    "soft_margin_linear" : {
        "C" : 1, # add your best hyperparameter
        "kernel" : 'linear',
        "val_score" :0.4466358885297192, # add the validation score you get on val set for the set hyperparams.
                         # Diff in your and our calculated score will results in severe penalites
        # add implementation specific hyperparams below (with one line explanation)
    },

    "soft_margin_rbf" : {
         "C" : 1, # add your best hyperparameter
         "kernel" : 'rbf',
         "val_score" :0.42857142857142855, # add the validation score you get on val set for the set hyperparams.
                          # Diff in your and our calculated score will results in severe penalites
         # add implementation specific hyperparams below (with one line explanation)
     }
}

ENSEMBLING_CONFIG = {
    'AdaBoost':{
        'num_trees' : 10,
        'max_depth':2,
        "val_score" : 0.8913832121045641,
    },

    'RandomForest':{
        'num_trees' :100,
        'max_depth':3,
        'feature_subsample_size':200,
        "val_score" : 0.87258,
    }
}

