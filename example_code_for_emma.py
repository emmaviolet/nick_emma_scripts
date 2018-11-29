import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn import linear_model, metrics
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
import copy

## Begin functions
def calc_error(targets, predictions):
    ''' Helper function to calculate error of predicted targets'''
    return np.mean(np.abs(predictions - targets) / targets)[0]

def cross_val(feat_train, targ_train, fold_cv = 10, fold_reg = 20,
                    detailed_feedback = False, plot_cv = False,
                    feedback = True, fit_intercept = True):
    """ Wrapper for scikit-lear linear model to perform cross validation and
        parameter selection
    """
    error = np.zeros(fold_reg * 2)
    score = np.zeros(fold_reg * 2)
    reg_vals = np.float_power(10, range(-fold_reg,fold_reg))
    for i, reg in enumerate(reg_vals):
        cv_model = linear_model.Ridge(alpha = reg, fit_intercept = fit_intercept,
                    normalize = True, copy_X = True)
        score[i] = np.mean(cross_val_score(cv_model, feat_train,
                    y = targ_train, cv = fold_cv))
        predicted = cross_val_predict(cv_model, feat_train, targ_train,
                    cv = fold_cv)
        error[i] = calc_error(targ_train, predicted)
        if detailed_feedback:
            print('Reg {0}:\n error = {1:4.3f} %  score = {2:4.3f}\n'
                    .format(reg, error[i]*100, score[i]))
    ix_min_err = np.argmin(error)
    v_min_err = error[ix_min_err]
    ix_max_scr = np.argmax(score)
    v_max_scr = score[ix_max_scr]
    if feedback:
        print('''Cross validation results
        Max score: {0:4.3f}    \t Regularization mag: {1}
        Min error: {2:4.3f} %  \t Regularization mag: {3}\n'''
            .format(v_max_scr,reg_vals[ix_max_scr],
                    v_min_err * 100, reg_vals[ix_min_err]))
    if plot_cv:
        ax1_cv = plt.gca()
        ax1_cv.plot(reg_vals, score, 'b')
        ax1_cv.set_ylabel('Score (R^2)')
        ax2_cv = ax1_cv.twinx()
        ax2_cv.plot(reg_vals, error, 'r')
        ax2_cv.set_ylabel('Error (%)')
        ax2_cv.set_xscale('log')
        plt.xlabel('Alpha (regularization magnitude)')
        plt.title('Cross validation error')
        plt.show()
    return (reg_vals[ix_max_scr], v_min_err, v_max_scr)

def test_features(features, targets, fold_cv = 10, fold_reg = 20,
    n_feat_init = 2, detailed_feedback = False, plot_cv = False,
    plot_results = True, feedback = True, fit_intercept = True):
    '''Iterate through combinations of passed features and  cross-validate them
        to minimize prediction error.
    '''
    n_feature_sets = features.shape[1] - n_feat_init + 1
    feat_rho = stats.spearmanr(targets, features).correlation
    feat_rho = np.square(feat_rho[1:, 0])
    feat_order = list(np.argsort(feat_rho)[-1::-1])
    errors = np.zeros(n_feature_sets)
    scores = np.zeros(n_feature_sets)
    reg_mag = np.zeros(n_feature_sets)
    feature_sets = []
    # run first feature set
    ix_feat_set = feat_order[0:n_feat_init]
    print('Set 0, feature indices {0}:'.format(ix_feat_set))
    feat_cv = features.iloc[:, ix_feat_set]
    (reg_val, error, score) = cross_val(feat_cv, targets, fold_cv = fold_cv,
        fold_reg = fold_reg, detailed_feedback = False, plot_cv = False,
        feedback = True, fit_intercept = fit_intercept)
    feature_sets.append(ix_feat_set.copy())
    errors[0] = error
    scores[0] = score
    reg_mag[0] = reg_val
    for i_set in range(n_feature_sets - 1):
        ix_feat_set.extend([feat_order[n_feat_init + i_set]]) # - 1
        print('Set {0}, feature indices {1}:'.format(i_set, ix_feat_set))
        feat_cv = features.iloc[:, ix_feat_set]
        (reg_val, error, score) = cross_val(feat_cv, targets, fold_cv = fold_cv,
            fold_reg = fold_reg, detailed_feedback = False, plot_cv = False,
            feedback = True)
        feature_sets.append(ix_feat_set.copy())
        errors[i_set + 1]  = error
        scores[i_set + 1]  = score
        reg_mag[i_set + 1] = reg_val
        if score < scores[i_set]:
            ix_feat_set.pop()
    return feature_sets, reg_mag, errors, scores

def select_model(features, targets, method = 'diff', fold_cv = 10, fold_reg = 20,
    n_feat_init = 2, fit_intercept = True, detailed_feedback = False, plot_cv = False,
    plot_results = True, feedback = True):
    ''' Wrapper for test_features method to select a model feature set using either the
        largest increase in the model's score (correlation with targets), or the
        max score.
    '''
    feature_sets, reg_mag, errors, scores = test_features(
        features, targets, fold_cv = fold_cv, fold_reg = fold_reg,
        n_feat_init = n_feat_init, detailed_feedback = detailed_feedback,
        plot_cv = plot_cv, plot_results = plot_results, feedback = plot_results, fit_intercept = fit_intercept)
    if method == 'diff':
        ix_model = np.argmax(np.diff(scores)) + 1
    elif method == 'max':
        ix_model = np.argmax(scores)
    elif method == 'all':
        ix_model = len(scores) - 1
    else:
        print('Method not recognized, defaulting to \'diff\'')
        ix_model = np.argmax(np.diff(scores)) + 1
    feat_names = features.columns[feature_sets[ix_model]].values
    model = linear_model.Ridge(alpha = reg_mag[ix_model],
            fit_intercept = fit_intercept, normalize = True, copy_X = True)
    return feat_names, model

def train_test_model(features, targets, model, feedback = True):
    ''' Train and test a model using the passed feature set'''
    model_params = model.get_params()
    reg = model_params['alpha']
    if feedback:
        print('Fitting model with alpha = {0}'.format(reg))
    # model = linear_model.Ridge(alpha = reg,
    #         fit_intercept = True, normalize = True, copy_X = True)
    model.fit(features, targets)
    score = model.score(features, targets)
    predictions = model.predict(features)
    error = calc_error(targets, predictions)
    print('''Results:
             Error: {0:4.3f} % \t Score: {1:4.3f}\n'''
            .format(error*100 , score))
    return model, predictions, score, error

def select_train_test(features, targets, reps = 4, feature_selection = 'diff',
    fold_cv = 10, fold_reg = 20, n_feat_init = 2, fit_intercept = True,  detailed_feedback = False,
    plot_cv = False, plot_results = True, feedback = True, rand_seed = False):
    ''' Select a feature set, train it, and test it.'''
    errors = []
    scores = []
    models = []
    predictions = []
    feature_sets_all = []
    for i_rep in range(reps):
        print('\n\nTrain/test rep {0}\n'.format(i_rep + 1))
        if not rand_seed:
            v_seed = i_rep * 69
        else:
            v_seed = np.random.randint(1,1000000)
        feat_train, feat_test, targ_train, targ_test = train_test_split(
            features, targets, test_size = 0.25, random_state = v_seed)
        feature_names, model = select_model(
            feat_train, targ_train, method = feature_selection, fold_cv = fold_cv, fold_reg = fold_reg, fit_intercept = fit_intercept,
            n_feat_init = n_feat_init, detailed_feedback = detailed_feedback,
            plot_cv = plot_cv, plot_results = plot_results, feedback = feedback)
        print('Selected features ({}):\n'.format(feature_names.shape[0]) +
                ', '.join(feature_names))
        feature_sets_all.append(feature_names)
        model_features = feat_test[feature_names]
        t_model, t_predictions, t_score, t_error = train_test_model(
            model_features, targ_test, model)
        models.append(t_model)
        predictions.append(t_predictions)
        errors.append(t_error)
        scores.append(t_score)
    return errors, scores, feature_sets_all, predictions, models

def boot_model(features, targets, n_boot = 1000, plot = False):
    '''
    Perform Monte-calro simulation over targest to determine the level of chance
    error.
    '''
    boot_error = np.zeros(n_boot)
    boot_score = np.zeros(n_boot)
    for i_boot in range(n_boot):
        boot_target = targets.sample(frac = 1, replace = True)
        (errors, scores) = cross_val_reg(features, boot_target, fold_cv = 10,
                            fold_reg = 20, reps = 1, detailed_feedback = False,
                            plot_cv = False, plot_results = False, feedback = False)[1:3]
        boot_error[i_boot] = errors
        boot_score[i_boot] = scores
    if plot:
        plt.hist(boot_error, bins = int(n_boot/10))
        plt.show()
    return (boot_error, boot_score)

def map_model_feature_coeff(features, models):
    ''' build list of all features,  in case there are different feature sets for each model'''
    all_features = []
    for f in features:
        all_features.extend(list(f))
    all_features = np.unique(all_features)
    # map model features to all_features
    ix_feat = []
    for f in features:
        ix_feat.append([int(np.where(all_features == t_fn)[0]) for t_fn in f])
    # map coefficients to all features
    n_feat = all_features.shape[0]
    all_coeff = np.zeros((len(ix_feat), n_feat))
    for i, t_ix in enumerate(ix_feat):
        all_coeff[i, t_ix] = models[i].coef_
    return

def plot_coef(ax, models, features, bar_width = 0.5):
    ''' Helper functions to plot bar graph of coefficents for each model'''
    n_model = len(models)
    all_features, ix_feat, all_coeff = map_model_feature_coeff(features, models)
    bar_x_step = bar_width*(n_model + 3)*2
    x, y = np.meshgrid(np.arange(1,all_coeff.shape[1]*bar_x_step, bar_x_step),
                        np.arange(all_coeff.shape[0]))
    x = x + y*(bar_width*1.25)
    # b1 = ax.bar( x.flatten(), np.abs(all_coeff).flatten(), log = True )
    b1 = ax.bar( x.flatten(), all_coeff.flatten(), log = True , color = 'b')
    b2 = ax.bar( x.flatten(), all_coeff.flatten()*-1, log = True , color = 'r')
    ax.legend((b1, b2), ('+', '-'), title = 'Coefficient sign')
    ax.set_ylabel('Coefficient magnitude')
    ax.set_xticks(x[2,:])
    ax.set_xticklabels(all_features, rotation='vertical')
    ax.set_title('Model coefficients, N = {0}'.format(n_model))

def plot_fit_results(ax, fit_error, fit_score):
    ''' Helper function to flot the results of the model.'''
    n_reps = len(fit_error)
    l1 = ax.plot(range(n_reps), fit_error, 'r')
    l2 = ax.plot(range(n_reps), fit_score, 'b')
    ax.set_ylim(0,1)
    ax.set_ylabel('Magnitude (%)')
    ax.set_xlabel('Repetition')
    plt.title('Results')
    ax.legend(('Error', 'Score'))
