from abc import ABCMeta, abstractmethod
from sklearn.model_selection._search import BaseSearchCV
from sklearn.model_selection._search import _check_param_grid
from sklearn.model_selection._split import check_cv
from sklearn.model_selection._validation import _fit_and_score
from sklearn.metrics._scorer import _check_multimetric_scoring
from sklearn.utils.validation import indexable, check_is_fitted, _check_fit_params
from sklearn.base import is_classifier, clone, BaseEstimator
from sklearn.model_selection import ParameterGrid
from joblib import Parallel, delayed
from itertools import product
import time 
import numpy as np

import arcus.azureml.experimenting.trainer as trainer

class ArcusBaseSearchCV(BaseSearchCV):
    __metaclass__ = ABCMeta

    def __init__(self, estimator, param_grid, *, scoring=None,
                 n_jobs=None, iid='deprecated', refit=True, cv=None,
                 verbose=0, pre_dispatch='2*n_jobs',
                 error_score=np.nan, return_train_score=False):
        super().__init__(
            estimator=estimator, scoring=scoring,
            n_jobs=n_jobs, iid=iid, refit=refit, cv=cv, verbose=verbose,
            pre_dispatch=pre_dispatch, error_score=error_score,
            return_train_score=return_train_score)
        self.param_grid = param_grid
        _check_param_grid(param_grid)

    def _run_search(self, evaluate_candidates):
        """Search all candidates in param_grid"""
        evaluate_candidates(ParameterGrid(self.param_grid))
    
    def fit(self, X, y=None, *, groups=None, **fit_params):
        self.initialize_fitting(X, y)

        estimator = self.estimator
        cv = check_cv(self.cv, y, classifier=is_classifier(estimator))

        scorers, self.multimetric_ = _check_multimetric_scoring(
            self.estimator, scoring=self.scoring)

        if self.multimetric_:
            if self.refit is not False and (
                    not isinstance(self.refit, str) or
                    # This will work for both dict / list (tuple)
                    self.refit not in scorers) and not callable(self.refit):
                raise ValueError("For multi-metric scoring, the parameter "
                                 "refit must be set to a scorer key or a "
                                 "callable to refit an estimator with the "
                                 "best parameter setting on the whole "
                                 "data and make the best_* attributes "
                                 "available for that metric. If this is "
                                 "not needed, refit should be set to "
                                 "False explicitly. %r was passed."
                                 % self.refit)
            else:
                refit_metric = self.refit
        else:
            refit_metric = 'score'

        X, y, groups = indexable(X, y, groups)
        fit_params = _check_fit_params(X, fit_params)

        n_splits = cv.get_n_splits(X, y, groups)

        base_estimator = clone(self.estimator)

        parallel = Parallel(n_jobs=self.n_jobs, verbose=self.verbose,
                            pre_dispatch=self.pre_dispatch)

        fit_and_score_kwargs = dict(scorer=scorers,
                                    fit_params=fit_params,
                                    return_train_score=self.return_train_score,
                                    return_n_test_samples=True,
                                    return_times=True,
                                    return_parameters=False,
                                    error_score=self.error_score,
                                    verbose=self.verbose)
        results = {}
        with parallel:
            all_candidate_params = []
            all_out = []

            def evaluate_candidates(candidate_params):
                candidate_params = list(candidate_params)
                n_candidates = len(candidate_params)

                if self.verbose > 0:
                    print("Fitting {0} folds for each of {1} candidates,"
                          " totalling {2} fits".format(
                              n_splits, n_candidates, n_candidates * n_splits))

                out = parallel(delayed(self._fit_score_and_log)(clone(base_estimator),
                                                       X, y,
                                                       train=train, test=test,
                                                       parameters=parameters,
                                                       **fit_and_score_kwargs)
                               for parameters, (train, test)
                               in product(candidate_params,
                                          cv.split(X, y, groups)))

                if len(out) < 1:
                    raise ValueError('No fits were performed. '
                                     'Was the CV iterator empty? '
                                     'Were there no candidates?')
                elif len(out) != n_candidates * n_splits:
                    raise ValueError('cv.split and cv.get_n_splits returned '
                                     'inconsistent results. Expected {} '
                                     'splits, got {}'
                                     .format(n_splits,
                                             len(out) // n_candidates))

                all_candidate_params.extend(candidate_params)
                all_out.extend(out)

                nonlocal results
                results = self._format_results(
                    all_candidate_params, scorers, n_splits, all_out)
                return results

            self._run_search(evaluate_candidates)

        # For multi-metric evaluation, store the best_index_, best_params_ and
        # best_score_ iff refit is one of the scorer names
        # In single metric evaluation, refit_metric is "score"
        if self.refit or not self.multimetric_:
            # If callable, refit is expected to return the index of the best
            # parameter set.
            if callable(self.refit):
                self.best_index_ = self.refit(results)
                if not isinstance(self.best_index_, numbers.Integral):
                    raise TypeError('best_index_ returned is not an integer')
                if (self.best_index_ < 0 or
                   self.best_index_ >= len(results["params"])):
                    raise IndexError('best_index_ index out of range')
            else:
                self.best_index_ = results["rank_test_%s"
                                           % refit_metric].argmin()
                self.best_score_ = results["mean_test_%s" % refit_metric][
                                           self.best_index_]
            self.best_params_ = results["params"][self.best_index_]

        if self.refit:
            # we clone again after setting params in case some
            # of the params are estimators as well.
            self.best_estimator_ = clone(clone(base_estimator).set_params(
                **self.best_params_))
            refit_start_time = time.time()
            if y is not None:
                self.best_estimator_.fit(X, y, **fit_params)
            else:
                self.best_estimator_.fit(X, **fit_params)
            refit_end_time = time.time()
            self.refit_time_ = refit_end_time - refit_start_time

        # Store the only scorer not as a dict for single metric evaluation
        self.scorer_ = scorers if self.multimetric_ else scorers['score']

        self.cv_results_ = results
        self.n_splits_ = n_splits

        return self

    @abstractmethod
    def log_results(self, train_score: float, test_score: float, sample_count: int, durations:np.array, parameters: dict, estimator):
        raise NotImplementedError

    @abstractmethod
    def initialize_fitting(self, X, y=None):
        raise NotImplementedError

    def _fit_score_and_log(self, estimator, X, y, scorer, train, test, verbose,
                   parameters, fit_params, return_train_score=False,
                   return_parameters=False, return_n_test_samples=False,
                   return_times=False, return_estimator=False,
                   error_score=np.nan):
        fit_result = _fit_and_score(estimator, X, y, scorer, train, test, verbose,
                      parameters, fit_params, True,
                      True, True,
                      True, True,
                      error_score)
        
        test_score = fit_result[0]['score']
        train_score = fit_result[1]['score']
        sample_count = fit_result[2]
        durations = [fit_result[3],fit_result[4]]
        parameters = fit_result[5]
        estimator = fit_result[6]

        self.log_results(train_score, test_score, sample_count, durations, parameters, estimator)

        if return_train_score:
            ret = [fit_result[0], fit_result[1]]
        else:
            ret = [fit_result[0]]
        if return_n_test_samples:
            ret.append(sample_count)
        if return_times:
            ret.extend(durations)
        if return_parameters:
            ret.append(parameters)
        if return_estimator:
            ret.append(estimator)
        
        return ret


class LocalArcusGridSearchCV(ArcusBaseSearchCV):
    __active_trainer: trainer.Trainer = None
    __current_idx: int = 0
    #TODO : Subclass the sklearn.model_selection._search.BaseSearchCV class. Override the fit(self, X, y=None, groups=None, **fit_params) method, and modify its internal evaluate_candidates(candidate_params) function. Instead of immediately returning the results dictionary from evaluate_candidates(candidate_params), perform your serialization here (or in the _run_search method depending on your use case). With some additional modifications, this approach has the added benefit of allowing you to execute the grid search sequentially (see the comment in the source code here: _search.py). Note that the results dictionary returned by evaluate_candidates(candidate_params) is the same as the cv_results dictionary. This approach worked for me, but I was also attempting to add save-and-restore functionality for interrupted grid search executions.
    def __init__(self, estimator, param_grid, *, scoring=None,
                 n_jobs=None, iid='deprecated', refit=True, cv=None,
                 verbose=0, pre_dispatch='2*n_jobs',
                 error_score=np.nan, return_train_score=False, active_trainer: trainer.Trainer = None):
        super().__init__(
            estimator=estimator, param_grid = param_grid, scoring=scoring,
            n_jobs=n_jobs, iid=iid, refit=refit, cv=cv, verbose=verbose,
            pre_dispatch=pre_dispatch, error_score=error_score,
            return_train_score=return_train_score)
    
        self.param_grid = param_grid
        self.__active_trainer = active_trainer
        _check_param_grid(param_grid)

    def initialize_fitting(self, X, y=None):
        print('Start fitting')
        self.__current_idx = 0

    def log_results(self, train_score: float, test_score: float, sample_count: int, durations:np.array, parameters: dict, estimator):
        self.__current_idx+=1
        if(self.__active_trainer is not None):
            self.__active_trainer.add_tuning_result(self.__current_idx, train_score, test_score, sample_count, durations, parameters, estimator)