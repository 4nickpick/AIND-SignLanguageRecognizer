import math
import statistics
import traceback
import warnings

import numpy as np
import sys
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences

import timeit

class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Bayesian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        model_likelihood_probabilities = {}

        for num_components_n in range(self.min_n_components, self.max_n_components + 1, 1):

            num_features_d = len(self.X[0])

            num_params_p = num_components_n ** 2 + 2 * num_features_d * num_components_n - 1

            try:
                a = -2
                b = self.base_model(num_components_n)
                c = b.score(self.X, self.lengths)
                d = num_params_p
                e = np.log(len(self.X))
                model_likelihood_l = a * c + d * e

                # model_likelihood_l = -2 * self.base_model(num_components_n).score(self.X, self.lengths) + num_params_p \
                #                         * np.log(len(self.X))

                model_likelihood_probabilities[num_components_n] = model_likelihood_l
            except:
                continue

        if model_likelihood_probabilities:
            return self.base_model(min(model_likelihood_probabilities, key=model_likelihood_probabilities.get))
        else:
            return self.base_model(self.n_constant)


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        model_discriminative_probabilities = {}

        for num_components_n in range(self.min_n_components, self.max_n_components + 1, 1):
            for word in self.words:
                model_key = str(num_components_n) + "-" + word
                x, lengths = self.hwords[word]
                try:
                    model_likelihood_l = self.base_model(num_components_n).score(x, lengths)
                    model_discriminative_probabilities[model_key] = model_likelihood_l
                except:
                    model_discriminative_probabilities[model_key] = 0

        model_likelihood_probabilities = {}
        for num_components_n in range(self.min_n_components, self.max_n_components + 1, 1):
            try:
                likelihood_probability = self.base_model(num_components_n).score(self.X, self.lengths)
                discriminative_probability = 0
                for word in self.words:
                    if word is not self.this_word:
                        model_key = str(num_components_n) + "-" + word
                        try:
                            discriminative_probability += model_discriminative_probabilities[model_key]
                        except:
                            continue
            except:
                continue

            model_likelihood_probabilities[num_components_n] = \
                likelihood_probability - ((1 / (len(self.words) - 1)) * discriminative_probability)

        if model_likelihood_probabilities:
            return self.base_model(max(model_likelihood_probabilities, key=model_likelihood_probabilities.get))
        else:
            return self.base_model(self.n_constant)


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        model_likelihood_probabilities = {}
        splitter = KFold(random_state=self.random_state, n_splits=3)

        for num_components_n in range(self.min_n_components, self.max_n_components + 1, 1):
            model_likelihood_components = []

            try:
                for train_index, test_index in splitter.split(self.sequences):
                    x, lengths = combine_sequences(test_index, self.sequences)
                    x_train, lengths_train = combine_sequences(train_index, self.sequences)
                    try:

                        model = GaussianHMM(n_components=num_components_n, covariance_type="diag", n_iter=1000,
                                            random_state=self.random_state, verbose=False).fit(x_train, lengths_train)
                        model_likelihood_components.append(model.score(x, lengths))
                    except:
                        continue

            except:
                try:
                    model_likelihood_components.append(self.base_model(self.n_constant).score(self.X, self.lengths))
                except:
                    continue

            if model_likelihood_components:
                model_likelihood_probabilities[num_components_n] = np.average(model_likelihood_components)

        if model_likelihood_probabilities:
            return self.base_model(max(model_likelihood_probabilities, key=model_likelihood_probabilities.get))
        else:
            return self.base_model(self.n_constant)

