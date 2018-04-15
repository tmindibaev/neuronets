import hashlib
import numpy as np
import skfuzzy as fuzz
import re

from collections import OrderedDict
from skfuzzy.control.antecedent_consequent import Antecedent, Consequent
from skfuzzy.control import ControlSystem, ControlSystemSimulation
from skfuzzy.control.rule import Rule


class RuleRegistry(object):
    _antecedent_symbol = 'Fa(%(Symbol)s)'
    _antecedent_regex = '(.*)([0-9]+)(.*)'
    _consequent_symbol = 'Fc(%(Symbol)s)'
    _consequent_regex = '(.*)([0-9]+)(.*)'
    _range_level_symbol = 'L'
    _range_level_regex = '(.*)([0-9]+)'
    _rule_prefix = 'R'

    def __init__(self, unique_symbols, input_block_length=5, resolution=10, max_iterations=500, verbose=True):
        """
        Rule registry

        :param unique_symbols:
        :param input_block_length:
        :param resolution:
        :param max_iterations:
        :param verbose:
        """

        if unique_symbols <= 0:
            raise AttributeError('The number of unique symbols must be greater than zero')

        if resolution < 5:
            raise AttributeError('Resolution must be equal or greater than 5')

        self._unique_symbols = unique_symbols
        self._input_block_length = input_block_length
        self._resolution = resolution
        self._max_iterations = max_iterations
        self._verbose = verbose

        if self._verbose:
            print ('Initializing Rule registry...')

        self._raw_rules = {}
        self._rules = {}
        self._antecedents = {}
        self._consequents = {}
        self._a_regex = re.compile(self._antecedent_regex)
        self._c_regex = re.compile(self._consequent_regex)
        self._l_regex = re.compile(self._range_level_regex)
        self._model = None
        self._controller = None

        # Setup membership functions
        self._range_levels = self.membership_ranges()

        # Generate antecedents
        if self._verbose:
            print ('Generating antecedents and consequents...')

        for i in range(self._unique_symbols):
            # Antecedents
            antecedent = Antecedent(np.linspace(0, 1, self._resolution-2), self.symbol(i))

            for label, values in self._range_levels.items():
                antecedent[label] = fuzz.trimf(antecedent.universe, values)

            self._antecedents[self.symbol(i)] = antecedent

            # Consequents
            consequent = Consequent(np.linspace(0, 1, self._resolution-2), self.symbol(i, True))

            for label, values in self._range_levels.items():
                consequent[label] = fuzz.trimf(consequent.universe, values)

            self._consequents[self.symbol(i, True)] = consequent

            print ('(%s) -> (%s)' % (antecedent, consequent))

    def setup_rules(self, X, Y):
        skipped = 0

        if self._verbose:
            print ('Generating rules...')

        for i in range(len(X)):
            x = X[i, :, 0]
            y = self.from_category(Y[i])

            # Retrieve unique elements and relative frequencies
            antecedents, antecedent_frequency = self.term_frequencies(x)

            # Generate antecedent and consequent
            key = ''
            antecedent = []
            consequent = self._consequents[self.symbol(y, True)][self.matching_level(1)]

            for j in range(len(antecedents)):
                term = self._antecedents[self.symbol(int(antecedents[j]))][self.matching_level(antecedent_frequency[j])]
                key += str(self.symbol(int(antecedents[j]))) + str(self.matching_level(antecedent_frequency[j]))
                antecedent.append(term)

            key += str(self.symbol(y, True)) + str(self.matching_level(1))
            hashed_key = str(hashlib.md5(key.encode('utf-8')).hexdigest())
            #hashlib.sha256(str(random.getrandbits(256)).encode('utf-8')).hexdigest()
            rule = Rule(self.concat_terms(antecedent, mode='AND'), consequent)

            if hashed_key not in self._raw_rules:
                self._raw_rules[hashed_key] = rule
            else:
                skipped += 1

        self._rules = self._raw_rules

        if self._verbose:
            print ('Generation complete. %d raw rules created, %d skipped' % (len(self._raw_rules), skipped))

    def fit_rules(self, model):
        if self._verbose:
            print ('Fitting rules with neural engine...')

        fitted_rules = OrderedDict()
        fitted_rules_raw = OrderedDict()
        rule_index = 0

        for key, rule in self._rules.items():
            antecedents = [0 for i in range(self._unique_symbols)]

            for term in rule.antecedent_terms:
                symbol = int(self.symbol_from_label(term.label))
                level = int(self.level_from_label(term.label))
                antecedents[symbol] = level

            n_antecedents = np.array(antecedents)
            r_antecedents = n_antecedents / np.linalg.norm(n_antecedents)

            avg_prediction = np.array([0.0 for t in range(self._unique_symbols)])
            valid_sequences = 0

            for i in range(self._max_iterations):
                sequence = self.generate_sequence(r_antecedents)
                if len(sequence) != self._input_block_length:
                    # Skip this sequence
                    continue

                # Predict with neural network
                n_sequence = np.ndarray(shape=(1, self._input_block_length, 1))
                n_sequence[0, :, 0] = sequence

                prediction = model.predict(n_sequence, batch_size=1)

                # Sum up prediction
                avg_prediction += prediction[0, :]
                valid_sequences += 1

            avg_prediction /= float(valid_sequences)

            # Generate weighted consequents
            consequents = []

            for j in range(self._unique_symbols):
                consequents.append(self._consequents[self.symbol(j, True)][self.matching_level(avg_prediction[j])])

            rule.consequent = tuple(consequents)

            # Remove redundant rules
            if rule.__repr__() in fitted_rules_raw:
                # Skip this rule
                continue
            else:
                fitted_rules_raw[rule.__repr__()] = rule

        for key, rule in fitted_rules_raw.items():
            fitted_rules[self._rule_prefix + str(rule_index)] = rule
            rule_index += 1

        self._rules = fitted_rules

        if self._verbose:
            print ('Fitting complete. KB is made up of %d rules:' % len(self._rules))

            for key, rule in self._rules.items():
                print('%s) %r' % (key, rule))

    def setup_model(self, rule=None):
        if len(self._rules) == 0:
            raise RuntimeError('Before setting up model in necessary to populate rule set')

        if self._verbose and rule is None:
            print ('Setting up rule engine...')

        if rule is not None:
            self._model = ControlSystem([rule])
        else:
            self._model = ControlSystem(tuple(self._rules.values()))

        self._controller = ControlSystemSimulation(self._model)

    def predict(self, x):
        if self._model is None or self._controller is None:
            raise RuntimeError('Before predicting in necessary to setup model')

        # Retrieve unique elements and relative frequencies
        antecedents, antecedent_frequency = self.term_frequencies(x)

        input_data = [0.0 for i in range(self._unique_symbols)]

        for i in range(len(antecedents)):
            input_data[antecedents[i]] = antecedent_frequency[i]

        for i in range(self._unique_symbols):
            self._controller.input[self.symbol(i)] = input_data[i]

        try:
            self._controller.compute()

        except AssertionError:
            if self._verbose:
                print ('Inference error')
            return None

        return self._controller.output

    def symbol(self, s, consequent=False):
        if consequent:
            return self._consequent_symbol % {
                'Symbol': str(s)
            }
        else:
            return self._antecedent_symbol % {
                'Symbol' : str(s)
            }

    def term_frequencies(self, x):
        terms, term_count = np.unique(x, return_counts=True)
        term_frequency = term_count / float(self._input_block_length)
        return terms, term_frequency

    def membership_ranges(self):
        ranges = OrderedDict()
        distribution = np.linspace(0, 1, self._resolution)

        for i in range(0, self._resolution-2):
            t = []

            if i == 0:
                t.append(distribution[0])
                t.append(distribution[0])
                t.append(distribution[1])

            elif i == self._resolution-3:
                t.append(distribution[self._resolution - 2])
                t.append(distribution[self._resolution - 1])
                t.append(distribution[self._resolution - 1])

            else:
                t.append(distribution[i])
                t.append(distribution[i + 1])
                t.append(distribution[i + 2])

            ranges[self._range_level_symbol + str(i)] = np.array(t)

        if self._verbose:
            print ('Membership function ranges:')

            for label, values in ranges.items():
                print('%s) %r' % (label, values))

        return ranges

    def generate_sequence(self, antecedents):
        sequence = []
        weights = np.argsort(antecedents)

        for i in range(self._input_block_length):
            n = np.random.uniform(0, 1)

            for j in range(len(weights)):
                if n <= antecedents[weights[j]]:
                    sequence.append(weights[j])
                    continue

        return np.array(sequence)

    def matching_level(self, x):
        for label, values in self._range_levels.items():
            if np.amin(values) <= x <= np.amax(values):
                return label

        return self._range_levels.keys()[0]

    def from_category(self, y):
        for i in range(self._unique_symbols):
            if y[i] == 1:
                return i

    @staticmethod
    def concat_terms(terms, mode='OR'):
        clause = terms[0]

        for i in range(1, len(terms)):
            if mode == 'OR':
                clause |= terms[i]
            else:
                clause &= terms[i]

        return clause

    def symbol_from_label(self, label):
        match = self._a_regex.search(label)
        return match.group(2)

    def level_from_label(self, label):
        match = self._l_regex.search(label)
        return match.group(2)

    @property
    def rules(self):
        return self._rules