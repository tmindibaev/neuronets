import boto3, boto3.session
import numpy as np
import os, uuid

from boto3.s3.transfer import S3Transfer
from keras.models import Sequential, model_from_json
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.utils.np_utils import to_categorical
from rule_registry import RuleRegistry
from tempfile import gettempdir


class SequenceProcessor(object):
    _s3_archimedes_region = '<REGION>'
    _s3_archimedes_bucket = '<BUCKET_NAME>'

    def __init__(self, sequence, test_sequence, model_uid=None, input_block_length=5, resolution=10, verbose=True):
        """
        Process a sequence

        :param sequence:
        :param test_sequence:
        :param model_uid:
        :param input_block_length:
        :param resolution:
        :param verbose:
        """

        if input_block_length < 1:
            raise AttributeError('Input block length must be greater than 0')

        if len(sequence) < input_block_length or len(test_sequence) < input_block_length:
            raise AttributeError('Sequences must be longer than input block')

        self._sequence = sequence
        self._test_sequence = test_sequence
        self._input_block_length = input_block_length
        self._resolution = resolution
        self._verbose = verbose
        self._rule_registry = None
        self._uuid = model_uid

        # Initialize AWS S3
        session = boto3.session.Session(region_name=self._s3_archimedes_region)
        s3_client = session.client('s3', config= boto3.session.Config(signature_version='s3v4'))
        self._s3_transfer = S3Transfer(s3_client)

        # Prepare sequences
        self.X, self.y, self.symbol_count = self.prepare_sequence(self.sequence)
        self.X_test, self.y_test, self.test_symbol_count = self.prepare_sequence(self._test_sequence, test=True)

        # Setup rules
        self.setup_rules()

        # Build neural model
        self._model = None

        if self._uuid is None:
            # Create model from scratch
            if self._verbose:
                print ('Creating new model...')

            self.init_model()

        else:
            # Load model
            self._model = self.load(self._uuid)

        # Compile model
        if self._verbose:
            print ('Compiling model...')

        self._model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

    def fit(self, batch_size=64, epochs=8):
        if self._verbose:
            print ('Training model...')

        self._model.fit(self.X, self.y, batch_size=batch_size, nb_epoch=epochs,
                        validation_data=(self.X_test, self.y_test), verbose=self._verbose)

        # Fit rules
        self._rule_registry.fit_rules(self._model)

        # Setup rule engine model
        self._rule_registry.setup_model()

    def predict_with_neural_network(self, x, batch_size=1):
        return self._model.predict(x, batch_size=batch_size)

    def predict_with_rule_engine(self, x):
        return self._rule_registry.predict(x)

    def load(self, uuid):
        if self._verbose:
            print ('Loading model from S3. UUID: %s' % uuid)

        j_uuid = gettempdir() + os.pathsep + str(uuid.uuid4()) + '.json'
        h5_uuid = gettempdir() + os.pathsep + str(uuid.uuid4()) + '.h5'

        self._s3_transfer.download_file(self._s3_archimedes_bucket, uuid + '.json', j_uuid)
        self._s3_transfer.download_file(self._s3_archimedes_bucket, uuid + '.h5', h5_uuid)

        model = model_from_json(open(j_uuid).read())
        model.load_weights(h5_uuid)

        return model

    def save(self):
        if self._verbose:
            print('Saving model to S3 (JSON and H5). UUID: %s' % self._uuid)

        f_name = gettempdir() + os.pathsep + self._uuid
        t_name = f_name + str(uuid.uuid4())

        open(t_name + '.json', 'w').write(self._model.to_json())
        self._model.save_weights(t_name + '.h5')

        self._s3_transfer.upload_file(t_name + '.json', self._s3_archimedes_bucket, self._uuid + '.json',
                                      extra_args={'ContentType': "application/json"})
        self._s3_transfer.upload_file(t_name + '.h5', self._s3_archimedes_bucket, self._uuid + '.h5')

        return self._uuid

    def init_model(self):
        self._uuid = str(uuid.uuid4())

        # Create new neural model
        self._model = Sequential()

        self._model.add(LSTM(self._input_block_length * 100, return_sequences=True, input_shape=(self._input_block_length, 1)))
        self._model.add(Dropout(0.2))
        self._model.add(LSTM(self._input_block_length * 50, return_sequences=False))
        self._model.add(Dropout(0.1))
        self._model.add(Dense(self._input_block_length * 10))
        self._model.add(Dropout(0.1))
        self._model.add(Dense(self._input_block_length * 5))
        self._model.add(Dense(self.symbol_count))
        self._model.add(Activation('softmax'))

        if self._verbose:
            print ('New model created. UUID: %s' % self._uuid)

    def prepare_sequence(self, sequence, test=False):
        if self._verbose:
            if test:
                dataset = 'Test'
            else:
                dataset = 'Training'
            print ('Preprocessing sequence (%s). Unique symbols: %d' % (dataset, self.unique_symbols_count(sequence)))

        X = np.ndarray((len(sequence) - self._input_block_length, self._input_block_length, 1))
        y = np.ndarray((len(sequence) - self._input_block_length, self.unique_symbols_count(sequence)))

        for i in range(len(sequence)-self._input_block_length):
            for j in range(self._input_block_length):
                X[i, j, 0] = sequence[i+j]

        for i in range(self._input_block_length, len(sequence) - self._input_block_length + self._input_block_length):
            y[i - self._input_block_length] = to_categorical([sequence[i]], self.unique_symbols_count(sequence))

        return X, y, self.unique_symbols_count(sequence)

    def setup_rules(self):
        if self._verbose:
            print ('Setting up rules...')

        # Initialize Rule registry
        self._rule_registry = RuleRegistry(self.symbol_count, input_block_length=self._input_block_length,
                                           resolution=self._resolution, verbose=self._verbose)

        # Setup raw rules
        self._rule_registry.setup_rules(self.X, self.y)

    @staticmethod
    def unique_symbols(sequence):
        return np.unique(sequence)

    @staticmethod
    def unique_symbols_count(sequence):
        return len(SequenceProcessor.unique_symbols(sequence))

    @property
    def uuid(self):
        return self._uuid

    @property
    def sequence(self):
        return self._sequence

    @property
    def input_block_length(self):
        return self._input_block_length

    @property
    def model(self):
        return self._model