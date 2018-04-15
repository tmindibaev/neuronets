import numpy as np
from fuzzyNet.processor import SequenceProcessor
from fuzzyNet.samples import SampleGenerator

if __name__ == '__main__':
    print("Archimedes test")

    # Discrete sequence
    print("Generating sequence...")

    #sample_generator = SampleGenerator(5000, 750)
    sample_generator = SampleGenerator(20000, 3000)
    train_set, test_set = sample_generator.generate_discrete_sequence()
    processor = SequenceProcessor(sequence=train_set, test_sequence=test_set, input_block_length=2, resolution=5)
    processor.fit(epochs=3)

    a = [0, 2]
    na = np.ndarray(shape=(1, 2, 1))
    na[0, :, 0] = a

    print(processor.predict_with_neural_network(na, batch_size=1))
    print(processor.predict_with_rule_engine(a))