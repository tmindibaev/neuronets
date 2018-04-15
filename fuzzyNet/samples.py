import numpy as np


class SampleGenerator(object):
    def __init__(self, num_samples, num_tests):
        self.num_samples = num_samples
        self.num_tests = num_tests

    def generate_discrete_sequence(self):
        symbols = (0, 1, 2, 3, 4, 5)
        data = []

        data.append(np.random.randint(0, high=len(symbols)))

        for i in range(1, self.num_samples + self.num_tests):
            # Apply "probabilistic-logic" rules
            if data[i-1] == symbols[0]:
                if self.uniform_threshold():
                    data.append(symbols[2])
                    continue
                else:
                    data.append(symbols[3])
                    continue

            if data[i-1] == symbols[1]:
                if self.uniform_threshold():
                    data.append(symbols[0])
                    continue
                else:
                    data.append(symbols[4])
                    continue

            if data[i-1] == symbols[2]:
                if self.uniform_threshold():
                    data.append(symbols[5])
                    continue
                else:
                    data.append(symbols[1])
                    continue

            if data[i-1] == symbols[3]:
                if self.uniform_threshold():
                    data.append(symbols[1])
                    continue
                else:
                    data.append(symbols[0])
                    continue

            if data[i-1] == symbols[4]:
                if self.uniform_threshold():
                    data.append(symbols[3])
                    continue
                else:
                    data.append(symbols[1])
                    continue

            if data[i - 1] == symbols[5]:
                if self.uniform_threshold():
                    data.append(symbols[4])
                    continue
                else:
                    data.append(symbols[2])
                    continue

        return np.array(data[0:self.num_samples]).astype(np.int32), np.array(data[self.num_samples:]).astype(np.int32)

    @staticmethod
    def uniform_threshold():
        if np.random.uniform() < 0.9:
            return True
        else:
            return False