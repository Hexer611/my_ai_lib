from numpy import subtract, log10, square
from numpy import sum as SUM


class MeanSquaredError:
    def calc(self, model_out, expected_out):
        self.back = subtract(expected_out, model_out)
        self.output = SUM(square(self.back))


class Binary_Cross_Entropy:
    def calc(self, model_out, expected_out):
        self.back = expected_out*(1-model_out)+(1-expected_out)*(-model_out)
        self.output = SUM((expected_out)*(-log10(model_out)) + (1-expected_out)*(-log10(1-model_out)))
