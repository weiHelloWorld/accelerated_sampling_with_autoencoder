
from pybrain.structure.modules.neuronlayer import NeuronLayer
from scipy import sqrt

class CircularLayer(NeuronLayer):

    def _forwardImplementation(self, inbuf, outbuf):
        assert(len(inbuf) % 2 == 0)
        for i in range(len(inbuf) / 2):
            radius = sqrt(inbuf[2 * i] ** 2 + inbuf[2 * i + 1] ** 2)
            outbuf[2 * i] = inbuf[2 * i] / radius
            outbuf[2 * i + 1] = inbuf[2 * i + 1] / radius

    def _backwardImplementation(self, outerr, inerr, outbuf, inbuf):
        for i in range(len(inbuf) / 2):
            radius = sqrt(inbuf[2 * i] ** 2 + inbuf[2 * i + 1] ** 2)
            x_p = inbuf[2 * i]
            x_q = inbuf[2 * i + 1]
            inerr[2 * i] = x_q / (radius **3) * (x_q * outerr[2 * i] - x_p * outerr[2 * i + 1])
            inerr[2 * i + 1] = x_p / (radius **3) * (x_p * outerr[2 * i + 1] - x_q * outerr[2 * i])

