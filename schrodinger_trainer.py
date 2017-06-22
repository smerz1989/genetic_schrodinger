from pybrain.supervised.trainers import BackpropTrainer
import numpy as np

class SchrodingerTrainer(BackpropTrainer):
    def __init__(self, module, dataset=None, learningrate=0.01, lrdecay=1.0,momentum=0., verbose=False, batchlearning=False,weightdecay=0.):
        BackpropTrainer.__init__(self,module,dataset,learningrate,lrdecay,momentum,verbose,batchlearning,weightdecay)

    def setData(self, dataset):
        """Associate the given dataset with the trainer."""
        self.ds = dataset

    def _calcDerivs(self, seq):
        self.module.reset()
        for sample in seq:
            self.module.activate(sample[0]) 
            error = 0
            ponderation = 0.
        for offset, sample in reversed(list(enumerate(seq))):
            target = sample[1]
            outerr = target - self.module.outputbuffer[offset][0]*np.sin(self.module.outputbuffer[offset][1])
        if len(sample) > 2:
            importance = sample[2]
            error += 0.5 * dot(importance, outerr ** 2)
            ponderation += sum(importance)
            self.module.backActivate(outerr * importance)
        else:
            error += 0.5 * sum(outerr ** 2)
            ponderation += len(target)
            str(outerr)
            self.module.backActivate(outerr)
        return error, ponderation
