#!/usr/bin/env python
# This file is part of netr.
# 
# netr is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License 3 as published by
# the Free Software Foundation.
# 
# netr is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with netr.  If not, see <http://www.gnu.org/licenses/>.
import sys, numpy, math, logging, os, random, ConfigParser
from progressbar import *
from metrics import Metric
from data import Data

class NN:
    def __init__(self, config): 
        self.config = config
        self.nLayer         = config.getint("Architecture", "layer")
        self.nNodes         = config.getint("Architecture", "nodes")
        self.nIter          = config.getint("Architecture", "iterations")
        self.etha           = config.getfloat("Factors", "initlearnrate")
        self.alpha          = config.getfloat("Factors", "momentum") 
        self.steepness      = config.getfloat("Factors", "steepness") 
        self.stepsizedec    = config.getfloat("Factors", "stepsizedec") 
        self.stepsizeinc    = config.getfloat("Factors", "stepsizeinc") 
        self.offset         = config.getfloat("Factors", "initoffset")
        self.mindpp         = config.getfloat("Thresholds", "mindpp") 
        self.mindsse        = config.getfloat("Thresholds", "mindsse") 
        self.mindsumweights = config.getfloat("Thresholds", "mindsumweights") 
        self.actfunc        = config.get("Architecture", "activation")
        self.weightsinit    = config.get("Architecture", "initweights")
        self.errfct         = config.get("Architecture", "errorfunction")
        self.metrics        = Metric(config.get("Output", "metrics"), config.getint("Output", "metricsclass"))
        self.verbosity      = config.getint("Output", "verbosity")
        self.interactive    = config.getboolean("Output", "interactive")

        self.weights = []
        self.outs    = []
        self.deltas  = []

        self.generateActivationFunction()
    
    ##############################################################################

    def generateActivationFunction(self):

        if self.actfunc == "logistic":
            def dphi(net):
                r = 1.0/(1.0+numpy.exp(-net * self.steepness))
                return numpy.multiply( r, (1.0-r) )
            self.phi  = lambda net: 1.0/(1.0+numpy.exp(-net * self.steepness))
            self.dphi = dphi

        elif self.actfunc == "tanh":
            self.phi  = lambda net: numpy.tanh(self.steepness * net)
            self.dphi = lambda net: self.steepness * (1.0-numpy.power(numpy.tanh(net), 2))

        elif self.actfunc == "linear":
            self.phi  = lambda net: self.steepness * net
            self.dphi = lambda net: self.steepness

        elif self.actfunc == "softmax":
            def phi(net):
                s = 1.0/numpy.exp(-net).sum()
                return s * numpy.exp(-net)
            self.phi = foo
            def dphi(net):
                r = self.phi(net)
                return numpy.multiply( r, (1.0-r) )
            self.dphi = dphi

        elif self.actfunc == "gauss":
            self.phi = lambda net: numpy.exp(-numpy.power(net-1,2) * self.steepness)
            self.dphi= lambda net: -2*numpy.multiply(net-1, numpy.exp(-numpy.power(net-1,2)))

        elif self.actfunc == "sin":
            self.phi = lambda net: numpy.sin(self.steepness * net)
            self.dphi= lambda net: self.steepness * numpy.cos(self.steepness * net)
        else:
            logging.error("Unknown activation function. Available: logistic, tanh, linear, softmax, gauss, sin")
            sys.exit(-1)

    ##############################################################################

    def reload(self, config, weights):
        self.__init__(config)
        self.weights = weights

    ##############################################################################

    def initWeights(self, cls, feat):
        self.nIn = feat
        self.nOut = cls

        def initWeights( generateMatrixFunc ):
            self.weights.append( generateMatrixFunc(self.nIn, self.nNodes) )
            for i in range(1, self.nLayer):
                self.weights.append( generateMatrixFunc(self.nNodes, self.nNodes) )
            self.weights.append( generateMatrixFunc(self.nNodes, self.nOut) )

        if self.weightsinit == "randuni":
            def mat(n,m): return self.offset * (numpy.mat(numpy.random.rand(n, m)) + 0.5)

        elif self.weightsinit == "randgauss":
            def mat(n,m): return self.offset * numpy.mat(numpy.random.standard_normal( [n, m] ))

        elif self.weightsinit == "uniform":
            def mat(n,m): return self.offset * numpy.mat(numpy.ones( [n, m] ))

        elif self.weightsinit == "exponential":
            def mat(n,m): return self.offset * numpy.mat(numpy.random.standard_exponential( size=[n, m] ))

        else:
            logging.error("Unknown weights initialization. Available: randuni, randgauss, uniform, exponential")
            sys.exit(-1)
        
        initWeights(mat)

        from copy import copy
        self.lastchange = copy(self.weights)

        self.outs   = [None] * (self.nLayer + 1)
        self.deltas = [None] * (self.nLayer + 1)

    ##############################################################################

    def test(self, data):
        conf = numpy.zeros([self.nOut, self.nOut], numpy.int16)
        allprobs = [ None ] * len(data)
        for i,row in enumerate(data):
            allprobs[i] = self.passForward(row)
            conf[ data.targets[i], allprobs[i].argmax() ] += 1
            #TODO: not needed?
            allprobs[i] /= allprobs[i].sum()
            
        return conf, 1-conf.trace()/float(conf.sum()), allprobs

    ##############################################################################

    def passForward(self, row):
        # input
        sum = row * self.weights[0]
        self.outs[0] = (sum, self.phi(sum))

        # next layers
        for w in range( 1, self.nLayer+1 ):
            sum = self.outs[w-1][1] * self.weights[w]
            self.outs[w] = (sum, self.phi(sum))

        return self.outs[-1][1][0]

    ##############################################################################

    def train(self, data):
        sse  = sys.maxint
        pp   = sys.maxint

        self.initWeights( data.cls, data.feat )

        interactive = self.interactive and os.isatty(sys.stdout.fileno())

        ref = numpy.zeros( [1,self.nOut] )
        c_old = 0
        allprobs = [None] * len(data)

        for i in range(self.nIter):
            conf = numpy.zeros( [ self.nOut, self.nOut ] )

            sumold = sse
            ppold = pp
            sse = 0.0
            sce = 0.0
            if interactive: pbar = ProgressBar(maxval=len(data)).start()
            for k,row in enumerate(data):
                probs = self.passForward(row)
                ref[0,c_old] = 0
                ref[0,data.targets[k]] = 1
                c_old = data.targets[k]

                diff = ref-probs
                if self.errfct == "sse":
                    self.deltas[-1] = numpy.multiply( diff, self.dphi( probs ) )
                    sse += numpy.power(diff, 2).sum()
                elif self.errfct == "sce":
                    self.deltas[-1] = diff * self.steepness

                    # cross entropy: 1/C * sum{ (tk*log(yk)) + (1-tk)*log(1-yk) }
                    sce -= ((numpy.multiply(ref, numpy.log( probs )) + numpy.multiply((1-ref), numpy.log( 1 - probs )))).sum() / self.nOut

                weightschange = self.passBackward(row)
                if interactive: pbar.update(k)

                # train statistics 
                c_pred = probs.argmax()
                conf[ data.targets[k], c_pred ] += 1
                allprobs[k] = probs

            #conf_, err, tepr = self.test( testdata )
            #conf_, err, tepr = self.test( data )
            output = self.metrics.obtain( data, allprobs, conf, 1-conf.trace()/float(conf.sum()) )

            if self.errfct == "sse":
                output["errfct"] = "SSE: % 6.4f" % sse
            elif self.errfct == "sce":
                output["errfct"] = "SCE: % 6.4f" % sce

            if interactive: pbar.finish()
            metrics = "%(lift)s%(pp)s%(fscore)s%(tester)s%(auc)s" % output
            logging.warning("iter: % 4d er: %.6f %s rate: %.4f%s", i+1, 1-conf.trace()/conf.sum(), output["errfct"], self.etha, metrics)
            
            if weightschange < self.mindsumweights:
                self.weights[-1] = self.weights[-1] + numpy.random.standard_normal([self.nNodes, self.nOut]) * 0.1
                logging.warning("disturbing weights for leaving local optimum...")
            
            if sumold - sse < self.mindsse or ppold - pp < self.mindpp:
                self.etha *= self.stepsizedec
            else:
                self.etha *= self.stepsizeinc
        return allprobs

    ##############################################################################

    def passBackward(self, row):
        # precompute deltas for the inner layers
        for l in range(self.nLayer)[::-1]:
            self.deltas[l] = self.deltas[l+1] * self.weights[l+1].T 
            self.deltas[l] = numpy.multiply( self.deltas[l], self.dphi(self.outs[l][0]) )
            #for i in range(self.nNodes):
            #    self.deltas[l][i] = 0.0
            #    for j in range(len(self.deltas[l+1])):
            #        self.deltas[l][i] += self.deltas[l+1][j] * self.weights[l+1][i,j]
            #    self.deltas[l][i] *= self.dphi( self.outs[l][i][0] )
        #self.etha *= (1-self.alpha)
        #output layer
        delta = (1-self.alpha) * self.etha * numpy.outer( self.outs[-2][1], self.deltas[-1] ) + self.alpha * self.lastchange[-1]
        self.weights[-1] = self.weights[-1] + delta
        self.lastchange[-1] = delta
        #for j in range(self.nOut):
        #    f = self.etha * self.deltas[-1][j]
        #    for i in range( self.nNodes ):
        #        self.weights[-1][i,j] += f * self.outs[-2][i][1] 

        # recalculate weights forwards
        #inner layers
        for l in range( 1, self.nLayer ):
            #for j in range(self.nNodes):
            #    f = self.etha * self.deltas[l][j]
            #    for i in range (self.nNodes):
            #        self.weights[l][i,j] += f * self.outs[l-1][i][1]
            delta = (1-self.alpha) * self.etha * numpy.outer( self.outs[l-1][1], self.deltas[l] ) + self.alpha * self.lastchange[l]
            self.weights[l] = self.weights[l] + delta
            self.lastchange[l] = delta

        # input vector once again influences w'
        #for j in range(self.nNodes):
        #    f = self.etha * self.deltas[0][j]
        #    for i in range(self.nIn):
        #        self.weights[0][i,j] += f * row[i] 
        delta = (1-self.alpha) * self.etha * numpy.outer( row, self.deltas[0] ) + self.alpha * self.lastchange[0]
        self.weights[0] = self.weights[0] + delta
        self.lastchange[0] = delta

        return sum( [ d.sum() for d in self.lastchange ] )

##############################################################################

    def savemodel(self, modelname):
        import pickle
        model = ( self.weights, self.config )
        pickle.dump(model, open(modelname, "w"))

    def loadmodel(self, modelname):
        import pickle
        self.weights, self.config = pickle.load(file(modelname))
        self.reload(self.config, self.weights)

##############################################################################
