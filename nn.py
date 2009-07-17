#!/usr/bin/env python
import sys, numpy, math, logging, os, random, ConfigParser
import gc
from progressbar import *
from metrics import Metric

class NN:
    def __init__(self, nIn, nOut, config): 
        self.config = config
        self.nIn = nIn
        self.nOut = nOut
        self.nLayer         = config.getint("Architecture", "layer")
        self.nNodes         = config.getint("Architecture", "nodes")
        self.nIter          = config.getint("Architecture", "iterations")
        self.etha           = config.getfloat("Factors", "initlearnrate")
        self.alpha          = config.getfloat("Factors", "momentum") 
        self.steepness      = config.getfloat("Factors", "steepness") 
        self.stepsizedec    = config.getfloat("Factors", "stepsizedec") 
        self.stepsizeinc    = config.getfloat("Factors", "stepsizeinc") 
        offset              = config.getfloat("Factors", "initoffset")
        self.mindpp         = config.getfloat("Thresholds", "mindpp") 
        self.mindsse        = config.getfloat("Thresholds", "mindsse") 
        self.mindsumweights = config.getfloat("Thresholds", "mindsumweights") 
        actfunc             = config.get("Architecture", "activation")
        weightsinit         = config.get("Architecture", "initweights")
        self.errfct         = config.get("Architecture", "errorfunction")
        self.metrics        = Metric(config.get("Output", "metrics"))
        self.verbosity      = config.getint("Output", "verbosity")
        self.interactive    = config.getboolean("Output", "interactive")

        self.weights = []
        self.outs    = []
        self.deltas  = []

        def initWeights( generateMatrixFunc ):
            self.weights.append( generateMatrixFunc(nIn, self.nNodes) )
            for i in range(1, self.nLayer):
                self.weights.append( generateMatrixFunc(self.nNodes, self.nNodes) )
            self.weights.append( generateMatrixFunc(self.nNodes, nOut) )

        if weightsinit == "randuni":
            def mat(n,m): return offset * (numpy.mat(numpy.random.rand(n, m)) + 0.5)

        elif weightsinit == "randgauss":
            def mat(n,m): return numpy.mat(offset*numpy.random.standard_normal( [n, m] ))

        elif weightsinit == "uniform":
            def mat(n,m): return offset * numpy.mat(numpy.ones( [n, m] ))

        elif weightsinit == "exponential":
            def mat(n,m): return offset * numpy.mat(numpy.random.standard_exponential( size=[n, m] ))

        else:
            logging.error("Unknown weights initialization. Available: randuni, randgauss, uniform, exponential")
            sys.exit(-1)
        
        initWeights(mat)

        from copy import copy
        self.lastchange = copy(self.weights)

        # deltas
        for i in range(self.nLayer):
            self.deltas.append( numpy.zeros([1,self.nNodes]) )
            self.outs.append( [0.0] * self.nNodes )
        self.deltas.append( numpy.zeros([1,nOut]) )
        self.outs.append( [0.0] * nOut )

        # generate activation function
        if actfunc == "logistic":
            self.phi = lambda net: 1.0/(1.0+numpy.exp(-net * self.steepness))
            # TODO: speedup teh bottleneck
            def foo(net):
                r = self.phi(net)
                return numpy.multiply( r, (1.0-r) )
            self.dphi= foo#lambda net: numpy.multiply( self.phi(net), (1.0-self.phi(net)) )
        elif actfunc == "tanh":
            self.phi = lambda net: numpy.tanh(self.steepness * net)
            self.dphi= lambda net: 1.0-numpy.power(numpy.tanh(self.steepness * net), 2)
            #self.dphi= lambda net: 1.0-numpy.power(net, 2) # wrong dphi from bpnn
        elif actfunc == "linear":
            self.phi = lambda net: net
            self.dphi= lambda net: 1
        elif actfunc == "softmax":
            def foo(net):
                s = 1.0/numpy.exp(-net).sum()
                return s * numpy.exp(-net)
            self.phi = foo
            def bar(net):
                r = self.phi(net)
                return numpy.multiply( r, (1.0-r) )
            self.dphi= bar
        elif actfunc == "gauss":
            self.phi = lambda net: numpy.exp(-numpy.power(net-1,2))
            self.dphi= lambda net: -2*numpy.multiply(net-1, numpy.exp(-numpy.power(net-1,2)))
        elif actfunc == "step":
            self.phi = lambda net: numpy.sign(net)
            self.dphi= lambda net: 1
        else:
            logging.error("Unknown activation function. Available: logistic, tanh, linear, softmax, gauss, step")
            sys.exit(-1)
    
    def reload(self, config, weights):
        self.__init__(self.nIn, self.nOut, config)
        self.weights = weights

    ##############################################################################

    def test(self, data):
        conf = numpy.zeros([self.nOut, self.nOut], numpy.int16)
        allprobs = [ None ] * len(data)
        for i,row in enumerate(data):
            probs = self.passForward(row[:,1:])
            c_pred = probs.argmax()
            conf[ int(row[0,0]), c_pred ] += 1
            allprobs[i] = probs / probs.sum()
            
        return conf, 1-conf.trace()/float(conf.sum()), allprobs

    ##############################################################################

    def passForward(self, row):
        # input
        #for numNode in range(self.nNodes):
        #    sum = 0.0
        #    for nInput in range(self.nIn):
        #        sum += row[nInput] * self.weights[0][nInput,numNode]
        #    self.outs[0][numNode] = (sum, self.phi(sum))
        sum = row * self.weights[0]
        self.outs[0] = (sum, self.phi(sum))

        #inner layers
        for w in range( 1, self.nLayer+1 ):
            #for numNode in range( self.nNodes ):
            #    sum = 0.0
            #    for nInput in range( self.nNodes ):
            #        sum += self.outs[w-1][nInput][1] * self.weights[w][nInput,numNode]
            #    self.outs[w][numNode] = (sum, self.phi(sum))
            sum = self.outs[w-1][1] * self.weights[w]
            self.outs[w] = (sum, self.phi(sum))

        #output layer
        #for numNode in range(self.nOut):
        #    sum = 0.0
        #    for nInput in range( self.nNodes ):
        #        sum += self.outs[self.nLayer-1][nInput][1] * self.weights[-1][nInput,numNode]
        #    self.outs[-1][numNode] = (sum, self.phi(sum))
        return self.outs[-1][1][0]

    ##############################################################################

    def train(self, data, testdata):
        sse  = sys.maxint
        pp   = sys.maxint
        lift = sys.maxint

        interactive = self.interactive and os.isatty(sys.stdout.fileno())

        ref = numpy.zeros( [1,self.nOut] )
        c_old = None
        allprobs = [None] * len(data)
        classes      = [ d[0,0] for d in data ]
        classes_test = [ d[0,0] for d in testdata ]
        #pIndices = enumerate(range(self.nIn-self.nOut,self.nIn))

        for i in range(self.nIter):
            conf = numpy.zeros( [ self.nOut, self.nOut ] )

            sumold = sse
            ppold = pp
            sse = 0.0
            sce = 0.0
            if interactive: pbar = ProgressBar(maxval=len(data)).start()
            for k,row in enumerate(data):
                probs = self.passForward(row[:,1:])
                #for j,c in pIndices: data[k][0,c] = probs[0,j]
                c = int(row[0,0])
                ref[0,c_old] = 0.0
                ref[0,c] = 1.0
                c_old = c

                diff = ref-probs
                if self.errfct == "sse":
                    self.deltas[-1] = numpy.multiply( diff, self.dphi( probs ) )
                    sse += numpy.power(diff, 2).sum()
                elif self.errfct == "sce":
                    self.deltas[-1] = diff * self.steepness

                    # cross entropy: 1/C * sum{ (tk*log(yk)) + (1-tk)*log(1-yk) }
                    sce -= ((numpy.multiply(ref, numpy.log( probs )) + numpy.multiply((1-ref), numpy.log( 1 - probs )))).sum() / self.nOut

                weightschange = self.passBackward(row[:,1:])
                if interactive: pbar.update(k)

                # train statistics 
                c_pred = probs.argmax()
                conf[ int(row[0,0]), c_pred ] += 1
                allprobs[k] = probs

            conf_, err, tepr = self.test( testdata )
            output = self.metrics.obtain( testdata, classes_test, classes, allprobs, tepr, conf, err )

            if self.errfct == "sse":
                output["errfct"] = "SSE: % 6.4f" % sse
            elif self.errfct == "sce":
                output["errfct"] = "SCE: % 6.4f" % sce

            if interactive: pbar.finish()
            metrics = "%(lift)s%(pp)s%(fscore)s%(tester)s" % output
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

def readJf(filename, appendConstOne = False):
    first = True
    cls, feat = 0,0
    data = []
    if appendConstOne: const = [1.0]# + [0.5]*cls
    else: const = []

    for l in file(filename).readlines():
        if first: 
            cls, feat = [ int(s) for s in l.split() ]
            first = False
            continue

        if l.startswith("-1"): break

        vec = numpy.matrix( [float(f) for f in l.split()]+const ) # <- last one for bias
        data.append( vec )

    if appendConstOne:
        feat += 1 # +1 for bias

    return cls, feat, data 

def writeProbs(probs, filename):
    f = open(filename,'w')
    for pr in probs:
        f.write(" ".join("%.6f"%s for s in pr.flat))
        f.write("\n")
    f.close()

##############################################################################
    
def main(config):
    #gc.set_debug(gc.DEBUG_LEAK | gc.DEBUG_STATS)
    cls, feat, dataTest = readJf(config.get("Input","test"), True)
    cls, feat, dataTrain = readJf(config.get("Input","train"), True)

    net = NN(feat, cls, config)
    model = config.get("Input", "loadmodel")
    if model:
        net.loadmodel(model)
    else:
        trpr = net.train( dataTrain, dataTest )

    conf, err, tepr = net.test( dataTest )

    print conf
    print err

    ftr = config.get("Output", "probstrain")
    fte = config.get("Output", "probstest")
    if ftr: writeProbs(trpr, ftr)
    if fte: writeProbs(tepr, fte)

    model = config.get("Output", "savemodel")
    if model:
        net.savemodel(model)
    

##############################################################################
if __name__ == "__main__":
    try:
        import psyco
        psyco.full()
    except ImportError: pass
    logging.basicConfig(format="%(message)s")

    from optparse import OptionParser, OptionGroup
    parser = OptionParser(version="1.0", description="skynet")

    g = OptionGroup(parser, "Input")
    g.add_option("-c", "--config", dest="config", help="configuration file", metavar="FILE")
    g.add_option("-t", "--tr", dest="train", help="train data", metavar="FILE")
    g.add_option("-T", "--te", dest="test", help="test data", metavar="FILE")
    g.add_option("-u", "--loadmodel", dest="loadmodel", help="load model from file", metavar="FILE")
    parser.add_option_group(g)

    g = OptionGroup(parser, "Architecture")
    g.add_option("-i", "--iter", dest="iter", type="int", default=10)
    g.add_option("-l", "--layer", dest="layer", type="int", default=1)
    g.add_option("-n", "--nodes", dest="nodes", type="int", help="nodes per hidden layer", default=2)
    g.add_option("-a", "--activation", dest="activation", type="string", help="activation function [logistic, tanh, linear, softmax]", default="logistic")
    g.add_option("-w", "--initweights", dest="initweights", type="string", help="weights initialization [randgauss, randuni, uniform]", default="randgauss")
    g.add_option("-r", "--errorfct", dest="errorfunction", help="error function [sse, sce]", default="sse")
    parser.add_option_group(g)

    g = OptionGroup(parser, "Factors")
    g.add_option("-o", "--initoffset", dest="io", type="float", help="initial weights scaling factor", default=0.01)
    g.add_option("-e", "--etha", dest="etha", type="float", help="learning rate", default=1.0)
    g.add_option("-p", "--alpha", dest="alpha", type="float", help="momentum", default=0.0)
    parser.add_option_group(g)

    g = OptionGroup(parser, "Output")
    choices=["lift", "pp", "fmeasure:<target_class>","tester",""]
    g.add_option("-m", "--metrics", dest="metrics", type="choice", help="metrics to display (comma separated) [%s]"%(",".join(choices))[:-1], default="", choices=choices)
    g.add_option("--probstr", dest="probstrain", help="write p(c|x) of train data to file", metavar="FILE")
    g.add_option("--probste", dest="probstest", help="write p(c|x) of test data to file", metavar="FILE")
    g.add_option("-v", "--verbosity", dest="verbosity", help="verbosity level [0,1,..]", default=1)
    g.add_option("-s", "--savemodel", dest="savemodel", help="save model to file", metavar="FILE")
    g.add_option("-b", "--interactive", dest="interactive", help="show progress bar", default="True")
    parser.add_option_group(g)

    (options, args) = parser.parse_args(args=None, values=None)

    config = ConfigParser.SafeConfigParser(options.__dict__)
    if options.config: config.read(options.config)

    if None in [config.get("Input","train"), config.get("Input", "test")]:
        print "-t and -T required!"
        sys.exit(-1)

    main(config)
