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

from optparse import OptionParser, OptionGroup
import ConfigParser

def usage():
    parser = OptionParser(version="1.0", description="netr")

    ############################################################################

    f = ["jf", "svm", "csv:<delimiter>"]

    g = OptionGroup(parser, "Input")
    g.add_option("-c", "--config", dest="config", help="configuration file", metavar="FILE")
    g.add_option("-t", "--tr", dest="train", help="train data", metavar="FILE")
    g.add_option("-T", "--te", dest="test", help="test data", metavar="FILE")
    g.add_option("-f", "--format", dest="format", help="data format %s"%f, )
    g.add_option("-u", "--loadmodel", dest="loadmodel", help="load model from file", metavar="FILE")
    parser.add_option_group(g)

    ############################################################################
    
    a = ["logistic", "tanh", "linear", "softmax", "gauss", "step"]
    w = ["randgauss", "randuni", "uniform"]
    e = ["sse", "sce"]

    g = OptionGroup(parser, "Architecture")
    g.add_option("-i", "--iter", dest="iterations", type="int", default=10)
    g.add_option("-l", "--layer", dest="layer", type="int", default=1)
    g.add_option("-n", "--nodes", dest="nodes", type="int", help="nodes per hidden layer", default=2)
    g.add_option("-a", "--activation", dest="activation", type="choice", help="activation function %s"%str(a), choices=a, default="logistic")
    g.add_option("-w", "--initweights", dest="initweights", type="choice", help="weights initialization %s"%str(w), choices=w, default="randgauss")
    g.add_option("-r", "--errorfct", dest="errorfunction", type="choice", help="error function"%e, choices=e, default="sse")
    parser.add_option_group(g)

    ############################################################################

    g = OptionGroup(parser, "Factors")
    g.add_option("-o", "--initoffset", dest="io", type="float", help="initial weights scaling factor", default=0.01)
    g.add_option("-e", "--etha", dest="etha", type="float", help="learning rate", default=0.1)
    g.add_option("-p", "--alpha", dest="alpha", type="float", help="momentum", default=0.0)
    parser.add_option_group(g)

    ############################################################################

    m = ["lift:<target_class>", "pp", "fmeasure:<target_class>","tester","auc:<target_class>",""]

    g = OptionGroup(parser, "Output")
    g.add_option("-m", "--metrics", dest="metrics", type="choice", help="metrics to display (comma separated) %s"%str(filter(len,m)), default="", choices=m)
    g.add_option("--probstr", dest="probstrain", help="write p(c|x) of train data to file", metavar="FILE")
    g.add_option("--probste", dest="probstest", help="write p(c|x) of test data to file", metavar="FILE")
    g.add_option("-v", "--verbosity", dest="verbosity", help="verbosity level [0,1,..]", default=1)
    g.add_option("-s", "--savemodel", dest="savemodel", help="save model to file", metavar="FILE")
    g.add_option("-b", "--interactive", dest="interactive", help="show progress bar", default="True")
    parser.add_option_group(g)

    ############################################################################

    (options, args) = parser.parse_args(args=None, values=None)

    config = ConfigParser.SafeConfigParser(options.__dict__)
    if options.config: config.read(options.config)

    return config
