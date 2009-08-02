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

import ConfigParser, logging, string, sys
from nn import NN
from data import Data

def main(config):
    net = NN(config)

    model = config.get("Input", "loadmodel")
    if model:
        net.loadmodel(model)
    else:
        dataTrain = Data(config.get("Input","train"), config.get("Input","format"), net.verbosity)
        try:
            trpr = net.train( dataTrain )
        except KeyboardInterrupt:
            sys.stderr.write("Aborting the training procedure...\n")
            pass
    
    ctest = config.get("Input","test")
    if ctest:
        dataTest = Data(ctest, config.get("Input","format"), net.verbosity)
        conf, err, tepr = net.test( dataTest )

        output = net.metrics.obtain( dataTest, tepr, conf, err )
        print 
        print "Test statistics:"
        print conf
        print "\n".join(map(string.strip,filter(len,output.values())))

    ftr = config.get("Output", "probstrain")
    fte = config.get("Output", "probstest")
    if ftr: Data.writeProbs(trpr, ftr)
    if fte: Data.writeProbs(tepr, fte)

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

    from usage import *
    config = usage()

    main(config)
