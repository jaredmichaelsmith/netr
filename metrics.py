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

class Metric:
    def __init__(self, metrics):
        self.metrics = metrics.split(",")
        for i,m in enumerate(metrics):
            self.metrics.append(m)
            if not ":" in m: continue
            k,v = m.split(":")
            if k == "fmeasure": 
                self.fmeasure_target = int(v)
                self.metrics[i] = k
        

    def obtain( self, testdata, classes_test, classes, allprobs, tepr, conf, err ):
        output = {"lift":"","pp":"","fscore":"","tester":""}
        if "lift" in self.metrics:
            lift = getLift(tepr, classes_test)
            output["lift"] = " lift: %.4f" % lift
        if "pp" in self.metrics:
            logL = 0
            for n in range(len(allprobs)):
                try:
                    logL -= math.log(allprobs[n][0,classes[n]])
                except ValueError: pass
            pp = logL / len(classes)
            output["pp"] = " pp: %.4f" % pp
        if "fmeasure" in self.metrics:
            rec = conf[ self.fmeasure_target, self.fmeasure_target ] / conf[ self.fmeasure_target, : ].sum()
            pr  = conf[ self.fmeasure_target, self.fmeasure_target ] / conf[ :, self.fmeasure_target ].sum()
            fscore = 2*rec * pr / (rec+pr)
            output["fscore"] = " prec: %.2f rec: %.2f f-score: %.2f" % (pr, rec, fscore)
        if "tester" in self.metrics:
            output["tester"] = " test-er: %.4f" % err

        return output

##############################################################################
from itertools import izip
def getLift( allprobs, data ):
    #print allprobs
    #print data
    pc1 = len( filter( lambda x: x == 1.0, data ) )/float(len(data))
    probs = [ (p[0,1],d) for p,d in izip( allprobs, data ) ]
    #print probs
    probs.sort( key = lambda x: x[0], reverse = True )
    top = int( len( data )*0.2 )
    probs = probs[:top]
    pk1 = len( filter( lambda x: x[1] == 1, probs ) ) / float(top)
    if pc1 == 0:
        return 0.0
    else:
        return pk1/pc1
