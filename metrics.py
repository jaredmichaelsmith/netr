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

import math, numpy

class Metric:
    def __init__(self, metrics, cls):
        self.metrics = metrics.split(",")
        self.target  = cls

    def obtain( self, data, prediction, conf, err ):
        output = {"lift":"","pp":"","fscore":"","tester":"","auc":""}
        self.newtargets = []
        for k in data.targets:
            if k == self.target: self.newtargets.append( 1 )
            else: self.newtargets.append(0)

        if "lift" in self.metrics:
            lift = getLift(prediction, self.newtargets, self.target)
            output["lift"] = " lift: %.4f" % lift

        if "pp" in self.metrics:
            logL = 0
            for n in range(len(prediction)):
                try:
                    logL -= math.log(prediction[n][0,data.targets[n]])
                except ValueError: pass
            pp = logL / len(data)
            output["pp"] = " pp: %.4f" % pp

        if "fmeasure" in self.metrics:
            rec = conf[ self.target, self.target ] / float(conf[ self.target, : ].sum())
            pr  = conf[ self.target, self.target ] / float(conf[ :, self.target ].sum())
            fscore = 2*rec * pr / (rec+pr)
            output["fscore"] = " prec: %.2f rec: %.2f f-score: %.2f" % (pr, rec, fscore)
        if "tester" in self.metrics:
            output["tester"] = " test-er: %.4f" % err

        if "auc" in self.metrics:
            output["auc"] = " AUC: %.4f" % auc( prediction, self.newtargets, self.target )

        return output

##############################################################################
from itertools import izip
def getLift( prediction, targets, targetclass ):
    #print allprobs
    #print data
    pc1 = len( filter( lambda x: x == 1, targets ) )/float(len(targets))
    probs = [ (p[0,targetclass],d) for p,d in izip( prediction, targets ) ]
    #print probs
    probs.sort( key = lambda x: x[0], reverse = True )
    top = int( len( targets )*0.2 ) 
    probs = probs[:top]
    pk1 = len( filter( lambda x: x[1] == 1, probs ) ) / float(top)
    if pc1 == 0:
        return 0.0
    else:
        return pk1/pc1

def auc( prediction, target, targetclass ):
    assert len(target) == len(prediction)
    pairs_with_duplicate_probs = zip([p[0,targetclass] for p in prediction], target)
    pairs_with_duplicate_probs.sort()
    
    pairs = {}
    
    # accumulate seen zeros while removing duplicates for the initial confusion matrix
    zeros = 0
    for p, c in pairs_with_duplicate_probs:
        zeros += (1-c)
        try:
            pairs[p].append(c)
        except KeyError:
            pairs[p] = [c]
    ones = len(pairs_with_duplicate_probs) - zeros
    
    auc = 0
    # initialize confusion matrix
    c= numpy.zeros([2,2])
    c[0,1] = zeros
    c[1,1] = ones
    
    # initialize first point on the auc plane
    sens, spec = [ float(c[i,i]) / sum(c[i]) for i in [0,1] ]
    prec = c.diagonal().sum() / float( c.sum() )
    spec_prev, sens_prev = spec, sens
       
    # for each threshold teh confusion matrix may change 
    # depending only on the ammount of zeros and ones assigned to the probability
    for k in sorted(pairs.keys()):
        v = pairs[k]
        ones = sum(v)
        zeros = len(v) - ones
        c[0,0] += zeros
        c[0,1] -= zeros
        c[1,0] += ones
        c[1,1] -= ones
        # calculate new point for the auc plane
        sens, spec = [ float(c[i,i]) / sum(c[i]) for i in [0,1] ]
        prec = c.diagonal().sum() / float( c.sum() )
    
        # add area to auc. sens = x-axis, spec = y-axis
        width = (sens - sens_prev)
        auc += spec * width
        auc += width * 0.5 * (spec_prev - spec)
    
        # memorize the last point for the next area-calculation
        spec_prev, sens_prev = spec, sens
    
    return auc

