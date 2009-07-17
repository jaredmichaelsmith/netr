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

import numpy

class Data:
    def __init__(self, filename, format):
        self.cls  = -1
        self.feat = -1
        self.data = []
        self.targets = []

        if format == "jf":
            self.readJf(filename)

        elif format.startswith("csv"):
            _,delimiter = format.split(":")
            self.readCSV(filename, delimiter)

        elif format == "svm":
            self.readSVM(filename)

##############################################################################

    def __len__(self): return len(self.data)

##############################################################################

    def readJf(self, filename):
        first = True
    
        for l in file(filename).readlines():
            if l.startswith("#"): continue
            if first: 
                self.cls, self.feat = [ int(s) for s in l.split() ]
                first = False
                continue
    
            if l.startswith("-1"): break
   
            line = l.split()
            vec = numpy.matrix( map(float, line[1:])+[1.0] ) # <- last one for bias
            self.data.append( vec )
            self.targets.append( int(line[0]) )

        self.feat += 1
    
##############################################################################

    def readCSV(self, filename, delimiter):
        import csv
        for line in csv.reader(open(filename), delimiter=delimiter):
            vec = numpy.matrix( map(float, line[1:])+[1.0] ) 
            self.data.append( vec )
            self.targets.append( int(line[0]) )

        self.feat = len( self.data[0] )
        self.cls  = len( set( self.targets ) )

##############################################################################

    def readSVM(self, filename):
        self.cls = 2
        maxfeat = -1
        for line in file(filename).readlines():
            line = line.split()

            if line[0] == "-1": self.target.append(0)
            else: self.target.append(1)
    
            last = int( line[-1].split(":")[0] )
            maxfeat = max(maxfeat, last)
           
        self.feat = maxfeat + 1

        for line in file(filename).readlines():
            line = line.split()
            vec = numpy.zeros( [1,self.feat] ) 

            for l in line[1:]:
                k,v = l.split(":")
                vec[1,int(k)] = float(v)
            vec[1,-1] = 1.0

            self.data.append( vec )

##############################################################################

    @staticmethod
    def writeProbs(probs, filename):
        f = open(filename,'w')
        for pr in probs:
            f.write(" ".join("%.6f"%s for s in pr.flat))
            f.write("\n")
        f.close()

##############################################################################
