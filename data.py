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

import numpy, gzip, sys, logging

class Data:

    def __init__(self, filename, format, verbosity=0):
        self.cls  = -1
        self.feat = -1
        self.data = []
        self.targets = []
        self.verbosity = verbosity

        if verbosity == 1: 
            sys.stderr.write("Reading '%s' (%s)... " % (filename, format))

        if filename.endswith( ".gz" ):
            self.open = gzip.open
        else:
            self.open = open

        try:
            if format == "jf":
                self.readJf(filename)
    
            elif format.startswith("csv") and ":" in format:
                _,delimiter = format.split(":")
                self.readCSV(filename, delimiter)
    
            elif format == "svm":
                self.readSVM(filename)
    
            else: 
                logging.error('Unknown format: %s' % format)
                sys.exit(-1)

        except IOError, e:
            logging.error("file not found!")
            sys.exit(-1)

        if verbosity == 1: 
            sys.stderr.write("done: %d classes, %d attributes, %d instances\n" % \
                (self.cls, self.feat, len(self.data)))

        self.__iter__ = self.data.__iter__

##############################################################################

    def __len__(self): return len(self.data)

##############################################################################

    def readJf(self, filename):
        first = True
    
        for l in self.open(filename):
            if l.startswith("#"): continue
            if first: 
                self.cls, self.feat = map(int, l.split())
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
        """
        Target variable must be in first column.
        """
        import csv
        for line in csv.reader(self.open(filename), delimiter=delimiter):
            if line[0].startswith("#"): continue
            vec = numpy.matrix( map(float, line[1:])+[1.0] ) 
            self.data.append( vec )
            self.targets.append( int(line[0]) )

        self.feat = len( self.data[0] )
        self.cls  = len( set( self.targets ) )

##############################################################################

    def readSVM(self, filename):
        self.cls = 2

        # find number of attributes
        maxfeat = -1
        for line in self.open(filename):
            if line.startswith("#"): continue
            line = line.split()

            if line[0] == "-1": self.target.append(0)
            else: self.target.append(1)
    
            last = int( line[-1].split(":")[0] )
            maxfeat = max(maxfeat, last)
           
        self.feat = maxfeat + 1

        # fill self.data
        for line in self.open(filename):
            if line.startswith("#"): continue
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
