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

import pickle, numpy, sys
from optparse import OptionParser

def loadmodel(modelname):
    w, c = pickle.load(file(modelname))
    return w

def main(options):
    weights = loadmodel(options.model)
    
    code = ""
    
    code += 'graph "%s" {' % options.model
    code += '	node [ fontname="Helvetica-Bold", shape="circle", width="0.3", '
    code += '		   style ="setlinewidth(2),filled" color="black", fillcolor = "greenyellow", label=""];'
    code += '	edge [ style="bold"];'
    code += '	graph[ nodesep=".35", ranksep="1.1" ];'
   
    # init nodes
    for layer,w in enumerate(weights):
        M,N = w.shape
        for i in range(M):
            code += ' W%d_%d[label="%d"]\n' % (layer,i,i)
    
        if layer == len(weights)-1:
            for i in range(N):
                code += ' W%d_%d[label="%d"]\n' % (layer+1,i,i)
    
    # declare edges
    s = ""
    for layer,w in enumerate(weights):
    
        # normalize line width
        w = 1.0/(1.0 + numpy.exp(-w))
        M,N = w.shape
    
        for i in range(M):
            for j in range(N):
                if w[i,j] < options.threshold: continue
                if options.labels:
                    s = ', label="%.2f"' % w[i,j]
                code += ' W%d_%d -- W%d_%d [ color="red", style="setlinewidth(%f)"%s ];\n' % (layer,i,layer+1,j,w[i,j],s)
    
    code += '}'
    print code

if __name__ == "__main__":
    parser = OptionParser(version="1.0", description="Copyright (C) 2009 Pavel Golik <http://code.google.com/p/netr>. This program comes with ABSOLUTELY NO WARRANTY; This is free software, and you are welcome to redistribute it under certain conditions; see 'LICENSE' for details.")

    parser.add_option("-t", "--threshold", dest="threshold", type="float", help="don't draw edges with weights less than <threshold>", default=0)
    parser.add_option("-l", "--labels", dest="labels", help="draw weights as edge labels", default=False, action="store_true")
    parser.add_option("-m", "--model", dest="model", help="model file (required)", metavar="FILE")
    (options, args) = parser.parse_args(args=None, values=None)

    if not options.model: 
        print parser.usage
    else:
        main(options)
