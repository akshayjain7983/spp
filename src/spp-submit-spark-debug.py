import sys;sys.path.append(r'/run/media/WORK/Eclipse/application/install/p2/pool/plugins/org.python.pydev.core_11.0.3.202310301107/pysrc')
import pydevd;pydevd.settrace()

import sys

from spp_ml_predictor import __main__ as spp

spp.main(sys.argv[1:])