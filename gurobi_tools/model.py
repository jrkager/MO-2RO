"""``OptimizationModel``: the ``gurobipy.Model`` subclass used throughout.

Every model in this code base is built through this class rather than through
``gurobipy.Model`` directly, so that two settings hold everywhere without being
repeated at each call site:

* **silence** -- ``LogToConsole = 0`` (plus ``OutputFlag = 0`` on the default
  environment, set once at import time);
* **exact integrality** -- ``IntegralityFocus = 1``. With Gurobi's default
  ``IntFeasTol = 1e-5`` a reported solution may have binaries up to 1e-5 off
  0/1, whose objective values are then off by up to ~1e-4. The algorithm
  compares such values against slab boundaries and against independently
  recomputed Pareto-front points, so that error must be eliminated at the
  source rather than absorbed by a tolerance (see ``algorithm/numerics.py``).

It also accumulates solve times across ``optimize()`` calls (``_accRuntime``
wall, ``_accProctime`` CPU) and tracks created variables by name in ``_vars``.
"""

import gurobipy as gp
gp.setParam('OutputFlag', 0)
from gurobipy import GRB
import time


class InfeasibleException(Exception):
    pass


class OptimizationModel(gp.Model):

    def __init__(self, *argc, **argv):
        self._vars = dict()
        self._wasReset = False
        self._accRuntime = 0
        self._accProctime = 0
        super().__init__()
        self.Params.LogToConsole = 0
        # Return truly integral MIP solutions -- see the module docstring.
        self.Params.IntegralityFocus = 1
        for k in argv:
            if hasattr(self.Params, k):
                setattr(self.Params, k, argv[k])

    def optimize(self, *argc, **argv):
        start_proc = time.process_time()
        super().optimize(*argc, **argv)
        end_proc = time.process_time()
        self._accRuntime += self.runtime
        self._accProctime += end_proc - start_proc

    def _trackVars(self, name, var):
        if name:
            self._vars[name] = var
        else:
            i = 0
            while "var" + str(i) in self._vars:
                i += 1
            self._vars["var" + str(i)] = var
        return var

    def addVar(self, *argc, **argv):
        return self._trackVars(argv.get("name"), super().addVar(*argc, **argv))

    def addVars(self, *argc, **argv):
        return self._trackVars(argv.get("name"), super().addVars(*argc, **argv))

    def run(self, callback=None, print_status=False, force_rerun=False):
        """Optimize unless this model has already been solved."""
        if self.status == GRB.LOADED or force_rerun or self._wasReset:
            self.optimize(callback=callback)
            self._wasReset = False
        if print_status:
            self.printStatus(raiseExceptions=True)

    def printStatus(self, raiseExceptions=False):
        if self.status == GRB.LOADED:
            print("Model loaded")
        if self.status == GRB.OPTIMAL:
            print("Model optimal")
        if self.status == GRB.INFEASIBLE:
            print("Model is infeasible")
            if raiseExceptions: raise InfeasibleException()
        if self.status == GRB.INF_OR_UNBD:
            print("Model is infeasible or unbounded")
            if raiseExceptions: raise InfeasibleException("Model is infeasible or unbounded")
        if self.status == GRB.UNBOUNDED:
            print("Model is unbounded (and maybe infeasible)")
            if raiseExceptions: raise InfeasibleException("Model is unbounded (and maybe infeasible)")

    def getOptimum(self, callback=None, print_status=False, force_rerun=False):
        self.run(callback=callback, print_status=print_status, force_rerun=force_rerun)
        return self.objVal

    def getSolutionVars(self):
        self.run()
        return self._vars

    def hasRun(self):
        return self.status != GRB.LOADED
