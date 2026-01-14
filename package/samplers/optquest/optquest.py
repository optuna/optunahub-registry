# mypy: ignore-errors

# MIT License
#
# Copyright (c) 2026 OptTek Systems, Inc
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import ctypes
from ctypes import c_char_p
from ctypes import c_double
from ctypes import c_int
from ctypes import c_void_p
from ctypes import CFUNCTYPE
from ctypes import POINTER
import os
import sys
import threading


oq_path = os.path.abspath(__file__)
oq_dir = os.path.dirname(oq_path)
dll_path = oq_dir

if sys.platform.startswith("win"):
    dll_file = "optquestlib.dll"
    engine_file = None  # windows finds the engine automatically, we don't need to point to it
else:
    dll_file = "libOptQuestLib.so"
    engine_file = "libOptQuestEngine.so"

# Constants (unchanged)
OQStatisticNone = 0
OQStatisticMean = 1
OQStatisticMedian = 2
OQStatisticPercentile = 3
OQStatisticStdDev = 4
OQStatisticVariance = 6
OQStatisticCoeffOfVar = 8
OQStatisticMin = 14
OQStatisticMax = 15
OQStatisticSum = 16

OQConfidenceType1 = 1
OQConfidenceType2 = 2
OQConfidenceLevel80 = 1
OQConfidenceLevel90 = 2
OQConfidenceLevel95 = 3
OQConfidenceLevel98 = 4
OQConfidenceLevel99 = 5
OQConfidenceLevel999 = 6

OQSampleMethodStochastic = 0b00100000000000000000000000000000
OQSampleMethodDynamic = 0b00010000000000000000000000000000
OQSampleMethodAsync = 0b00001000000000000000000000000000
OQSampleMethodVariance = 0b0000001
OQSampleMethodUniform = 0b0000010
OQSampleMethodZero = 0b0000100
OQSampleMethodGradient = 0b0001000
OQSampleMethodMin = 0b0010000
OQSampleMethodMax = 0b0100000
OQSampleMethodRDS = 0b1000000

OQSampleVariogramSpherical = "SPHERICAL"
OQSampleVariogramPow = "POW"
OQSampleVariogramExp = "EXP"
OQSampleVariogramGaussian = "GAUSSIAN"
OQSampleVariogramTanh = "TANH"
OQSampleVariogramSech = "SECH"
OQSampleVariogramLinear = "LINEAR"
OQSampleVariogramQuadratic = "QUADRATIC"
OQSampleVariogramPartition = "PARTITION"
OQSampleVariogramMatern = "MATERN"

OQTerminationReasonNotStarted = 0
OQTerminationReasonRunning = 1
OQTerminationReasonLP = 3
OQTerminationReasonAutoStop = 4
OQTerminationReasonOptimalFound = 5
OQTerminationReasonMaxIterations = 6
OQTerminationReasonMaxTime = 7
OQTerminationReasonUserStopped = 8
OQTerminationReasonBestFound = 9
OQTerminationReasonException = 10
OQTerminationReasonFactoriesDone = 11
OQTerminationReasonInfeasible = 12
OQTerminationReasonCannotGenerate = 13
OQTerminationReasonMaxDemoIterations = 14

# Define pointer types for OptQuest handles
HOptQuestModel = c_void_p
HOptQuestSolution = c_void_p

# Define array types for complex inputs
DoubleArray = POINTER(c_double)
DoubleArray2D = POINTER(POINTER(c_double))
StringArray = POINTER(c_char_p)
StringArray2D = POINTER(POINTER(c_char_p))

# Define callback function types
OQEvaluator = CFUNCTYPE(c_int, HOptQuestSolution, c_void_p)
OQMonitor = CFUNCTYPE(c_int, HOptQuestSolution, c_void_p)

# Shared DLL instance and initialization lock


class OptQuestBase:
    _dll = None
    _dll_lock = threading.Lock()

    @classmethod
    def _initialize_dll(cls):
        # print("Initializing OptQuest DLL...")
        if cls._dll is None:
            # print("Loading OptQuest DLL...")
            with cls._dll_lock:
                # print("Acquired DLL lock.")
                if cls._dll is None:
                    # print("DLL not loaded yet, proceeding to load.")
                    full_engine_path = os.path.join(dll_path, engine_file) if engine_file else None
                    full_dll_path = os.path.join(dll_path, dll_file)
                    mode = (
                        ctypes.RTLD_GLOBAL
                        if hasattr(ctypes, "RTLD_GLOBAL")
                        else 0 | ctypes.RTLD_NOW
                        if hasattr(ctypes, "RTLD_NOW")
                        else 0
                    )
                    if not os.path.exists(full_dll_path):
                        raise FileNotFoundError("DLL not found:", full_dll_path)
                    if full_engine_path is not None:  # on linux we need to load the engine first
                        # print("Loading OptQuest Engine DLL...")
                        cls._engine = ctypes.CDLL(full_engine_path, mode=mode)
                    # print("Loading OptQuest Main DLL...")
                    cls._dll = ctypes.CDLL(full_dll_path, mode=mode)  # Use CDLL for __cdecl
                    cls._set_function_prototypes()
                    # print("OptQuest DLL loaded successfully.")

    @classmethod
    def _set_function_prototypes(cls):
        # Define function prototypes for both classes
        cls._dll.OQCreateModel.restype = HOptQuestModel
        cls._dll.OQCreateModel.argtypes = []
        cls._dll.OQDeleteModel.restype = None
        cls._dll.OQDeleteModel.argtypes = [HOptQuestModel]
        cls._dll.OQSetLicense.restype = c_int
        cls._dll.OQSetLicense.argtypes = [HOptQuestModel, c_char_p]
        cls._dll.OQLogSetup.restype = c_int
        cls._dll.OQLogSetup.argtypes = [HOptQuestModel, c_char_p]
        cls._dll.OQLogSolutions.restype = c_int
        cls._dll.OQLogSolutions.argtypes = [HOptQuestModel, c_char_p]
        cls._dll.OQAddContinuousVariable.restype = c_int
        cls._dll.OQAddContinuousVariable.argtypes = [HOptQuestModel, c_char_p, c_double, c_double]
        cls._dll.OQAddIntegerVariable.restype = c_int
        cls._dll.OQAddIntegerVariable.argtypes = [HOptQuestModel, c_char_p, c_double, c_double]
        cls._dll.OQAddDiscreteVariable.restype = c_int
        cls._dll.OQAddDiscreteVariable.argtypes = [
            HOptQuestModel,
            c_char_p,
            c_double,
            c_double,
            c_double,
        ]
        cls._dll.OQAddBinaryVariable.restype = c_int
        cls._dll.OQAddBinaryVariable.argtypes = [HOptQuestModel, c_char_p]
        cls._dll.OQAddDesignVariable.restype = c_int
        cls._dll.OQAddDesignVariable.argtypes = [HOptQuestModel, c_char_p, c_int]
        cls._dll.OQAddEnumerationVariable.restype = c_int
        cls._dll.OQAddEnumerationVariable.argtypes = [HOptQuestModel, c_char_p, DoubleArray, c_int]
        cls._dll.OQAddPermutationVariable.restype = c_int
        cls._dll.OQAddPermutationVariable.argtypes = [HOptQuestModel, c_char_p, StringArray, c_int]
        cls._dll.OQAddTupleVariable.restype = c_int
        cls._dll.OQAddTupleVariable.argtypes = [
            HOptQuestModel,
            c_char_p,
            DoubleArray2D,
            c_int,
            c_int,
        ]
        cls._dll.OQAddGeolocationVariable.restype = c_int
        cls._dll.OQAddGeolocationVariable.argtypes = [
            HOptQuestModel,
            c_char_p,
            DoubleArray2D,
            c_int,
        ]
        cls._dll.OQAddSelectionVariable.restype = c_int
        cls._dll.OQAddSelectionVariable.argtypes = [HOptQuestModel, c_char_p, StringArray, c_int]
        cls._dll.OQAddExchangeableGroup.restype = c_int
        cls._dll.OQAddExchangeableGroup.argtypes = [
            HOptQuestModel,
            c_char_p,
            StringArray2D,
            c_int,
            c_int,
        ]
        cls._dll.OQAddVariableExclusiveRange.restype = c_int
        cls._dll.OQAddVariableExclusiveRange.argtypes = [
            HOptQuestModel,
            c_char_p,
            c_double,
            c_double,
        ]
        cls._dll.OQAddOutputVariable.restype = c_int
        cls._dll.OQAddOutputVariable.argtypes = [HOptQuestModel, c_char_p]
        cls._dll.OQSetOutputStatistic.restype = c_int
        cls._dll.OQSetOutputStatistic.argtypes = [HOptQuestModel, c_char_p, c_int, c_double]
        cls._dll.OQAddExpressionVariable.restype = c_int
        cls._dll.OQAddExpressionVariable.argtypes = [HOptQuestModel, c_char_p, c_char_p]
        cls._dll.OQAddObjective.restype = c_int
        cls._dll.OQAddObjective.argtypes = [HOptQuestModel, c_char_p, c_char_p, c_char_p]
        cls._dll.OQAddMinimizeObjective.restype = c_int
        cls._dll.OQAddMinimizeObjective.argtypes = [HOptQuestModel, c_char_p, c_char_p]
        cls._dll.OQAddMaximizeObjective.restype = c_int
        cls._dll.OQAddMaximizeObjective.argtypes = [HOptQuestModel, c_char_p, c_char_p]
        cls._dll.OQSetObjectiveGoal.restype = c_int
        cls._dll.OQSetObjectiveGoal.argtypes = [HOptQuestModel, c_char_p, c_double, c_double]
        cls._dll.OQSetObjectiveConfidence.restype = c_int
        cls._dll.OQSetObjectiveConfidence.argtypes = [
            HOptQuestModel,
            c_char_p,
            c_int,
            c_int,
            c_double,
        ]
        cls._dll.OQSetObjectiveStatistic.restype = c_int
        cls._dll.OQSetObjectiveStatistic.argtypes = [HOptQuestModel, c_char_p, c_int, c_double]
        cls._dll.OQAddConstraint.restype = c_int
        cls._dll.OQAddConstraint.argtypes = [HOptQuestModel, c_char_p, c_char_p]
        cls._dll.OQSetReplications.restype = c_int
        cls._dll.OQSetReplications.argtypes = [HOptQuestModel, c_int, c_int]
        cls._dll.OQSetSerialReplications.restype = c_int
        cls._dll.OQSetSerialReplications.argtypes = [HOptQuestModel]
        cls._dll.OQSetRandomSeed.restype = c_int
        cls._dll.OQSetRandomSeed.argtypes = [HOptQuestModel, c_int]
        cls._dll.OQStopOnSuccessfulMILP.restype = c_int
        cls._dll.OQStopOnSuccessfulMILP.argtypes = [HOptQuestModel]
        cls._dll.OQAddSampleMetric.restype = c_int
        cls._dll.OQAddSampleMetric.argtypes = [HOptQuestModel, c_char_p, c_char_p, c_char_p]
        cls._dll.OQSetSampleMethod.restype = c_int
        cls._dll.OQSetSampleMethod.argtypes = [HOptQuestModel, c_int]
        cls._dll.OQSetSampleVariogram.restype = c_int
        cls._dll.OQSetSampleVariogram.argtypes = [HOptQuestModel, c_char_p]
        cls._dll.OQSetSampleRDSParams.restype = c_int
        cls._dll.OQSetSampleRDSParams.argtypes = [
            HOptQuestModel,
            c_char_p,
            c_char_p,
            DoubleArray,
            c_int,
        ]
        cls._dll.OQDescribe.restype = c_char_p
        cls._dll.OQDescribe.argtypes = [HOptQuestModel]
        cls._dll.OQInitialize.restype = c_int
        cls._dll.OQInitialize.argtypes = [HOptQuestModel]
        cls._dll.OQOptimize.restype = c_int
        cls._dll.OQOptimize.argtypes = [HOptQuestModel, c_int, OQEvaluator, OQMonitor, c_void_p]
        cls._dll.OQSample.restype = c_int
        cls._dll.OQSample.argtypes = [HOptQuestModel, c_int, OQEvaluator, OQMonitor, c_void_p]
        cls._dll.OQParallelOptimize.restype = c_int
        cls._dll.OQParallelOptimize.argtypes = [
            HOptQuestModel,
            c_int,
            c_int,
            OQEvaluator,
            OQMonitor,
            c_void_p,
        ]
        cls._dll.OQParallelSample.restype = c_int
        cls._dll.OQParallelSample.argtypes = [
            HOptQuestModel,
            c_int,
            c_int,
            OQEvaluator,
            OQMonitor,
            c_void_p,
        ]
        cls._dll.OQGetBestSolutions.restype = c_int
        cls._dll.OQGetBestSolutions.argtypes = [HOptQuestModel, c_int]
        cls._dll.OQGetBestSolutionsAt.restype = HOptQuestSolution
        cls._dll.OQGetBestSolutionsAt.argtypes = [HOptQuestModel, c_int]
        cls._dll.OQGetAllSolutions.restype = c_int
        cls._dll.OQGetAllSolutions.argtypes = [HOptQuestModel]
        cls._dll.OQGetAllSolutionsAt.restype = HOptQuestSolution
        cls._dll.OQGetAllSolutionsAt.argtypes = [HOptQuestModel, c_int]
        cls._dll.OQGetNumberOfCompletedIterations.restype = c_int
        cls._dll.OQGetNumberOfCompletedIterations.argtypes = [HOptQuestModel]
        cls._dll.OQGetTerminationReason.restype = c_int
        cls._dll.OQGetTerminationReason.argtypes = [HOptQuestModel]
        cls._dll.OQGetLastError.restype = c_char_p
        cls._dll.OQGetLastError.argtypes = [HOptQuestModel]
        cls._dll.OQInterrupt.restype = None
        cls._dll.OQInterrupt.argtypes = [HOptQuestModel]
        cls._dll.OQContinue.restype = None
        cls._dll.OQContinue.argtypes = [HOptQuestModel]
        cls._dll.OQCreateSolution.restype = HOptQuestSolution
        cls._dll.OQCreateSolution.argtypes = [HOptQuestModel]
        cls._dll.OQGetEmptySolution.restype = HOptQuestSolution
        cls._dll.OQGetEmptySolution.argtypes = [HOptQuestModel]
        cls._dll.OQCopySolution.restype = HOptQuestSolution
        cls._dll.OQCopySolution.argtypes = [HOptQuestModel, HOptQuestSolution]
        cls._dll.OQAddAllSolutions.restype = c_int
        cls._dll.OQAddAllSolutions.argtypes = [HOptQuestModel, HOptQuestModel]
        cls._dll.OQIsValid.restype = c_int
        cls._dll.OQIsValid.argtypes = [HOptQuestSolution]
        cls._dll.OQGetIteration.restype = c_int
        cls._dll.OQGetIteration.argtypes = [HOptQuestSolution]
        cls._dll.OQGetReplication.restype = c_int
        cls._dll.OQGetReplication.argtypes = [HOptQuestSolution]
        cls._dll.OQGetNumberOfCompletedReplications.restype = c_int
        cls._dll.OQGetNumberOfCompletedReplications.argtypes = [HOptQuestSolution]
        cls._dll.OQIsComplete.restype = c_int
        cls._dll.OQIsComplete.argtypes = [HOptQuestSolution]
        cls._dll.OQIsFeasible.restype = c_int
        cls._dll.OQIsFeasible.argtypes = [HOptQuestSolution]
        cls._dll.OQHasMetConfidence.restype = c_int
        cls._dll.OQHasMetConfidence.argtypes = [HOptQuestSolution]
        cls._dll.OQIsEvaluated.restype = c_int
        cls._dll.OQIsEvaluated.argtypes = [HOptQuestSolution]
        cls._dll.OQGetValue.restype = c_double
        cls._dll.OQGetValue.argtypes = [HOptQuestSolution, c_char_p]
        cls._dll.OQSetValue.restype = c_int
        cls._dll.OQSetValue.argtypes = [HOptQuestSolution, c_char_p, c_double]
        cls._dll.OQSubmit.restype = c_int
        cls._dll.OQSubmit.argtypes = [HOptQuestSolution]
        cls._dll.OQReject.restype = c_int
        cls._dll.OQReject.argtypes = [HOptQuestSolution]
        cls._dll.OQPredict.restype = c_int
        cls._dll.OQPredict.argtypes = [HOptQuestSolution]
        cls._dll.OQUpdate.restype = c_int
        cls._dll.OQUpdate.argtypes = [HOptQuestSolution]
        cls._dll.OQAddSuggested.restype = c_int
        cls._dll.OQAddSuggested.argtypes = [HOptQuestSolution]
        cls._dll.OQAddEvaluated.restype = c_int
        cls._dll.OQAddEvaluated.argtypes = [HOptQuestSolution]
        cls._dll.OQAddRequired.restype = c_int
        cls._dll.OQAddRequired.argtypes = [HOptQuestSolution]
        cls._dll.OQGetLastSolutionError.restype = c_char_p
        cls._dll.OQGetLastSolutionError.argtypes = [HOptQuestSolution]
        cls._dll.OQDeleteSolution.restype = None
        cls._dll.OQDeleteSolution.argtypes = [HOptQuestSolution]


def Evaluator(func):
    def callback(hsol, context):
        return func(OptQuestSolution(hsol))

    return OQEvaluator(callback)


def Monitor(func):
    def callback(hsol, context):
        return func(OptQuestSolution(hsol))

    return OQMonitor(callback)


class OptQuestModel(OptQuestBase):
    def __init__(self):
        self._initialize_dll()  # Ensure DLL is initialized
        self.model = self._dll.OQCreateModel()
        if not self.model:
            error = self.get_last_error()
            raise ValueError("Failed to create model:", error)

    def __del__(self):
        if hasattr(self, "model") and self.model:
            self._dll.OQDeleteModel(self.model)
            self.model = None

    def set_license(self, license_id):
        return int(self._dll.OQSetLicense(self.model, c_char_p(license_id.encode("utf-8")))) != 0

    def log_setup(self, file_spec):
        return int(self._dll.OQLogSetup(self.model, c_char_p(file_spec.encode("utf-8")))) != 0

    def log_solutions(self, file_spec):
        return int(self._dll.OQLogSolutions(self.model, c_char_p(file_spec.encode("utf-8")))) != 0

    def add_continuous_variable(self, name, min_val, max_val):
        return (
            int(
                self._dll.OQAddContinuousVariable(
                    self.model,
                    c_char_p(name.encode("utf-8")),
                    c_double(min_val),
                    c_double(max_val),
                )
            )
            != 0
        )

    def add_integer_variable(self, name, min_val, max_val):
        return (
            int(
                self._dll.OQAddIntegerVariable(
                    self.model,
                    c_char_p(name.encode("utf-8")),
                    c_double(min_val),
                    c_double(max_val),
                )
            )
            != 0
        )

    def add_discrete_variable(self, name, min_val, max_val, step):
        return (
            int(
                self._dll.OQAddDiscreteVariable(
                    self.model,
                    c_char_p(name.encode("utf-8")),
                    c_double(min_val),
                    c_double(max_val),
                    c_double(step),
                )
            )
            != 0
        )

    def add_binary_variable(self, name):
        return int(self._dll.OQAddBinaryVariable(self.model, c_char_p(name.encode("utf-8")))) != 0

    def add_design_variable(self, name, num):
        return (
            int(
                self._dll.OQAddDesignVariable(
                    self.model, c_char_p(name.encode("utf-8")), c_int(num)
                )
            )
            != 0
        )

    def add_enumeration_variable(self, name, values):
        n = len(values)
        if n == 0:
            return 0
        c_values = (c_double * n)(*values)
        return (
            int(
                self._dll.OQAddEnumerationVariable(
                    self.model, c_char_p(name.encode("utf-8")), c_values, c_int(n)
                )
            )
            != 0
        )

    def add_permutation_variable(self, group, names):
        if isinstance(names, str):
            names = [names]
        n = len(names)
        if n == 0:
            return 0
        keep_alive = [name.encode("utf-8") for name in names]  # keep c strings alive
        c_names = (c_char_p * n)(*keep_alive)
        return (
            int(
                self._dll.OQAddPermutationVariable(
                    self.model, c_char_p(group.encode("utf-8")), c_names, c_int(n)
                )
            )
            != 0
        )

    def add_tuple_variable(self, name, tuples):
        nt = len(tuples)
        if nt == 0:
            return 0
        nv = len(tuples[0])
        if nv == 0:
            return 0
        c_tuples = (POINTER(c_double) * nt)()
        for i, tup in enumerate(tuples):
            if len(tup) != nv:
                return 0
            c_tuples[i] = (c_double * nv)(*tup)
        return (
            int(
                self._dll.OQAddTupleVariable(
                    self.model, c_char_p(name.encode("utf-8")), c_tuples, c_int(nt), c_int(nv)
                )
            )
            != 0
        )

    def add_geolocation_variable(self, name, locations):
        nl = len(locations)
        if nl == 0:
            return 0
        nv = len(locations[0])
        if nv != 2:  # Assuming [lat, lon]
            return 0
        c_locations = (POINTER(c_double) * nl)()
        for i, loc in enumerate(locations):
            if len(loc) != nv:
                return 0
            c_locations[i] = (c_double * nv)(*loc)
        return (
            int(
                self._dll.OQAddGeolocationVariable(
                    self.model, c_char_p(name.encode("utf-8")), c_locations, c_int(nl)
                )
            )
            != 0
        )

    def add_selection_variable(self, name, selVars):
        if isinstance(selVars, str):
            selVars = [selVars]
        n = len(selVars)
        if n == 0:
            return 0
        keep_alive = [sv.encode("utf-8") for sv in selVars]  # keep c strings alive
        c_names = (c_char_p * n)(*keep_alive)
        return (
            int(
                self._dll.OQAddSelectionVariable(
                    self.model, c_char_p(name.encode("utf-8")), c_names, c_int(n)
                )
            )
            != 0
        )

    def add_exchangeable_group(self, name, group):
        ng = len(group)
        if ng == 0:
            return 0
        nv = len(group[0])
        if nv == 0:
            return 0
        keep_alive = []  # we have to keep the c strings alive so they are not garbage collected
        c_group = (POINTER(c_char_p) * ng)()
        for i, grp in enumerate(group):
            if len(grp) != nv:
                return 0
            row = (c_char_p * nv)()
            for j, var in enumerate(grp):
                keep_alive.append(var.encode("utf-8"))
                row[j] = c_char_p(keep_alive[-1])
            c_group[i] = row

        return (
            int(
                self._dll.OQAddExchangeableGroup(
                    self.model, c_char_p(name.encode("utf-8")), c_group, c_int(ng), c_int(nv)
                )
            )
            != 0
        )

    def add_variable_exclusive_range(self, name, min_val, max_val):
        return (
            int(
                self._dll.OQAddVariableExclusiveRange(
                    self.model,
                    c_char_p(name.encode("utf-8")),
                    c_double(min_val),
                    c_double(max_val),
                )
            )
            != 0
        )

    def add_output_variable(self, name):
        return int(self._dll.OQAddOutputVariable(self.model, c_char_p(name.encode("utf-8")))) != 0

    def set_output_statistic(self, name, statistic, statistic_value=None):
        if statistic_value is None:
            statistic_value = 0
        return (
            int(
                self._dll.OQSetOutputStatistic(
                    self.model,
                    c_char_p(name.encode("utf-8")),
                    c_int(statistic),
                    c_double(statistic_value),
                )
            )
            != 0
        )

    def add_expression_variable(self, name, expression):
        return (
            int(
                self._dll.OQAddExpressionVariable(
                    self.model,
                    c_char_p(name.encode("utf-8")),
                    c_char_p(expression.encode("utf-8")),
                )
            )
            != 0
        )

    def add_objective(self, name, direction, expression):
        return (
            int(
                self._dll.OQAddObjective(
                    self.model,
                    c_char_p(name.encode("utf-8")),
                    c_char_p(direction.encode("utf-8")),
                    c_char_p(expression.encode("utf-8")),
                )
            )
            != 0
        )

    def add_minimize_objective(self, name, expression):
        return (
            int(
                self._dll.OQAddMinimizeObjective(
                    self.model,
                    c_char_p(name.encode("utf-8")),
                    c_char_p(expression.encode("utf-8")),
                )
            )
            != 0
        )

    def add_maximize_objective(self, name, expression):
        return (
            int(
                self._dll.OQAddMaximizeObjective(
                    self.model,
                    c_char_p(name.encode("utf-8")),
                    c_char_p(expression.encode("utf-8")),
                )
            )
            != 0
        )

    def set_objective_goal(self, name, min_val, max_val):
        return (
            int(
                self._dll.OQSetObjectiveGoal(
                    self.model,
                    c_char_p(name.encode("utf-8")),
                    c_double(min_val),
                    c_double(max_val),
                )
            )
            != 0
        )

    def set_objective_confidence(self, name, ctype, clevel, epct):
        return (
            int(
                self._dll.OQSetObjectiveConfidence(
                    self.model,
                    c_char_p(name.encode("utf-8")),
                    c_int(ctype),
                    c_int(clevel),
                    c_double(epct),
                )
            )
            != 0
        )

    def set_objective_statistic(self, name, statistic, statistic_value):
        return (
            int(
                self._dll.OQSetObjectiveStatistic(
                    self.model,
                    c_char_p(name.encode("utf-8")),
                    c_int(statistic),
                    c_double(statistic_value),
                )
            )
            != 0
        )

    def add_constraint(self, name, expression):
        return (
            int(
                self._dll.OQAddConstraint(
                    self.model,
                    c_char_p(name.encode("utf-8")),
                    c_char_p(expression.encode("utf-8")),
                )
            )
            != 0
        )

    def set_replications(self, min_reps, max_reps=None):
        if max_reps is None:
            max_reps = min_reps
        return int(self._dll.OQSetReplications(self.model, c_int(min_reps), c_int(max_reps))) != 0

    def set_serial_replications(self):
        return int(self._dll.OQSetSerialReplications(self.model)) != 0

    def set_random_seed(self, seed):
        return int(self._dll.OQSetRandomSeed(self.model, seed)) != 0

    def stop_on_successful_milp(self):
        return int(self._dll.OQStopOnSuccessfulMILP(self.model)) != 0

    def add_sample_metric(self, name, response, error):
        return (
            int(
                self._dll.OQAddSampleMetric(
                    self.model,
                    c_char_p(name.encode("utf-8")),
                    c_char_p(response.encode("utf-8")),
                    c_char_p(error.encode("utf-8")),
                )
            )
            != 0
        )

    def set_sample_method(self, method):
        return int(self._dll.OQSetSampleMethod(self.model, c_int(method))) != 0

    def set_sample_variogram(self, variogram):
        return (
            int(self._dll.OQSetSampleVariogram(self.model, c_char_p(variogram.encode("utf-8"))))
            != 0
        )

    def set_sample_rds_params(self, name, distribution, rdsParams):
        n = len(rdsParams)
        if n == 0:
            return 0
        c_values = (c_double * n)(*rdsParams)
        return int(
            self._dll.OQSetSampleRDSParams(
                self.model,
                c_char_p(name.encode("utf-8")),
                c_char_p(distribution.encode("utf-8")),
                c_values,
                n,
            )
        )

    def set_sample_rds_uniform(self, name, nBins, a, b):
        return self.set_sample_rds_params(name, "UNIFORM", [nBins, a, b])

    def set_sample_rds_normal(self, name, nBins, mu, sigma):
        return self.set_sample_rds_params(name, "NORMAL", [nBins, mu, sigma])

    def set_sample_rds_lognormal(self, name, nBins, s, scale):
        return self.set_sample_rds_params(name, "LOGNORMAL", [nBins, s, scale])

    def set_sample_rds_beta(self, name, nBins, alpha, beta):
        return self.set_sample_rds_params(name, "BETA", [nBins, alpha, beta])

    def set_sample_rds_poisson(self, name, nBins, lam):
        return self.set_sample_rds_params(name, "POISSON", [nBins, lam])

    def set_sample_rds_exponential(self, name, nBins, scale):
        return self.set_sample_rds_params(name, "EXPONENTIAL", [nBins, scale])

    def set_sample_rds_cauchy(self, name, nBins, location, scale):
        return self.set_sample_rds_params(name, "CAUCHY", [nBins, location, scale])

    def set_sample_rds_gamma(self, name, nBins, a, scale):
        return self.set_sample_rds_params(name, "GAMMA", [nBins, a, scale])

    def set_sample_rds_chisquare(self, name, nBins, nu):
        return self.set_sample_rds_params(name, "CHISQUARE", [nBins, nu])

    def set_sample_rds_pareto(self, name, nBins, gamma, a, location, scale):
        return self.set_sample_rds_params(name, "PARETO", [nBins, gamma, a, location, scale])

    def set_sample_rds_t(self, name, nBins, nu):
        return self.set_sample_rds_params(name, "T", [nBins, nu])

    def set_sample_rds_weibull(self, name, nBins, c, scale):
        return self.set_sample_rds_params(name, "WEIBULL", [nBins, c, scale])

    def set_sample_rds_discrete(self, name, binEdges):
        return self.set_sample_rds_params(name, "DISCRETE", binEdges)

    def describe(self):
        lines = self._dll.OQDescribe(self.model)
        return ctypes.string_at(lines).decode("utf-8") if lines else ""

    def initialize(self):
        return int(self._dll.OQInitialize(self.model)) != 0

    def optimize(self, num_solutions, evaluator=None, monitor=None, npar=None):
        eval_func = Evaluator(evaluator) if evaluator else ctypes.cast(None, OQEvaluator)
        mon_func = Monitor(monitor) if monitor else ctypes.cast(None, OQMonitor)
        if npar is None:
            result = self._dll.OQOptimize(
                self.model, c_int(num_solutions), eval_func, mon_func, None
            )
        else:
            result = self._dll.OQParallelOptimize(
                self.model, c_int(npar), c_int(num_solutions), eval_func, mon_func, None
            )
        return int(result) != 0

    def sample(self, num_solutions, evaluator=None, monitor=None, npar=None):
        eval_func = Evaluator(evaluator) if evaluator else ctypes.cast(None, OQEvaluator)
        mon_func = Monitor(monitor) if monitor else ctypes.cast(None, OQMonitor)
        if npar is None:
            result = self._dll.OQSample(
                self.model, c_int(num_solutions), eval_func, mon_func, None
            )
        else:
            result = self._dll.OQParallelSample(
                self.model, c_int(npar), c_int(num_solutions), eval_func, mon_func, None
            )
        return int(result) != 0

    def get_best_solutions(self, limit=1):
        num_solutions = int(self._dll.OQGetBestSolutions(self.model, c_int(limit)))
        solutions = []
        if num_solutions >= 0:
            for i in range(num_solutions):
                sol_ptr = self._dll.OQGetBestSolutionsAt(self.model, c_int(i))
                if sol_ptr:
                    solutions.append(OptQuestSolution(sol_ptr, True))
        return solutions

    def get_all_solutions(self):
        num_solutions = int(self._dll.OQGetAllSolutions(self.model))
        solutions = []
        if num_solutions >= 0:
            for i in range(num_solutions):
                sol_ptr = self._dll.OQGetAllSolutionsAt(self.model, c_int(i))
                if sol_ptr:
                    solutions.append(OptQuestSolution(sol_ptr, True))
        return solutions

    def get_number_of_completed_iterations(self):
        return int(self._dll.OQGetNumberOfCompletedIterations(self.model))

    def get_termination_reason(self):
        return int(self._dll.OQGetTerminationReason(self.model))

    def get_last_error(self):
        error = self._dll.OQGetLastError(self.model)
        return ctypes.string_at(error).decode("utf-8") if error else ""

    def interrupt(self):
        self._dll.OQInterrupt(self.model)

    def continue_opt(self):
        self._dll.OQContinue(self.model)

    def create_solution(self):
        sol_ptr = self._dll.OQCreateSolution(self.model)
        return OptQuestSolution(sol_ptr, True) if sol_ptr else None

    def get_empty_solution(self):
        sol_ptr = self._dll.OQGetEmptySolution(self.model)
        return OptQuestSolution(sol_ptr, True) if sol_ptr else None

    def copy_solution(self, src_solution):
        sol_ptr = self._dll.OQCopySolution(self.model, src_solution.solution)
        return OptQuestSolution(sol_ptr, True) if sol_ptr else None

    def add_all_solutions(self, src_model):
        return int(self._dll.OQAddAllSolutions(self.model, src_model.model)) != 0


class OptQuestSolution(OptQuestBase):
    def __init__(self, solution, should_delete=False):
        self._initialize_dll()  # Ensure DLL is initialized
        self.solution = solution
        self.should_delete = should_delete

    def __del__(self):
        if hasattr(self, "solution") and self.solution and self.should_delete:
            self._dll.OQDeleteSolution(self.solution)
            self.solution = None

    def is_valid(self):
        return int(self._dll.OQIsValid(self.solution)) != 0

    def __bool__(self):
        return int(self._dll.OQIsValid(self.solution)) != 0

    def get_iteration(self):
        return int(self._dll.OQGetIteration(self.solution))

    def get_replication(self):
        return int(self._dll.OQGetReplication(self.solution))

    def get_number_of_completed_replications(self):
        return int(self._dll.OQGetNumberOfCompletedReplications(self.solution))

    def is_complete(self):
        return int(self._dll.OQIsComplete(self.solution)) != 0

    def is_feasible(self):
        return int(self._dll.OQIsFeasible(self.solution)) != 0

    def has_met_confidence(self):
        return int(self._dll.OQHasMetConfidence(self.solution)) != 0

    def is_evaluated(self):
        return int(self._dll.OQIsEvaluated(self.solution)) != 0

    def get_value(self, name):
        return float(self._dll.OQGetValue(self.solution, c_char_p(name.encode("utf-8"))))

    def set_value(self, name, value):
        return int(
            self._dll.OQSetValue(self.solution, c_char_p(name.encode("utf-8")), c_double(value))
        )

    def __getitem__(self, name):
        return float(self._dll.OQGetValue(self.solution, c_char_p(name.encode("utf-8"))))

    def __setitem__(self, name, value):
        self._dll.OQSetValue(self.solution, c_char_p(name.encode("utf-8")), c_double(value))

    def submit(self):
        return int(self._dll.OQSubmit(self.solution)) != 0

    def reject(self):
        return int(self._dll.OQReject(self.solution)) != 0

    def predict(self):
        return int(self._dll.OQPredict(self.solution)) != 0

    def update(self):
        return int(self._dll.OQUpdate(self.solution)) != 0

    def add_suggested(self):
        return int(self._dll.OQAddSuggested(self.solution)) != 0

    def add_evaluated(self):
        return int(self._dll.OQAddEvaluated(self.solution)) != 0

    def add_required(self):
        return int(self._dll.OQAddRequired(self.solution)) != 0

    def get_last_error(self):
        error = self._dll.OQGetLastSolutionError(self.solution)
        return ctypes.string_at(error).decode("utf-8") if error else ""
