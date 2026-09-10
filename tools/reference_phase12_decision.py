"""Compile archived C++ decision methods against small in-memory adapters."""

import ctypes as ct
import json
import subprocess
import tempfile
from dataclasses import replace
from pathlib import Path

import numpy as np
from scipy.special import expit

from mdanderson_stats.parallel_phase12_decision import (
    phase12_source_decision,
    phase12_source_final_selection,
)
from mdanderson_stats.parallel_phase12_model import fit_phase12_model

source = Path("research/raw/P12Xuelin/extracted/Phase12Xuelin/DFKernel/TrialDesign.cpp").read_text()


def method(name):
    start = source.index("void TrialDesign::" + name + "(")
    opening = source.index("{", start)
    depth = 1
    end = opening + 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[start:end]


header = r"""
#include <vector>
#include <algorithm>
using namespace std;
using DoubleVector=vector<double>;
struct Arm { bool closed=false; bool selected=false;
 int GetDoseCount(){return 6;} int GetCohortSize(){return 5;}
 void Close(){closed=true;} void Select(){selected=true;} };
struct StoppingRules { double pl=0,pu=1;
 bool IsFutile(double p){return (pl>0 && p<pl)||(pu>0 && p>pu);}
 bool IsSuperior(double p){return IsFutile(p);} };
struct Kernel { vector<double> m_vProbTox=vector<double>(6), probs=vector<double>(60);
 void SetData(vector<Arm*> a,double t){} void CalculateToxRate(){}
 void CalculateProbabilities(vector<double>& out){out=probs;} };
struct Randomizer {
 vector<int> m_vSuspended=vector<int>(6);
 vector<double> m_vRi=vector<double>(6);
};
struct DF { vector<int> m_vAdmissible=vector<int>(6); };
struct TrialDesign {
 double m_dCurrentTime=0; bool terminated=false;
 vector<Arm*> m_vArms; Kernel* m_pKernel; Randomizer* m_pRandomizer;
 vector<double> m_vPostProbs=vector<double>(60);
 vector<int> m_vClosed=vector<int>(6),m_vNumPatients=vector<int>(6);
 vector<int> m_vSelected=vector<int>(6),m_vFuture=vector<int>(6);
 vector<StoppingRules*> m_vStoppingRules; DF m_DF3Plus3;
 void TerminateTrial(){terminated=true;} bool IsTerminated(){return terminated;}
 void EvaluateStoppingRules(double); void CalculateToxRate(); void PickWinner(double);
};
"""
wrapper = r"""
extern "C" void run(double* p,double* state,double* result){
 Arm arm;Kernel k;Randomizer r;TrialDesign t;
 t.m_vArms.push_back(&arm); t.m_pKernel=&k; t.m_pRandomizer=&r;
 StoppingRules rules[6];rules[0].pu=.95;rules[1].pl=.05;rules[2].pu=.9;
 rules[3].pu=.8;rules[4].pu=.8;rules[5].pl=.01;
 for(int i=0;i<6;i++)t.m_vStoppingRules.push_back(&rules[i]);
 for(int i=0;i<54;i++)k.probs[i]=p[i];
 for(int i=0;i<6;i++){
  k.m_vProbTox[i]=p[54+i];t.m_DF3Plus3.m_vAdmissible[i]=state[i];
  t.m_vClosed[i]=state[6+i];r.m_vSuspended[i]=state[12+i];t.m_vNumPatients[i]=state[18+i];
 }
 t.EvaluateStoppingRules(0);
 int selected=-1, future=-1;
 for(int i=0;i<6;i++){
  result[i]=t.m_vClosed[i];result[6+i]=r.m_vSuspended[i];result[12+i]=r.m_vRi[i];
  if(t.m_vSelected[i])selected=i;
 }
 t.PickWinner(0);
 for(int i=0;i<6;i++)if(t.m_vFuture[i])future=i;
 result[18]=t.terminated;result[19]=arm.closed;result[20]=selected;result[21]=future;
}
"""
rng = np.random.default_rng(8531)
base = fit_phase12_model(np.zeros((6, 4)), draws=8, warmup=0, chains=2, rng=rng)
with tempfile.TemporaryDirectory(prefix="phase12-cpp-rules-") as temp:
    path = Path(temp)
    code = (
        header
        + "\n".join(method(n) for n in ["PickWinner", "EvaluateStoppingRules", "CalculateToxRate"])
        + wrapper
    )
    (path / "source.cpp").write_text(code)
    subprocess.run(
        [
            "c++",
            "-std=c++11",
            "-shared",
            "-fPIC",
            str(path / "source.cpp"),
            "-o",
            str(path / "source.dylib"),
        ],
        check=True,
    )
    lib = ct.CDLL(str(path / "source.dylib"))
    pointer = ct.POINTER(ct.c_double)
    lib.run.argtypes = [pointer] * 3
    selections = closures = futilities = 0
    for case in range(100):
        initial = rng.random(6) > 0.2
        initial[0] = True
        closed = ~initial | (rng.random(6) < 0.1)
        suspended = rng.random(6) < 0.1
        enrolled = rng.integers(0, 11, 6)
        score = rng.normal(0, 2, 6)
        pair = expit(score[:, None] - score[None, :])
        np.fill_diagonal(pair, 0)
        weight = pair[:, 0].copy()
        weight[0] = 0.5
        efficacy = rng.random(6)
        future = rng.random(6)
        toxicity = rng.random(6)
        if case % 5 == 0:
            efficacy[:] = 0.02
        elif case % 5 == 1:
            efficacy[2] = 0.99
            pair[2, :] = 0.99
            pair[2, 2] = 0
        if case % 10 == 1:
            closed[2] = True
        # Guarantee some open positive mass; all-closed zero division is tested separately.
        closed[0] = False
        toxicity[0] = 0.1
        fit = replace(
            base,
            reference_superiority=weight,
            efficacy_probability=efficacy,
            future_probability=future,
            pairwise_superiority=pair,
            toxicity_probability=toxicity,
        )
        out = phase12_source_decision(
            fit, enrolled, phase_one_admissible=initial, closed=closed, suspended=suspended
        )
        final = phase12_source_final_selection(fit, closed=out.closed, suspended=out.suspended)
        p = np.r_[weight, efficacy, future, pair.ravel(), toxicity].astype(float)
        state = np.r_[initial, closed, suspended, enrolled].astype(float)
        result = np.zeros(22)
        lib.run(*(x.ctypes.data_as(pointer) for x in (p, state, result)))
        np.testing.assert_array_equal(out.closed, result[:6])
        np.testing.assert_array_equal(out.suspended, result[6:12])
        np.testing.assert_allclose(out.probability, result[12:18], atol=1e-14)
        assert out.terminated == bool(result[18])
        assert out.arm_closed == bool(result[19])
        assert out.selected == (None if result[20] < 0 else int(result[20]))
        assert final == (None if result[21] < 0 else int(result[21]))
        selections += out.selected is not None
        closures += out.arm_closed
        futilities += out.reason == "futility"
    report = {
        "cases": 100,
        "seed": 8531,
        "early_selections": selections,
        "arm_closures": closures,
        "futility_stops": futilities,
        "scope": (
            "Unmodified extracted C++ PickWinner, EvaluateStoppingRules and CalculateToxRate "
            "methods with in-memory state and supplied posterior adapters; "
            "no numerical integration or calendar controller parity claimed."
        ),
    }
    Path("docs/phase12-decision-reference.json").write_text(json.dumps(report, indent=2) + "\n")
    print(report)
