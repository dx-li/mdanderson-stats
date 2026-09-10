"""Exact coin-enumerated comparison with the original six-dose C++ phase-I methods."""

import ctypes as ct
import json
import re
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from mdanderson_stats.parallel_phase12_progression import phase12_phase_one

source = Path("research/raw/P12Xuelin/extracted/Phase12Xuelin/DFKernel/DF3Plus3.h").read_text()


def method(name):
    match = re.search(r"(?:int|bool) " + name + r"\(", source)
    assert match is not None
    pos = match.start()
    start = source.rfind("\n", 0, pos) + 1
    opening = source.index("{", pos)
    end = opening + 1
    depth = 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[start:end]


header = r"""
#include <vector>
#include <algorithm>
using namespace std;
static int coins[2], coinindex;
struct BinomialDistribution { BinomialDistribution(double,double){}
 double GetValue(){return coins[coinindex++];} };
struct Vec:vector<int>{Vec():vector<int>(6){} void Initialize(int x){fill(begin(),end(),x);} };
struct Patient {int dose,tox;int GetDose(){return dose;}bool IsEvent(int){return tox;}
 double GetOutcomeTime(int){return 0.;} };
struct Arm {vector<Patient*> m_vPatients; bool closed=false;
 int GetPatientCount(){return m_vPatients.size();}int GetDoseCount(){return 6;}
 void Close(){closed=true;} };
struct DF3Plus3 { Vec m_vTox,m_vPat,m_vAdmissible,m_vDoseOpen,m_vDoseClosed;
 int m_nCurrentDose=0;bool m_bStopTrial=false;
"""
wrapper = r"""
};
extern "C" void run(double* state,int dose,int c0,int c1,double* out){
 DF3Plus3 df;Arm arm;vector<Patient> patients;df.m_nCurrentDose=dose;
 coins[0]=c0;coins[1]=c1;coinindex=0;
 for(int i=0;i<6;i++){
  df.m_vDoseOpen[i]=state[12+i];df.m_vDoseClosed[i]=state[18+i];
  df.m_vAdmissible[i]=state[24+i];
  for(int j=0;j<state[i];j++)patients.push_back({i,j<state[6+i]});
 }
 for(auto& patient:patients)arm.m_vPatients.push_back(&patient);
 vector<Arm*> arms{&arm};int next=df.GetNextDose(arms,0.);
 for(int i=0;i<6;i++){
  out[i]=df.m_vDoseOpen[i];out[6+i]=df.m_vDoseClosed[i];out[12+i]=df.m_vAdmissible[i];
 }
 out[18]=df.m_bStopTrial;out[19]=arm.closed;out[20]=next;
}
"""
rng = np.random.default_rng(8541)
with tempfile.TemporaryDirectory(prefix="phase12-phase1-reference-") as temp:
    path = Path(temp)
    (path / "source.cpp").write_text(
        header
        + "\n".join(method(n) for n in ["GetNextDose", "IsTrialOver", "Randomize", "OpenDoses"])
        + wrapper
    )
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
    lib.run.argtypes = [pointer, ct.c_int, ct.c_int, ct.c_int, pointer]
    states = lowest_closed = 0
    for trial in range(100):
        dose = 0
        op = np.array([1, 0, 0, 0, 0, 0], dtype=bool)
        closed = np.zeros(6, dtype=bool)
        adm = np.zeros(6, dtype=bool)
        n = np.zeros(6, dtype=int)
        tox = np.zeros(6, dtype=int)
        truth = rng.uniform(0, 0.6, 6)
        for step in range(40):
            result = phase12_phase_one(
                n, tox, current_dose=dose, opened=op, closed=closed, admissible=adm
            )
            state = np.r_[n, tox, op, closed, adm].astype(float)
            probabilities = np.zeros(6)
            native_paths = []
            for c0, c1 in [(0, 0), (0, 1), (1, 0), (1, 1)]:
                out = np.zeros(21)
                lib.run(state.ctypes.data_as(pointer), dose, c0, c1, out.ctypes.data_as(pointer))
                np.testing.assert_array_equal(result.opened, out[:6])
                np.testing.assert_array_equal(result.closed, out[6:12])
                np.testing.assert_array_equal(result.admissible, out[12:18])
                assert result.done == bool(out[18])
                assert result.trial_closed == bool(out[19])
                if not result.done:
                    assert out[20] >= 0
                    probabilities[int(out[20])] += 0.25
                native_paths.append(out)
            np.testing.assert_array_equal(result.probability, probabilities)
            states += 1
            if result.done:
                lowest_closed += result.trial_closed
                break
            chosen = native_paths[int(rng.integers(0, 4))]
            op, closed, adm = (
                chosen[:6].astype(bool),
                chosen[6:12].astype(bool),
                chosen[12:18].astype(bool),
            )
            dose = int(chosen[20])
            n[dose] += 1
            tox[dose] += rng.random() < truth[dose]
        else:
            raise AssertionError("source phase I did not finish within its finite enrollment limit")
    report = {
        "trials": 100,
        "states": states,
        "coin_paths_per_state": 4,
        "seed": 8541,
        "lowest_dose_closures": lowest_closed,
        "scope": (
            "Original extracted GetNextDose, IsTrialOver, OpenDoses and Randomize; "
            "complete phase-I histories with supplied Bernoulli outcomes and all coin branches, "
            "assuming outcomes observed before each decision. "
            "Calendar scheduling checked separately."
        ),
    }
    Path("docs/phase12-progression-reference.json").write_text(json.dumps(report, indent=2) + "\n")
    print(report)
