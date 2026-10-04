// Reference harness for the unchanged cached MDACC RunningVariance implementation.
// Compile with RunningVariance.cpp and its directory on the include path.
// Input: row count, then row-major blocks of 60 real values. Output: mean variance.
#include "RunningVariance.h"
#include <array>
#include <iomanip>
#include <iostream>
int main() {
    int n;
    if (!(std::cin >> n) || n < 1 || n > 100000) return 1;
    std::array<MDACC_Biostat::RunningVariance, 60> stats;
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < 60; ++j) {
            double x;
            if (!(std::cin >> x)) return 2;
            stats[j].Push(x);
        }
    }
    std::cout << std::setprecision(17);
    for (const auto& stat : stats)
        std::cout << stat.Mean() << " " << stat.Variance() << "\n";
}
