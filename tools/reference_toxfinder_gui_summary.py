"""Execute original ToxFinder Java summary/variance methods without GUI startup.

No original bytecode is distributed. Requires a checksum-matching original JAR
and Java with jdk.compiler. The source invokes original public methods directly.
"""

import argparse
import hashlib
import json
import os
import pathlib
import subprocess
import tempfile

EXPECTED = "39e72d4b535595f548b6602496227bc92a6d9107fbb3ce5560973342fd39897a"
SOURCE = r"""
import com.tessella.wingada.tablevisualisation.RawColumnValue;
import com.tessella.wingada.parameter.toxParam.ToxicitySurfaceProvider;
public class ToxSummaryProbe {
 public static void main(String[] args) {
  for (double[] xs: new double[][]{{1,2,3},{1,3},{2,2,2},{2}}) {
   RawColumnValue r=new RawColumnValue();for(double x:xs)r.add(x);
   System.out.printf(java.util.Locale.ROOT,"%d %.17g %.17g %.17g %.17g%n",
       r.getCount(),r.getMean(),r.getMeanSD(),
       ToxicitySurfaceProvider.calculateVariance(xs),
       ToxicitySurfaceProvider.calculateVariance(xs,r.getMean()));
  }
 }
}
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("jar", type=pathlib.Path)
    a = parser.parse_args()
    jar = a.jar.resolve()
    if hashlib.sha256(jar.read_bytes()).hexdigest() != EXPECTED:
        raise ValueError("original ToxFinder JAR checksum mismatch")
    with tempfile.TemporaryDirectory(prefix="toxfinder-java-summary-") as d:
        src = pathlib.Path(d) / "ToxSummaryProbe.java"
        src.write_text(SOURCE)
        subprocess.run(
            [
                "java",
                "--module",
                "jdk.compiler/com.sun.tools.javac.Main",
                "-classpath",
                str(jar),
                str(src),
            ],
            check=True,
            capture_output=True,
            timeout=30,
            text=True,
        )
        r = subprocess.run(
            ["java", "-classpath", d + os.pathsep + str(jar), "ToxSummaryProbe"],
            check=True,
            capture_output=True,
            timeout=30,
            text=True,
        )
    rows = []
    for line in r.stdout.splitlines():
        n, mean, sem, var, pop = line.split()
        rows.append(
            {
                "count": int(n),
                "mean": float(mean),
                "native_mean_sd": float(sem),
                "posterior_variance": None if var == "NaN" else float(var),
                "known_mean_variance": float(pop),
            }
        )
    print(json.dumps({"jar_sha256": EXPECTED, "rows": rows}, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
