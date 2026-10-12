"""Original ToxFinder stored-variance/display-SD conversion with isolated storage.

Overrides only parameter storage and GUI event/validation plumbing; original
setGADAValue, getGADAValue and getDisplayValue perform all conversions.
"""

import hashlib
import json
import pathlib
import subprocess
import sys
import tempfile

jar = pathlib.Path(sys.argv[1]).resolve()
assert hashlib.sha256(jar.read_bytes()).hexdigest() == (
    "39e72d4b535595f548b6602496227bc92a6d9107fbb3ce5560973342fd39897a"
)
source = r"""
import com.tessella.wingada.parameter.VarianceAsSDValue;
public class ToxVarianceProbe {
 public static void main(String[] args) throws Exception {
  for(String value: new String[]{"0.25","0.81","3.0"}) {
   VarianceAsSDValue v=new VarianceAsSDValue(){
    private Object value;
    public void setValue(Object x){value=x;}
    public Object getValue(){return value;}
   };
   v.setGADAValue(value);
   System.out.println(value+" "+v.getDisplayValue()+" "+v.getGADAValue());
  }
 }
}
"""
with tempfile.TemporaryDirectory(prefix="tox-variance-") as directory:
    path = pathlib.Path(directory) / "ToxVarianceProbe.java"
    path.write_text(source)
    subprocess.run(
        [
            "java",
            "--module",
            "jdk.compiler/com.sun.tools.javac.Main",
            "-classpath",
            str(jar),
            str(path),
        ],
        check=True,
        capture_output=True,
    )
    result = subprocess.run(
        ["java", "-classpath", directory + ":" + str(jar), "ToxVarianceProbe"],
        check=True,
        capture_output=True,
        text=True,
    )
rows = [
    dict(zip(("stored", "displayed_sd", "saved"), map(float, line.split()), strict=True))
    for line in result.stdout.splitlines()
]
print(json.dumps(rows, indent=2))
