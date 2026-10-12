// External original data assembly only; independently supplied synthetic patient.
// Compile against the hash-pinned BOIN_Data.dll. Optional argument: patient XML
// to deserialize/re-serialize through the original type and XML serializer.
using System;
using System.IO;
using System.Xml.Serialization;
using System.Globalization;
using System.Security.Cryptography;
using MDACC.Biostat.Utilities;
class Probe {
 static void Main(string[] args) {
  using(var hash=SHA256.Create()) {
   var digest=BitConverter.ToString(hash.ComputeHash(File.ReadAllBytes(typeof(Patient).Assembly.Location))).Replace("-", "").ToLowerInvariant();
   if(digest!="c8045dd491b33f78d845d87da9bee75cbf68f50df7aca27ac54e1e7caeec24a1") throw new Exception("Unexpected original assembly");
  }
  CultureInfo.CurrentCulture=CultureInfo.InvariantCulture;
  var serializer=new XmlSerializer(typeof(Patient));
  Patient p;
  if(args.Length==1) {
   using(var stream=File.OpenRead(args[0])) p=(Patient)serializer.Deserialize(stream);
  } else {
   p=new Patient();p.PID=41;p.Fingerprint=101;p.Cohort=2;
   p.DoseIndexA=1;p.DoseIndexB=2;p.OutcomeCode=ThreeStateLogical.tslUNKNOWN;
   p.EntryDate=new DateTime(2026,10,12);p.UpdatedDate=new DateTime(2026,10,13);
  }
  serializer.Serialize(Console.Out,p);
 }
}
