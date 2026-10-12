// Hash-pinned original serializer; only synthetic numerical array output retained.
using System;
using System.IO;
using System.Xml;
using System.Xml.Serialization;
using System.Security.Cryptography;
using BOIN;
class Probe {
 static void Main() {
  using(var hash=SHA256.Create()) {
   var digest=BitConverter.ToString(hash.ComputeHash(File.ReadAllBytes(typeof(SimData).Assembly.Location))).Replace("-", "").ToLowerInvariant();
   if(digest!="c8045dd491b33f78d845d87da9bee75cbf68f50df7aca27ac54e1e7caeec24a1") throw new Exception("Unexpected assembly");
  }
  var data=new SimData();
  data.ComboTrialData_NEvaluable=new int[][] {new int[]{3,6,9},new int[]{12,15,18}};
  data.ComboTrialData_NDLT=new int[][] {new int[]{0,1,2},new int[]{3,4,5}};
  // Ignore unrelated properties, whose optional dependencies are not loaded.
  // The two original jagged-array properties and their serializer attributes remain unchanged.
  var overrides=new XmlAttributeOverrides();
  foreach(var property in typeof(SimData).GetProperties()) {
   if(property.Name!="ComboTrialData_NEvaluable" && property.Name!="ComboTrialData_NDLT")
    overrides.Add(typeof(SimData),property.Name,new XmlAttributes {XmlIgnore=true});
  }
  var writer=new StringWriter();new XmlSerializer(typeof(SimData),overrides).Serialize(writer,data);
  var xml=new XmlDocument();xml.LoadXml(writer.ToString());
  Console.WriteLine("<NativeComboArrays>");
  Console.WriteLine(xml.SelectSingleNode("//ComboTrialData_NEvaluable").OuterXml);
  Console.WriteLine(xml.SelectSingleNode("//ComboTrialData_NDLT").OuterXml);
  Console.WriteLine("</NativeComboArrays>");
 }
}
