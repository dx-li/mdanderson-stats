using System;using System.IO;using System.Linq;using System.Reflection;using System.Globalization;using System.Xml.Serialization;
// Original managed patient serialization under explicitly selected cultures.
class ManagedPatientProbe {
 static void Main(string[] args){
  var assembly=Assembly.LoadFrom(args[0]);var type=assembly.GetTypes().Single(t=>t.Name=="Patient");
  foreach(var culture in new[]{"en-US","en-GB"}){
   System.Threading.Thread.CurrentThread.CurrentCulture=new CultureInfo(culture);
   var patient=Activator.CreateInstance(type);
   Action<string,object> set=(key,value)=>type.GetProperty(key).SetValue(patient,value,null);
   set("PID",17);set("Fingerprint",123456);set("Cohort",2);set("DoseIndex",1);
   set("DoseDescription","Dose two");set("OutcomeCode",2);
   set("EntryDate",new DateTime(2026,10,2));set("UpdatedDate",new DateTime(2026,10,5));
   var serializer=new XmlSerializer(type);
   string path=Path.Combine(args[1],culture+".xml");
   using(var writer=new StreamWriter(path))serializer.Serialize(writer,patient);
   object restored;using(var reader=new StreamReader(path))restored=serializer.Deserialize(reader);
   if(!type.GetProperty("EntryDate").GetValue(restored,null).Equals(new DateTime(2026,10,2)))throw new Exception("date round trip");
   using(var writer=new StreamWriter(Path.Combine(args[1],culture+"-roundtrip.xml")))serializer.Serialize(writer,restored);
   Console.WriteLine(culture+" roundtrip passed");
  }
 }
}
