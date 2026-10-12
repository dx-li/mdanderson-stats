using System;using System.IO;using System.Xml.Serialization;using System.Text;
using ARand.Simulation;using ARand.Design;
// Synthetic inputs passed to unmodified original managed assemblies.
class ManagedStudyProbe {
 static void Save(Simulation s,string path){using(var w=new StreamWriter(path))new XmlSerializer(typeof(Simulation)).Serialize(w,s);}
 static void Main(string[] args){
  foreach(bool tte in new[]{false,true}){
   var s=new Simulation();s.Name=tte?"Synthetic TTE reference":"Synthetic binary reference";s.Seed=12345;
   var t=s.TrialDetails;t.CurrentTime=8;t.BinStatModel=tte?BinaryStatModel.EnumBinaryStatModel.TTE_NO_COVARIATES:BinaryStatModel.EnumBinaryStatModel.BINARY_NO_COVARIATES;
   t.MaxPatientsInTrial=6;t.MinPatientsInTrial=2;t.MaxTrialLength=20;
   t.StoppingRule.IsMaxPatientsInTrial=true;t.StoppingRule.IsMinPatientsInTrial=true;t.StoppingRule.IsMaxTrialLength=true;
   t.StoppingRule.AdditionalFu=2;t.StoppingRule.FinalMaxProbBest=.5;
   t.Randomizer.QtyPatsEnrollB4Adapt=4;t.Randomizer.TunningParameter=.5;
   foreach(var p in t.Calculator.CalcOutcome.CalcParametersColl){p.Param1=2;p.Param2=tte?3:2;}
   foreach(var a in t.StoppingRule.StoppingRuleArms){a.pl=.05;a.pu=.95;a.pufinal=.5;}
   t.Calculator.Calc_TTE_Meth=tte?"Median":"NA";
   s.Scenarios[0].QtyReps=100;
   if(tte){s.Scenarios[0].SimDetails.SimOutcome.SimType="TTE";s.Scenarios[0].SimDetails.SimOutcome.ParamSets=new[]{ParamSet.CreateDefaultTTE("Arm1",(Param.Names)3),ParamSet.CreateDefaultTTE("Arm2",(Param.Names)3)};}
   var a1=new Patient();a1.PatientID="synthetic-observed";a1.Treatment="Arm1";a1.Group="No Groups";a1.StartTime=1;a1.Outcome.Name="Outcome";a1.Outcome.OutcomeValue=1;a1.Outcome.time=3;
   var a2=new Patient();a2.PatientID="synthetic-pending";a2.Treatment="Arm2";a2.Group="No Groups";a2.StartTime=7;a2.Outcome.Name="Outcome";a2.Outcome.OutcomeValue=-1;
   t.CurrentPatients=new[]{a1,a2};
   var checks=new StringBuilder();Console.WriteLine((tte?"tte":"binary")+" valid="+s.IsValid(checks)+" "+checks);
   Save(s,Path.Combine(args[0],tte?"tte.xml":"binary.xml"));
   var copy=Simulation.Deserialize(Path.Combine(args[0],tte?"tte.xml":"binary.xml"));
   Save(copy,Path.Combine(args[0],tte?"tte-roundtrip.xml":"binary-roundtrip.xml"));
  }
 }
}
