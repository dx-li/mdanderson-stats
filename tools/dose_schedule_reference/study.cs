// Reference-only: native CRT module initializer disabled in a temporary DLL copy.
// All managed DTO constructors/properties and the original serializer are unchanged.
using System;using System.IO;using MDSFKernel;using MDACC.Biostat.Utilities;
class P {static void Main(){var app=new DSFSimApp();var d=app.DSFSimulatorData;var p=d.m_pDSFParameters;
p.m_numDoses=3;p.m_numSchedules=2;p.m_maxDaysFollowUp=60;p.m_numAdminsPerCycle=3;p.m_numDaysBetweenCycleStarts=28;p.m_maxPatients=12;p.m_numRepetitions=20;p.m_dMeanTimeBetweenArrivals=7;p.m_dTargetCumuProbToxByTau=.3;p.m_dMaxCumuProbToxByTau=.4;p.m_dUpperAcceptableProbCutoff=.8;p.m_dFractionLowGradeTox=.25;p.m_dTimeToResolveLowGrade=10;
var scenario=new DSFScenario(3,2);scenario.m_id=1;for(int a=0;a<3;a++)for(int b=0;b<2;b++)scenario.m_vTrueCumuProbToxByTau[a*5+b]=.1+.1*a+.05*b;d.m_pDSFScenarios.Add(scenario);
var patient=new DSFTreatment(3);patient.m_iDayOfEntry=0;patient.m_iDoseAssigned=0;patient.m_iScheduleAssigned=0;patient.m_iOutcome=0;patient.m_iDaysToOutcome=60;patient.m_numAdmins=3;patient.m_vDaysGiven=new int[]{0,1,2};patient.m_vDosesGiven=new int[]{0,1,0};d.m_pDSFTreatments.Add(patient);
var date=new DateTime(2000,1,1);app.UILastChangedTime=date;app.SimulationReportLastChangedTime=date;app.TreatmentDataLastChangedTime=date;app.TreatmentDataLastSavedTime=date;app.TreatmentReportLastChangedTime=date;d.m_pDSFTreatments.m_dtLastChangedTime=date;d.m_pDSFTreatments.m_dtStartTrial=date;d.m_pDSFScenarios.m_dtLastChangedTime=date;d.m_pDSFTreatmentInfo.m_dtLastChangedTime=date;
string xml=null;BSSerializer.Serialize(app,ref xml);Console.Write(xml);}}
