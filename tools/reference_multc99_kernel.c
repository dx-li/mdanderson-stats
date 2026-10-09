#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "header.h"
extern int MIX_K;
int main(void) {
 double priors[][5]={{2,3,1,1,0},{7,13,2,3,.05},{5,15,1.5,3.5,.1},{5,10,1.5,2,-.1},{.5,.5,1,1,.2}};
 printf("{\"probabilities\":[");
 int sep=0;
 for(int j=0;j<5;j++) for(int n=0;n<=12;n+=3) for(int x=0;x<=n;x+=n?n/3:1) {
  double *p=priors[j];double result=Lambda4(x,n,p[0],p[1],p[2],p[3],p[4]);
  printf("%s{\"x\":%d,\"n\":%d,\"aS\":%.17g,\"bS\":%.17g,\"aE\":%.17g,\"bE\":%.17g,\"margin\":%.17g,\"probability\":%.17g}",sep++?",":"",x,n,p[0],p[1],p[2],p[3],p[4],result);
 }
 printf("],\"boundaries\":[");sep=0;
 for(int mix=0;mix<=1;mix++) for(int conditional=0;conditional<=1;conditional++) for(int nmin=1;nmin<=6;nmin+=5) {
  Design d={0};Event e={0},parent={0};int lo[21],hi[21];d.Nmin=nmin;d.Nmax=20;d.Sprior_type=mix?2:1;d.Sprior_weight[0]=.3;d.Sprior_weight[1]=.7;MIX_K=d.Sprior_type;
  e.design=&d;e.Nmax=20;e.Nmin=nmin;e.parent=conditional?&parent:NULL;e.aS=7;e.bS=13;e.aE=2;e.bE=3;e.deltaL=.05;e.deltaU=.1;e.pL=.1;e.pU=.9;e.boundary_type=2;e.lowerbound=lo;e.upperbound=hi;
  e.mix_aS[0]=7;e.mix_bS[0]=13;e.mix_aS[1]=6;e.mix_bS[1]=4;
  compute_event_upperbound(&e);compute_event_lowerbound(&e);
  printf("%s{\"mixture\":%d,\"conditional\":%d,\"nmin\":%d,\"lower\":[",sep++?",":"",mix,conditional,nmin);
  for(int n=0;n<=20;n++)printf("%s%d",n?",":"",lo[n]);printf("],\"upper\":[");
  for(int n=0;n<=20;n++)printf("%s%d",n?",":"",hi[n]);printf("]}");
 }
 printf("],\"calibration\":[");
 for(int j=0;j<3;j++) {
  double mus[][4]={{.1,.2,.3,.4},{.2,.1,.4,.3},{.3,.3,.2,.2}};
  double widths[]={.2,.25,.15};int definition[4]={1,0,1,0};double alpha[4];
  Design design={0};Event event={0};design.number_of_element=4;design.muE=mus[j];design.aE=alpha;design.current_event=&event;event.definition=definition;
  getdir(&design,widths[j],.9,2);
  printf("%s{\"means\":[",j?",":"");
  for(int k=0;k<4;k++)printf("%s%.17g",k?",":"",mus[j][k]);
  printf("],\"width\":%.17g,\"coverage\":0.9,\"alpha\":[",widths[j]);
  for(int k=0;k<4;k++)printf("%s%.17g",k?",":"",alpha[k]);printf("]}");
 }
 printf("],\"precision\":[");sep=0;
 for(int j=0;j<3;j++)for(int k=0;k<3;k++){
  double means[]={.2,.4,.7},widths[]={.15,.25,.4};Event event={0};event.aS=7;event.bS=13;event.aE=2;event.bE=3;
  int n=compute_Nmax(&event,means[j],widths[k],.9);
  printf("%s{\"mean\":%.17g,\"width\":%.17g,\"coverage\":0.9,\"sample_size\":%d}",sep++?",":"",means[j],widths[k],n);
 }
 printf("]}\n");return 0;
}
