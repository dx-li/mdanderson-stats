I. Introduction:

The flecs90 processor is designed to translate code written in the
FLECS extended control language for Fortran 77 into fixed form Fortran
90 code.  Because Fortran 90 contains all of the extended capabilities
of FLECS, flecs90 is designed to provide a one time conversion from
FLECS code to Fortran 90 code.

II. Packing List:

The following files should be contained in the distribution of the
flecs90 program.

(1) README - This file.
    
(2) getopt.c, flecs90.c, util.c, work.c and flecs90.h - The C source 
    code for the flecs90 program.

(3) flecs90.1 - A UNIX syle man page for the flecs90 program.

(4) flecs90.ps - A postscript document which describes the
    flecs90 program.

III. Installation:

The files getopt.c, flecs90.c, util.c, work.c and flecs90.h contain
the C source code for the flecs90 program. They should be compiled
using an ANSI standard C compiler such as gcc. For example, the
following command works on our UNIX system:

   gcc -o flecs90 getopt.c flecs90.c util.c work.c

IV. Contacting the Authors:

If you need help with the installation of flecs90 or have questions
about the C code please send electronic mail to:

Internet: kruss@odin.mda.uth.tmc.edu

VI. Legalities:

This code is placed in the public domain. Enjoy.

                             NO WARRANTY

    WE PROVIDE ABSOLUTELY NO WARRANTY OF ANY KIND EITHER EXPRESSED OR
IMPLIED, INCLUDING BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. THE ENTIRE RISK
AS TO THE QUALITY AND PERFORMANCE OF THE PROGRAM IS WITH YOU. SHOULD
THIS PROGRAM PROVE DEFECTIVE, YOU ASSUME THE COST OF ALL NECESSARY
SERVICING, REPAIR OR CORRECTION.

    IN NO EVENT SHALL THE UNIVERSITY OF TEXAS OR ANY OF ITS COMPONENT
INSTITUTIONS INCLUDING M. D. ANDERSON HOSPITAL BE LIABLE TO YOU FOR
DAMAGES, INCLUDING ANY LOST PROFITS, LOST MONIES, OR OTHER SPECIAL,
INCIDENTAL OR CONSEQUENTIAL DAMAGES ARISING OUT OF THE USE OR
INABILITY TO USE (INCLUDING BUT NOT LIMITED TO LOSS OF DATA OR DATA OR
ITS ANALYSIS BEING RENDERED INACCURATE OR LOSSES SUSTAINED BY THIRD
PARTIES) THE PROGRAM.

(Above NO WARRANTY modified from the GNU NO WARRANTY statement.)
