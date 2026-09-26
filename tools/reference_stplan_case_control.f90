! Independent source probe; no original STPLAN implementation is bundled.
program reference_stplan_case_control
  use pcaco_mod, only: pr_risk, pow_caco
  implicit none
  real(8) :: frequency, disease_exposed, disease_unexposed, pairs, target, level
  integer :: which
  common /mcaco/frequency,disease_exposed,disease_unexposed,pairs,level,target,which
  real(8),external :: fmcaco
  interface
    function ppoi2(rate1,rate2,time1,time2,sig,opt)
      real(8) :: ppoi2
      real(8),intent(in) :: rate1,rate2,time1,sig
      real(8),intent(inout) :: time2
      integer,optional :: opt
    end function
  end interface
  print '(A)', 'routine,a1,a2,a3,a4,a5,a6,power'
  call unmatched([.3d0,.1d0,.05d0,60d0,60d0,.05d0])
  call unmatched([.3d0,.05d0,.1d0,60d0,60d0,.05d0])
  call unmatched([.4d0,.4d0,.2d0,100.5d0,75d0,.025d0])
  call unmatched([.2d0,.1d0,.1d0,80d0,120d0,.01d0])
  call matched([.3d0,.1d0,.05d0,60d0,.05d0,0d0])
  call matched([.4d0,.4d0,.2d0,80d0,.025d0,0d0])
  call matched([.3d0,.05d0,.1d0,60d0,.05d0,0d0])
  call two_poisson([1d0,2d0,10d0,10d0,.05d0,0d0])
  call two_poisson([2d0,1d0,12.5d0,17d0,.025d0,0d0])
  call two_poisson([1d0,1.5d0,100d0,100d0,.05d0,0d0])
contains
  subroutine emit(name,a,power)
    character(*),intent(in) :: name
    real(8),intent(in) :: a(6),power
    write(*,'(A,7(",",ES25.17E3))') name,a,power
  end subroutine
  subroutine unmatched(a)
    real(8),intent(in) :: a(6)
    real(8) :: pc,pu
    call pr_risk(a(1),a(2),a(3),pc,pu)
    call emit('unmatched',a,pow_caco(pc,a(4),pu,a(5),a(6)))
  end subroutine
  subroutine matched(a)
    real(8),intent(in) :: a(6)
    frequency=a(1)
    disease_exposed=a(2)
    disease_unexposed=a(3)
    pairs=a(4)
    level=a(5)
    target=0d0
    which=6
    call emit('matched_native',a,fmcaco(0d0))
  end subroutine
  subroutine two_poisson(a)
    real(8),intent(in) :: a(6)
    real(8) :: time2,power
    time2=a(4)
    power=ppoi2(a(1),a(2),a(3),time2,a(5),0)
    call emit('poisson_native',a,power)
  end subroutine
end program
