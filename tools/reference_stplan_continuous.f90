! Original test harness for numeric routines in the separately retrieved
! STPLAN 4.5 archive. No original STPLAN source is included here.
! Each row records a routine, six arguments (unused slots zero), and power.
program reference_stplan_continuous
  implicit none
  real(8), external :: pnor1, pnev2, pnuv2, pexp1, pexp2, pcor1, pcor2
  print '(A)', 'routine,a1,a2,a3,a4,a5,a6,power'
  call emit('pnor1', [0d0,1d0,12d0,.025d0,0d0,0d0], pnor1(0d0,1d0,12d0,.025d0))
  call emit('pnor1', [.5d0,1d0,20.5d0,.05d0,0d0,0d0], pnor1(.5d0,1d0,20.5d0,.05d0))
  call emit('pnor1', [1.2d0,2.5d0,100d0,.01d0,0d0,0d0], pnor1(1.2d0,2.5d0,100d0,.01d0))
  call emit('pnev2', [.5d0,.025d0,20d0,30d0,1d0,0d0], pnev2(.5d0,.025d0,20d0,30d0,1d0))
  call emit('pnev2', [.7d0,.05d0,12.5d0,14.5d0,2d0,0d0], pnev2(.7d0,.05d0,12.5d0,14.5d0,2d0))
  call emit('pnev2', [0d0,.05d0,20d0,30d0,1d0,0d0], pnev2(0d0,.05d0,20d0,30d0,1d0))
  call emit('pnuv2', [.5d0,.025d0,20d0,40d0,1d0,2d0], pnuv2(.5d0,.025d0,20d0,40d0,1d0,2d0))
  call emit('pnuv2', [2d0,.05d0,5.5d0,10d0,2d0,1d0], pnuv2(2d0,.05d0,5.5d0,10d0,2d0,1d0))
  call emit('pnuv2', [0d0,.01d0,40d0,10d0,3d0,1d0], pnuv2(0d0,.01d0,40d0,10d0,3d0,1d0))
  call lognormal(100d0,80d0,.35d0,15d0,22d0,.025d0)
  call lognormal(1d0,1.5d0,.8d0,25.5d0,50d0,.05d0)
  call emit('pexp1', [.5d0,20d0,.05d0,0d0,0d0,0d0], pexp1(.5d0,20d0,.05d0))
  call emit('pexp1', [2d0,12.5d0,.025d0,0d0,0d0,0d0], pexp1(2d0,12.5d0,.025d0))
  call emit('pexp1', [1d0,8d0,.05d0,0d0,0d0,0d0], pexp1(1d0,8d0,.05d0))
  call emit('pexp2', [2d0,20d0,40d0,.05d0,0d0,0d0], pexp2(2d0,20d0,40d0,.05d0))
  call emit('pexp2', [1.5d0,41d0,61d0,.025d0,0d0,0d0], pexp2(1.5d0,41d0,61d0,.025d0))
  call emit('pexp2', [1d0,20d0,40d0,.05d0,0d0,0d0], pexp2(1d0,20d0,40d0,.05d0))
  call emit('pcor1', [0d0,.5d0,20d0,.025d0,0d0,0d0], pcor1(0d0,.5d0,20d0,.025d0))
  call emit('pcor1', [-.4d0,.2d0,40.5d0,.05d0,0d0,0d0], pcor1(-.4d0,.2d0,40.5d0,.05d0))
  call emit('pcor1', [.7d0,.85d0,30d0,.01d0,0d0,0d0], pcor1(.7d0,.85d0,30d0,.01d0))
  call emit('pcor2', [0d0,.5d0,20d0,30d0,.025d0,0d0], pcor2(0d0,.5d0,20d0,30d0,.025d0))
  call emit('pcor2', [-.4d0,.2d0,40.5d0,50d0,.05d0,0d0], pcor2(-.4d0,.2d0,40.5d0,50d0,.05d0))
  call emit('pcor2', [.5d0,.5d0,20d0,80d0,.05d0,0d0], pcor2(.5d0,.5d0,20d0,80d0,.05d0))
contains
  subroutine emit(name,arguments,power)
    character(*), intent(in) :: name
    real(8), intent(in) :: arguments(6),power
    write(*,'(A,7(",",ES25.17E3))') name,arguments,power
  end subroutine emit
  subroutine lognormal(mean1,mean2,cv,n1,n2,sig)
    real(8),intent(in) :: mean1,mean2,cv,n1,n2,sig
    real(8) :: delta,sd,power
    delta=abs(log(mean1)-log(mean2))
    sd=sqrt(log(1d0+cv**2))
    power=pnev2(delta,sig,n1,n2,sd)
    call emit('lognormal',[mean1,mean2,cv,n1,n2,sig],power)
  end subroutine lognormal
end program reference_stplan_continuous
