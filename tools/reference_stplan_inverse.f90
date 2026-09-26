! Independent probe of separately acquired STPLAN inverse planning routines.
program reference_stplan_inverse
  implicit none
  logical,external :: qnor1,qnev2,qcor1,qexp2
  real(8),external :: pnor1,pnev2,pcor1,pexp2
  real(8) :: m0,m1,sd,n,n2,alpha,target,power
  logical :: ok
  integer :: status,which
  print '(A)', 'case_id,value,target,achieved,status,ok'
  do which=2,5
    m0=0d0;m1=.5d0;sd=1d0;n=10d0;alpha=.05d0;target=.8d0
    if (which==5) n=20d0
    ok=qnor1(m0,m1,sd,n,alpha,target,which,1,status)
    power=pnor1(m1-m0,sd,n,alpha)
    select case(which)
    case(2)
      call emit('normal_difference_native_approx',m1-m0,target,power,status,ok)
    case(3)
      call emit('normal_sd_native_approx',sd,target,power,status,ok)
    case(4)
      call emit('normal_n',n,target,power,status,ok)
    case(5)
      call emit('normal_alpha',alpha,target,power,status,ok)
    end select
  end do
  m0=0d0;m1=.5d0;sd=1d0;n=2d0;n2=2d0;alpha=.05d0;target=.8d0
  ok=qnev2(m0,m1,n,n2,sd,alpha,target,1,8,status)
  power=pnev2(m1-m0,alpha,n,n2,sd)
  call emit('normal_equal_n',n,target,power,status,ok)
  m0=0d0;m1=.4d0;n=4d0;alpha=.05d0;target=.8d0
  ok=qcor1(m0,m1,n,alpha,target,1,3,.true.,status)
  power=pcor1(m0,m1,n,alpha)
  call emit('correlation_n',n,target,power,status,ok)
  m0=10d0;m1=15d0;n=2d0;n2=2d0;alpha=.05d0;target=.8d0
  ok=qexp2(m0,m1,n,n2,alpha,target,1,7,status)
  power=pexp2(m1/m0,2*n,2*n2,alpha)
  call emit('exponential_equal_n',n,target,power,status,ok)
contains
  subroutine emit(name,value,target,achieved,status,ok)
    character(*),intent(in) :: name
    real(8),intent(in) :: value,target,achieved
    integer,intent(in) :: status
    logical,intent(in) :: ok
    write(*,'(A,3(",",ES25.17E3),",",I0,",",L1)') name,value,target,achieved,status,ok
  end subroutine
end program
