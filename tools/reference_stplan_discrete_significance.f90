! Native IWHICH=4 inverse significance; source routines acquired separately.
program reference_stplan_discrete_significance
  implicit none
  logical, external :: qbin1, qpoi1
  real(8), external :: crbin1, crpoi1
  call stcuio(5,6)
  print '(A)', 'method,null,alternative,size_or_time,target,alpha,power,critical,status,ok'
  call run('binomial',.2d0,.4d0,40d0,.8d0)
  call run('binomial',.6d0,.4d0,50d0,.8d0)
  call run('binomial',.2d0,.4d0,2d0,.8d0)
  call run('binomial',.2d0,.4d0,20d0,.95d0)
  call run('poisson',1d0,2d0,10d0,.8d0)
  call run('poisson',2d0,1d0,12.5d0,.8d0)
  call run('poisson',.1d0,.05d0,1d0,.8d0)
  call run('poisson',.1d0,.05d0,1d0,.99d0)
contains
  subroutine run(method,null,alternative,size_or_time,target)
    character(*), intent(in) :: method
    real(8), intent(in) :: null,alternative,size_or_time,target
    real(8) :: p0,p1,n,alpha,power,critical
    integer :: status
    logical :: ok
    p0=null; p1=alternative; n=size_or_time; power=target; alpha=.05d0
    if (method=='binomial') then
      ok=qbin1(p0,p1,n,alpha,power,p0<=p1,4,status)
      critical=crbin1(p0,p1,n,alpha)
    else
      ok=qpoi1(p0,p1,n,alpha,power,p0<=p1,4,status)
      critical=crpoi1(p0,p1,n,alpha)
    end if
    write(*,'(A,8(",",ES25.17E3),",",L1)') &
      method,null,alternative,size_or_time,target,alpha,power,critical,dble(status),ok
  end subroutine run
end program reference_stplan_discrete_significance
