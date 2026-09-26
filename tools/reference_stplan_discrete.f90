! Independent probe; original numerical routines must be acquired separately.
program reference_stplan_discrete
  use asin_trans_mod, only: arctr
  use bin_hist_cont_pow_mod, only: pow_rcn, pow_ren, pow_tcp, pow_tep, &
    pow_z_sig, pow_e_prob_larger, power_bin_hist_control
  implicit none
  real(8), external :: pbin1, ppoi1, pfish, pplsc, dinvnr
  logical, external :: qbin2, qbn2m, qres2, qbink
  call stcuio(5,6)
  print '(A)', 'routine,a1,a2,a3,a4,a5,a6,a7,a8,power'
  call run('arcsine',[.2d0,.4d0,40d0,60d0,.05d0,1d0,0d0,0d0])
  call run('arcsine',[.7d0,.5d0,25.5d0,35d0,.01d0,2d0,0d0,0d0])
  call run('arcsine',[.3d0,.3d0,20d0,30d0,.05d0,2d0,0d0,0d0])
  call run('median',[.5d0,.1d0,120d0,.05d0,1d0,0d0,0d0,0d0])
  call run('median',[.3d0,.15d0,75.5d0,.05d0,2d0,0d0,0d0,0d0])
  call run('historical',[.2d0,.4d0,80d0,40d0,.05d0,1d0,0d0,0d0])
  call run('historical',[.7d0,.5d0,100d0,50.5d0,.025d0,1d0,0d0,0d0])
  call run('historical',[.3d0,.3d0,80d0,20d0,.05d0,1d0,0d0,0d0])
  call run('responders',[.75d0,.8d0,150d0,100d0,.15d0,.95d0,0d0,0d0])
  call run('responders',[.5d0,.625d0,120.5d0,90d0,.25d0,.99d0,0d0,0d0])
  call run('fisher',[.2d0,.4d0,50d0,.05d0,0d0,0d0,0d0,0d0])
  call run('fisher',[.2d0,.21d0,2d0,.025d0,0d0,0d0,0d0,0d0])
  call run('fisher',[.3d0,.3d0,20d0,.05d0,0d0,0d0,0d0,0d0])
  call run('ksample3',[.2d0,.4d0,.6d0,40d0,50d0,60d0,.05d0,2d0])
  call run('ksample3',[.4d0,.4d0,.4d0,20d0,30d0,40d0,.01d0,2d0])
  call run('ksample2',[.2d0,.4d0,0d0,40d0,60d0,0d0,.05d0,1d0])
  call run('ksample2',[.2d0,.4d0,0d0,40d0,60d0,0d0,.05d0,2d0])
  call run('retention',[.05d0,100d0,80d0,3d0,0d0,0d0,0d0,0d0])
  call run('retention',[.2d0,50d0,30d0,2.5d0,0d0,0d0,0d0,0d0])
  call run('binomial',[.2d0,.4d0,40d0,.05d0,0d0,0d0,0d0,0d0])
  call run('binomial',[.6d0,.4d0,50d0,.025d0,0d0,0d0,0d0,0d0])
  call run('binomial',[.2d0,.4d0,1d0,.05d0,0d0,0d0,0d0,0d0])
  call run('poisson',[1d0,2d0,10d0,.05d0,0d0,0d0,0d0,0d0])
  call run('poisson',[2d0,1d0,12.5d0,.025d0,0d0,0d0,0d0,0d0])
  call run('poisson',[.1d0,.05d0,1d0,.05d0,0d0,0d0,0d0,0d0])
contains
  subroutine run(name,a)
    character(*),intent(in) :: name
    real(8),intent(in) :: a(8)
    real(8) :: w(8), power, probabilities(10), sizes(10), probk(2)
    integer :: rc, k
    logical :: ok
    w=a
    power=0d0
    ok=.true.
    rc=0
    select case(name)
    case('arcsine')
      ok=qbin2(w(1),w(2),w(3),w(4),w(5),power,int(w(6)),6,rc)
    case('median')
      ok=qbn2m(w(2),w(3),w(4),power,w(1),int(w(5)),4,rc)
    case('historical')
      pow_rcn=1d0/w(3)
      pow_ren=1d0/w(4)
      pow_tcp=arctr(w(1))
      pow_tep=arctr(w(2))
      pow_z_sig=-dinvnr(w(5)/w(6),1d0-w(5)/w(6))
      pow_e_prob_larger=w(2)>w(1)
      power=power_bin_hist_control()
    case('responders')
      ok=qres2(w(1),w(2),w(3),w(4),w(5),w(6),power,7,rc)
    case('fisher')
      power=pfish(w(1),w(2),w(3),w(4))
    case('ksample2','ksample3')
      k=3
      if (name=='ksample2') k=2
      probabilities=0d0
      sizes=0d0
      probabilities(1:k)=w(1:k)
      sizes(1:k)=w(4:3+k)
      ok=qbink(k,probabilities,sizes,w(7),power,probk,int(w(8)),4,rc)
    case('retention')
      power=pplsc(w(1),w(2),w(3),w(4))
    case('binomial')
      power=pbin1(w(1),w(2),w(3),w(4))
    case('poisson')
      power=ppoi1(w(1),w(2),w(3),w(4))
    case default
      error stop 'Unknown probe routine'
    end select
    if (.not.ok .or. rc/=0) error stop 'Native routine rejected reference input'
    write(*,'(A,9(",",ES25.17E3))') name,a,power
  end subroutine run
end program reference_stplan_discrete
