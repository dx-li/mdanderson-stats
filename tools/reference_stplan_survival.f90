! Independent source probe for separately acquired STPLAN survival routines.
program reference_stplan_survival
  use exp_death_mod, only: pdeath
  use p2exp_mod, only: p2exp
  implicit none
  real(8),external :: p1exp,prct2,pint2,pschc
  real(8) :: tb,l1,l2
  integer :: imdtyp
  common /c2expp/tb,l1,l2,imdtyp
  print '(A)', 'routine,a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,power'
  call run('one',[10d0,15d0,5d0,12d0,6d0,.05d0,0d0,0d0,0d0,0d0])
  call run('one',[15d0,10d0,3d0,10d0,0d0,.025d0,0d0,0d0,0d0,0d0])
  call run('one',[10d0,10d0,1d0,2d0,0d0,.05d0,0d0,0d0,0d0,0d0])
  call run('george_desu',[10d0,15d0,5d0,12d0,6d0,.05d0,0d0,0d0,0d0,0d0])
  call run('george_desu',[15d0,10d0,3d0,10d0,0d0,.025d0,0d0,0d0,0d0,0d0])
  call run('george_desu',[10d0,10d0,5d0,12d0,6d0,.05d0,0d0,0d0,0d0,0d0])
  call run('information',[10d0,15d0,5d0,12d0,6d0,.05d0,0d0,0d0,0d0,0d0])
  call run('information',[15d0,10d0,3d0,10d0,0d0,.025d0,0d0,0d0,0d0,0d0])
  call run('information',[10d0,10d0,5d0,12d0,6d0,.05d0,0d0,0d0,0d0,0d0])
  call run('historical',[.05d0,5d0,12d0,6d0,.05d0,.1d0,40d0,20d0,0d0,0d0])
  call run('historical',[.05d0,5d0,12d0,6d0,.05d0,.1d0,40d0,20d0,.2d0,1d0])
  call run('historical',[.15d0,5d0,12d0,6d0,.025d0,.1d0,40d0,20d0,.2d0,1d0])
  call run('piecewise_native',[1.5d0,5d0,10d0,5d0,.05d0,3d0,.2d0,.05d0,1d0,0d0])
  call run('piecewise_native',[1.5d0,5d0,1d0,.5d0,.05d0,.3d0,.2d0,.05d0,1d0,0d0])
  call run('piecewise_native',[1.5d0,5d0,10d0,5d0,.05d0,30d0,.2d0,.05d0,2d0,0d0])
contains
  subroutine run(name,a)
    character(*),intent(in) :: name
    real(8),intent(in) :: a(10)
    real(8) :: w(10),power,pd
    w=a
    select case(name)
    case('one')
      pd=pdeath(ta=w(4),tf=w(5),mean=w(2))
      power=p1exp(w(1),w(2),w(3),w(4),pd,w(6))
    case('george_desu')
      power=prct2(w(1),w(2),w(3),w(4),w(5),w(6))
    case('information')
      power=pint2(w(1),w(2),w(3),w(4),w(5),w(6))
    case('historical')
      power=pschc(w(1),w(2),w(3),w(4),w(5),w(6),w(7),w(8),w(9),w(10)==1d0)
    case('piecewise_native')
      tb=w(6)
      l1=w(7)
      l2=w(8)
      imdtyp=int(w(9))
      power=p2exp(w(1),w(2),w(3),w(4),w(5))
    case default
      error stop 'Unknown routine'
    end select
    write(*,'(A,11(",",ES25.17E3))') name,a,power
  end subroutine
end program
