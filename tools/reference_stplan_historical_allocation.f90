! Reference driver for separately acquired, unmodified STPLAN 4.5 routines.
! Original source is neither included here nor redistributed with the package.
program reference_stplan_historical_allocation
  implicit none
  logical, external :: qschc
  double precision, external :: pschc
  print '(A)', 'case_id,experimental_hazard,control_hazard,accrual_rate,followup_duration,historical_deaths,historical_alive,continued_followup,alpha,target,allocation,accrual_duration,achieved,status,ok'
  call emit('manual_25_4',log(2d0)/24d0,25d0/433d0,5d0,12d0,25d0,25d0,.true.)
  call emit('manual_25_5',log(2d0)/36d0,-log(.8d0)/7.75d0,3d0,0d0,10d0,40d0,.true.)
  call emit('small_history',.05d0,.1d0,10d0,6d0,5d0,10d0,.true.)
  call emit('large_history',.05d0,.1d0,10d0,6d0,1000d0,1000d0,.true.)
  call emit('no_continued_followup',.05d0,.1d0,5d0,6d0,40d0,20d0,.false.)
  call emit('small_effect',.09d0,.1d0,10d0,0d0,10d0,10d0,.true.)
contains
  subroutine emit(name, he0, hc0, rate0, follow0, dead0, alive0, follow_old)
    character(*), intent(in) :: name
    double precision, intent(in) :: he0, hc0, rate0, follow0, dead0, alive0
    logical, intent(in) :: follow_old
    double precision :: he,hc,rate,follow,dead,alive,alpha,target,w,accrual,power
    logical :: ok, continued
    integer :: status
    he=he0; hc=hc0; rate=rate0; follow=follow0; dead=dead0; alive=alive0
    alpha=.05d0; target=.8d0; w=0d0; accrual=1d0; continued=follow_old
    ok=qschc(he,rate,accrual,follow,alpha,target,hc,dead,alive,w,continued,7,status)
    power=pschc(he,rate,accrual,follow,alpha,hc,dead,alive,w,continued)
    write(*,'(A,6(",",ES25.17),",",L1,5(",",ES25.17),",",I0,",",L1)') &
      name,he,hc,rate,follow,dead,alive,continued,alpha,target,w,accrual,power,status,ok
  end subroutine
end program
