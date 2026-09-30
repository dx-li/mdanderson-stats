! Independent driver around the unchanged STPLAN 4.5 matched-pairs routines.
! The original routines are linked from the ignored research/raw source tree.
program reference_stplan_matched_pairs
  implicit none
  logical, external :: qmpd
  real(8), external :: pmpd
  real(8) :: delta, n, alpha, target, z11, z10, z01, z00
  real(8) :: returned, achieved, one_sided_alpha, psi
  real(8), external :: ppsi
  integer :: i, iwhich, sides, status
  logical :: ok
  character(len=40), parameter :: ids(*) = [character(len=40) :: &
       'known_asymmetric_one_sided', 'known_asymmetric_two_sided', &
       'pilot_counts_scaled_same_proportions', 'inverse_difference', 'inverse_alpha', &
       'inverse_n', 'low_target_n_native', 'no_pilot_theta_unequal', &
       'no_pilot_default', 'forward_power_nonzero_delta', 'low_target_no_pilot_lost_sign', &
       'pilot_all_concordant_positive_delta']
  integer, parameter :: whiches(*) = [4,4,4,1,3,2,2,5,6,4,5,4]
  integer, parameter :: side_values(*) = [1,2,1,1,2,1,1,2,1,2,1,1]
  real(8), parameter :: deltas(*) = [.35d0,.35d0,.35d0,.5d0,.35d0, &
       .35d0,.35d0,.3d0,.3d0,.05d0,.05d0,.1d0]
  real(8), parameter :: ns(*) = [80d0,80d0,80d0,80d0,80d0, &
       80d0,80d0,80d0,80d0,30d0,80d0,50d0]
  real(8), parameter :: alphas(*) = [.05d0,.05d0,.05d0,.05d0,.05d0, &
       .05d0,.05d0,.05d0,.05d0,.01d0,.05d0,.05d0]
  real(8), parameter :: targets(*) = [.8d0,.8d0,.8d0,.8d0,.8d0, &
       .8d0,.1d0,.8d0,.8d0,.8d0,.01d0,.8d0]
  real(8), parameter :: z11s(*) = [28d0,28d0,280d0,28d0,28d0, &
       28d0,28d0,0d0,0d0,28d0,0d0,20d0]
  real(8), parameter :: z10s(*) = [17d0,17d0,170d0,17d0,17d0, &
       17d0,17d0,.2d0,0d0,9d0,.2d0,0d0]
  real(8), parameter :: z01s(*) = [9d0,9d0,90d0,9d0,9d0, &
       9d0,9d0,.65d0,0d0,17d0,.65d0,0d0]
  real(8), parameter :: z00s(*) = [26d0,26d0,260d0,26d0,26d0, &
       26d0,26d0,0d0,0d0,26d0,0d0,80d0]

  print '(A)', 'case_id,iwhich,sides,delta,n,alpha,target,z11,z10,z01,z00,ok,status,result,achieved,recommendation'
  do i = 1, size(ids)
    delta = deltas(i); n = ns(i); alpha = alphas(i); target = targets(i)
    z11 = z11s(i); z10 = z10s(i); z01 = z01s(i); z00 = z00s(i)
    sides = side_values(i); iwhich = whiches(i); status = -999
    ok = qmpd(delta,n,alpha,target,sides,iwhich,status,z11,z10,z01,z00)
    one_sided_alpha = alpha
    if (sides == 2) one_sided_alpha = alpha / 2d0
    if (iwhich == 5) then
      psi = z10 + z01 - 2d0*z10*z01
      achieved = ppsi(delta,n,one_sided_alpha,psi)
      returned = n / 4d0
    else if (iwhich == 6) then
      psi = .1d0 + .9d0 - 2d0*.1d0*.9d0
      achieved = ppsi(delta,n,one_sided_alpha,psi)
      returned = n / 6d0
    else
      achieved = pmpd(delta,n,one_sided_alpha,z11,z10,z01,z00)
      returned = -1d0
    end if
    if (iwhich == 5 .or. iwhich == 6) then
      write(*,'(A,",",I0,",",I0,8(",",ES24.16E3),",",L1,",",I0,3(",",ES24.16E3))') &
           trim(ids(i)),iwhich,sides,deltas(i),ns(i),alphas(i),targets(i), &
           z11s(i),z10s(i),z01s(i),z00s(i),ok,status,n,achieved,returned
    else
      returned = -1d0
      if (iwhich == 1) returned = delta
      if (iwhich == 2) returned = n
      if (iwhich == 3) returned = alpha
      if (iwhich == 4) returned = target
      write(*,'(A,",",I0,",",I0,8(",",ES24.16E3),",",L1,",",I0,3(",",ES24.16E3))') &
           trim(ids(i)),iwhich,sides,deltas(i),ns(i),alphas(i),targets(i), &
           z11s(i),z10s(i),z01s(i),z00s(i),ok,status,returned,achieved,-1d0
    end if
  end do
end program reference_stplan_matched_pairs

! Minimal local shim for the original QRANGE diagnostic-unit dependency.
subroutine gtcuio(input_unit, output_unit)
  implicit none
  integer, intent(out) :: input_unit, output_unit
  input_unit = 5
  output_unit = 6
end subroutine gtcuio

real(8) function ppsi(delta, n, alpha, psi)
  implicit none
  real(8), intent(in) :: delta, n, alpha, psi
  real(8), external :: dinvnr
  real(8) :: argument, denominator, cumulative, complement
  denominator = sqrt(psi**2 - .25d0 * delta**2 * (3d0 + psi))
  argument = (-dinvnr(1d0-alpha,alpha)*psi + &
       abs(delta)*sqrt(n*psi)) / denominator
  call cumnor(argument,cumulative,complement)
  ppsi = cumulative
end function ppsi
