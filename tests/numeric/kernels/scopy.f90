! scopy -- BLAS level-1 y := x.
! Known-good: x=7 -> every element of y is 7, exact.
subroutine scopy(x, y, n)
   implicit none
   integer, intent(in) :: n
   real, intent(in) :: x(n)
   real, intent(out) :: y(n)
   integer :: i
   do concurrent (i = 1:n)
      y(i) = x(i)
   end do
end subroutine scopy
