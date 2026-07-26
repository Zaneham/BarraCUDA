! sswap -- BLAS level-1 x <-> y, elementwise. Two output buffers.
! Known-good: x=3, y=8 -> x=8, y=3, exact.
subroutine sswap(x, y, n)
   implicit none
   integer, intent(in) :: n
   real, intent(inout) :: x(n), y(n)
   integer :: i
   real :: t
   do concurrent (i = 1:n)
      t = x(i)
      x(i) = y(i)
      y(i) = t
   end do
end subroutine sswap
