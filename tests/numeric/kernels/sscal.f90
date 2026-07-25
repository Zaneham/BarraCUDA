! sscal -- BLAS level-1 x := a*x, in place.
! Known-good: x=2, a=3 -> every element 6, exact.
subroutine sscal(x, a, n)
   implicit none
   integer, intent(in) :: n
   real, intent(in) :: a
   real, intent(inout) :: x(n)
   integer :: i
   do concurrent (i = 1:n)
      x(i) = a * x(i)
   end do
end subroutine sscal
