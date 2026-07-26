! saxpy -- BLAS level-1 y := a*x + y, as a do concurrent loop.
! LFortran offloads the loop to a kernel; kath compiles it for each backend.
! Known-good: with x=1, y=3, a=2 every element comes back 5, exactly.
subroutine saxpy(x, y, a, n)
   implicit none
   integer, intent(in) :: n
   real, intent(in) :: x(n), a
   real, intent(inout) :: y(n)
   integer :: i
   do concurrent (i = 1:n)
      y(i) = a * x(i) + y(i)
   end do
end subroutine saxpy
