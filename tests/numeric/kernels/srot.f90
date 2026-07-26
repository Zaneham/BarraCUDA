subroutine srot(x, y, c, s, n)
  integer, intent(in) :: n
  real, intent(inout) :: x(n), y(n)
  real, intent(in) :: c, s
  real :: t
  integer :: i
  do concurrent (i = 1:n)
    t = c*x(i) + s*y(i)
    y(i) = c*y(i) - s*x(i)
    x(i) = t
  end do
end subroutine
