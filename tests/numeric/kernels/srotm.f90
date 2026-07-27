subroutine srotm(x, y, h11, h12, h21, h22, n)
  integer, intent(in) :: n
  real, intent(inout) :: x(n), y(n)
  real, intent(in) :: h11, h12, h21, h22
  real :: w, z
  integer :: i
  do concurrent (i = 1:n)
    w = x(i)
    z = y(i)
    x(i) = w*h11 + z*h12
    y(i) = w*h21 + z*h22
  end do
end subroutine
