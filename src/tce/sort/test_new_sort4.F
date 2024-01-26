! test
! nvfortran  -i8 -g -Mdalign -Kieee -Mllalign -Mbackslash -Mcache_align -Mextend -stdpar=gpu -gpu=managed -cuda -cudalib=cublas,cutensor -fast test_new_sort4.F tce_sort4_cutensor.F -L/opt/nvidia/hpc_sdk/Linux_x86_64/nightly_240121/compilers/lib -lblas_ilp64 && ./a.out 30
!
      program test_new_sort4
      use omp_lib
      implicit none
      double precision :: f
      double precision, allocatable, dimension(:) :: u,s,r,x,y
      double precision :: tt0, tt1, dt0, dt1, dt2, dt3
      double precision :: junk
      integer :: T1,T2,T3,T4
      integer :: i,j,k,l,xxx
      integer :: num
      integer :: seed1,seed2
      double precision :: tmin,tmax,tavg
      double precision :: rmin,rmax,ravg
      double precision :: cmin,cmax,cavg
      double precision :: z
      double precision :: ddot
      external :: ddot
      integer :: argnum, arglen, err
      character(len=32) :: argtmp
      if (command_argument_count().eq.2) then
        call get_command_argument(1,argtmp,arglen,err)
        if (err.eq.0) read(argtmp,'(i4)') T1
        T2 = T1
        T3 = T1
        T4 = T1
      else if (command_argument_count().eq.5) then
        call get_command_argument(1,argtmp,arglen,err)
        if (err.eq.0) read(argtmp,'(i4)') T1
        call get_command_argument(2,argtmp,arglen,err)
        if (err.eq.0) read(argtmp,'(i4)') T2
        call get_command_argument(3,argtmp,arglen,err)
        if (err.eq.0) read(argtmp,'(i4)') T3
        call get_command_argument(4,argtmp,arglen,err)
        if (err.eq.0) read(argtmp,'(i4)') T4
      else
        T1 = 20
        T2 = T1
        T3 = T1
        T4 = T1
      endif
      print*,'{tce,new}_sort_4 test with tilesize = ',T1,T2,T3,T4
      allocate( u(1:T1*T2*T3*T4),
     &          s(1:T1*T2*T3*T4),
     &          r(1:T1*T2*T3*T4),
     &          x(1:T1*T2*T3*T4),
     &          y(1:T1*T2*T3*T4), stat=err)
      seed1 = 1
      seed2 = 2
      do i = 1,T1*T2*T3*T4
#if defined(GCC)
! GNU extension
        u(i) = rand(seed1)
        s(i) = 0.0
        r(i) = 0.0
#else
        u(i) = i
        s(i) = 0.0
        r(i) = 0.0
#endif
      enddo
      num = 0
      tmin = 999999999.9
      tmax = -999999999.9
      junk = 0.0
      do xxx=1,2
          tmin=9**9;tmax=0;tavg=0;
          rmin=9**9;rmax=0;ravg=0;
          cmin=9**9;cmax=0;cavg=0;
      write(6,*) '===================================='
      write(6,*) '(i,j,k,l),time,speedup'
! This is not particularly efficient but other methods
! are harder to implement in Fortran 77.
      do i = 1,4
      do j = 1,4
      do k = 1,4
      do l = 1,4
      if ((((i.ne.j).and.(i.ne.k)).and.(i.ne.l)).and.
     &    (((j.ne.k).and.(j.ne.l)).and.(k.ne.l))) then
        num = num+1
        f = 1.0 ! scaling factor
! reference
        tt0 = omp_get_wtime()
        call tce_sort_4(u,r,T1,T2,T3,T4,i,j,k,l,f)
        tt1 = omp_get_wtime()
        dt0 = tt1-tt0
        junk = junk + ddot(T1*T2*T3*T4,u,1,s,1)
        write(6,7) 'TCE',i,j,k,l,dt0,dt0/dt0
        ravg = ravg + dt0
        rmin = min(rmin,dt0)
        rmax = max(rmax,dt0)
! new
        tt0 = omp_get_wtime()
        call new_sort_4(u,s,T1,T2,T3,T4,i,j,k,l,f)
        tt1 = omp_get_wtime()
        dt1 = tt1-tt0
        junk  = junk + ddot(T1*T2*T3*T4,u,1,s,1)
        write(6,7) 'NEW',i,j,k,l,dt1,dt0/dt1
        tavg = tavg + (dt1)
        tmin = min(tmin,(dt1))
        tmax = max(tmax,(dt1))
! validate
        call diff4d(T1,T2,T3,T4,s,r,z)
        if (z.gt.0.0) then
            print*,'WRONG ',z
            exit
        endif
! cutensor
        tt0 = omp_get_wtime()
        call tce_sort_4_y(u,y,T1,T2,T3,T4,i,j,k,l,f)
        tt1 = omp_get_wtime()
        dt3 = tt1-tt0
        junk = junk + ddot(T1*T2*T3*T4,u,1,s,1)
        write(6,7) 'CUTENSOR',i,j,k,l,dt3,dt0/dt3
        cavg = cavg + dt0
        cmin = min(cmin,dt0)
        cmax = max(cmax,dt0)
! validate
        call diff4d(T1,T2,T3,T4,s,r,z)
        if (z.gt.0.0) then
            print*,'WRONG ',z
            exit
        endif
      endif
      enddo
      enddo
      enddo
      enddo
      enddo
      tavg = tavg/num
      ravg = ravg/num
      cavg = cavg/num
      write(6,1) rmin,rmax,ravg
      write(6,1) tmin,tmax,tavg
      print*,'junk = ',junk
   1  format('min = ',f10.5,' max = ',f10.5,' avg = ',f10.5)
   3  format(4a2,10x,10x,10x,10x)
   7  format(a8,1x,4i2,1x,1f10.5,' (s) ',f10.5)
      end program
      subroutine diff4d(T1,T2,T3,T4,x,y,z)
      implicit none
      integer i,j,k,l,n,T1,T2,T3,T4
      double precision, dimension(T1*T2*T3*T4) :: x,y
      double precision z
      z = 0.0
      do i = 1,T1*T2*T3*T4
        z=z+abs(x(i)-y(i))
      enddo
      end subroutine

      subroutine tce_sort_4(unsorted,sorted,a,b,c,d,i,j,k,l,factor)
      implicit none
      integer a,b,c,d
      integer i,j,k,l
      integer id(4),jd(4),ia,ib,j1,j2,j3,j4
      double precision sorted(a*b*c*d)
      double precision unsorted(a*b*c*d)
      double precision factor
      jd(1) = a
      jd(2) = b
      jd(3) = c
      jd(4) = d
      if ((l.eq.4).or.(l.eq.3)) then
      do j1 = 1,a
       id(1) = j1
       do j2 = 1,b
        id(2) = j2
        do j3 = 1,c
         id(3) = j3
         do j4 = 1,d
          id(4) = j4
          ia = id(4)+jd(4)*(id(3)-1+jd(3)*(id(2)-1+jd(2)*(id(1)-1)))
          ib = id(l)+jd(l)*(id(k)-1+jd(k)*(id(j)-1+jd(j)*(id(i)-1)))
          sorted(ib) = unsorted(ia) * factor
         enddo
        enddo
       enddo
      enddo
      else if (l.eq.2) then
      do j1 = 1,a
       id(1) = j1
       do j3 = 1,c
        id(3) = j3
        do j2 = 1,b
         id(2) = j2
         do j4 = 1,d
          id(4) = j4
          ia = id(4)+jd(4)*(id(3)-1+jd(3)*(id(2)-1+jd(2)*(id(1)-1)))
          ib = id(l)+jd(l)*(id(k)-1+jd(k)*(id(j)-1+jd(j)*(id(i)-1)))
          sorted(ib) = unsorted(ia) * factor
         enddo
        enddo
       enddo
      enddo
      else
      do j2 = 1,b
       id(2) = j2
       do j3 = 1,c
        id(3) = j3
        do j1 = 1,a
         id(1) = j1
         do j4 = 1,d
          id(4) = j4
          ia = id(4)+jd(4)*(id(3)-1+jd(3)*(id(2)-1+jd(2)*(id(1)-1)))
          ib = id(l)+jd(l)*(id(k)-1+jd(k)*(id(j)-1+jd(j)*(id(i)-1)))
          sorted(ib) = unsorted(ia) * factor
         enddo
        enddo
       enddo
      enddo
      endif
      return
      end

!
! written by Jeff Hammond, Intel Labs, June 2014
! see https://github.com/jeffhammond/nwchem-tce-triples-kernels/blob/master/src/new_sort4.f for history
!
! There is currently no difference between the OpenMP and non-OpenMP versions
! except for the directives, but preprocessing at fine granularity leads to ugly
! code because Fortran does not have anything like C99/C++11 _Pragma.
!
      subroutine new_sort_4(u,s,a,b,c,d,i,j,k,l,f)
      implicit none
      integer a,b,c,d
      integer i,j,k,l
      integer id(4),jd(4),ia,ib,j1,j2,j3,j4
      double precision s(a*b*c*d)
      double precision u(a*b*c*d)
      double precision f
      integer iax,ibx
      jd(1) = a
      jd(2) = b
      jd(3) = c
      jd(4) = d
      if (l.eq.4) then
!$omp  parallel do collapse(2)
!$omp& private(j1,j2,j3,j4,iax,ibx,ia,ib,id) shared(s,u)
!$omp& firstprivate(a,b,c,d,i,j,k,l,f,jd)
       do j1 = 1,a
        do j2 = 1,b
         do j3 = 1,c
          id(1) = j1
          id(2) = j2
          id(3) = j3
          !id(4) = j4 ! 1234 are only used once and we know l=4, hence ijk must not be 4
          iax = d*(j3-1+c*(j2-1+b*(j1-1)))
          ibx = d*(id(k)-1+jd(k)*(id(j)-1+jd(j)*(id(i)-1)))
!$omp simd
          do j4 = 1,d
           ia = j4+iax
           ib = j4+ibx
           s(ib) = u(ia) * f
          enddo
         enddo
        enddo
       enddo
!$omp end parallel do
      else if (l.eq.3) then
!$omp  parallel do collapse(2)
!$omp& private(j1,j2,j3,j4,iax,ibx,ia,ib,id) shared(s,u)
!$omp& firstprivate(a,b,c,d,i,j,k,l,f,jd)
       do j1 = 1,a
        do j2 = 1,b
         do j4 = 1,d
          id(1) = j1
          id(2) = j2
          !id(3) = j3
          id(4) = j4
          iax = -1+c*(j2-1+b*(j1-1))
          ibx = c*(id(k)-1+jd(k)*(id(j)-1+jd(j)*(id(i)-1)))
!$omp simd
          do j3 = 1,c
           ia = j4+d*(j3+iax)
           ib = j3+ibx
           s(ib) = u(ia) * f
          enddo
         enddo
        enddo
       enddo
!$omp end parallel do
      else if (l.eq.2) then
!$omp  parallel do collapse(2)
!$omp& private(j1,j2,j3,j4,iax,ibx,ia,ib,id) shared(s,u)
!$omp& firstprivate(a,b,c,d,i,j,k,l,f,jd)
       do j1 = 1,a
        do j3 = 1,c
         do j4 = 1,d
          id(1) = j1
          !id(2) = j2
          id(3) = j3
          id(4) = j4
          iax = -1+b*(j1-1)
          ibx = b*(id(k)-1+jd(k)*(id(j)-1+jd(j)*(id(i)-1)))
!$omp simd
          do j2 = 1,b
           ia = j4+d*(j3-1+c*(j2+iax))
           ib = j2+ibx
           s(ib) = u(ia) * f
          enddo
         enddo
        enddo
       enddo
!$omp end parallel do
      else ! l.eq.1
!$omp  parallel do collapse(2)
!$omp& private(j1,j2,j3,j4,iax,ibx,ia,ib,id) shared(s,u)
!$omp& firstprivate(a,b,c,d,i,j,k,l,f,jd)
       do j2 = 1,b
        do j3 = 1,c
         do j4 = 1,d
          !id(1) = j1
          id(2) = j2
          id(3) = j3
          id(4) = j4
          ibx = a*(id(k)-1+jd(k)*(id(j)-1+jd(j)*(id(i)-1)))
!$omp simd
          do j1 = 1,a
           ia = j4+d*(j3-1+c*(j2-1+b*(j1-1)))
           ib = j1+ibx
           s(ib) = u(ia) * f
          enddo
         enddo
        enddo
       enddo
!$omp end parallel do
      endif
      return
      end

      subroutine new_sortacc_4(u,s,a,b,c,d,i,j,k,l,f)
      implicit none
      integer a,b,c,d
      integer i,j,k,l
      integer id(4),jd(4),ia,ib,j1,j2,j3,j4
      double precision s(a*b*c*d)
      double precision u(a*b*c*d)
      double precision f
      integer iax,ibx
      jd(1) = a
      jd(2) = b
      jd(3) = c
      jd(4) = d
      if (l.eq.4) then
!$omp  parallel do collapse(2)
!$omp& private(j1,j2,j3,j4,iax,ibx,ia,ib,id) shared(s,u)
!$omp& firstprivate(a,b,c,d,i,j,k,l,f,jd)
       do j1 = 1,a
        do j2 = 1,b
         do j3 = 1,c
          id(1) = j1
          id(2) = j2
          id(3) = j3
          !id(4) = j4 ! 1234 are only used once and we know l=4, hence ijk must not be 4
          iax = d*(j3-1+c*(j2-1+b*(j1-1)))
          ibx = d*(id(k)-1+jd(k)*(id(j)-1+jd(j)*(id(i)-1)))
!$omp simd
          do j4 = 1,d
           ia = j4+iax
           ib = j4+ibx
           s(ib) = s(ib) + u(ia) * f
          enddo
         enddo
        enddo
       enddo
!$omp end parallel do
      else if (l.eq.3) then
!$omp  parallel do collapse(2)
!$omp& private(j1,j2,j3,j4,iax,ibx,ia,ib,id) shared(s,u)
!$omp& firstprivate(a,b,c,d,i,j,k,l,f,jd)
       do j1 = 1,a
        do j2 = 1,b
         do j4 = 1,d
          id(1) = j1
          id(2) = j2
          !id(3) = j3
          id(4) = j4
          iax = -1+c*(j2-1+b*(j1-1))
          ibx = c*(id(k)-1+jd(k)*(id(j)-1+jd(j)*(id(i)-1)))
!$omp simd
          do j3 = 1,c
           ia = j4+d*(j3+iax)
           ib = j3+ibx
           s(ib) = s(ib) + u(ia) * f
          enddo
         enddo
        enddo
       enddo
!$omp end parallel do
      else if (l.eq.2) then
!$omp  parallel do collapse(2)
!$omp& private(j1,j2,j3,j4,iax,ibx,ia,ib,id) shared(s,u)
!$omp& firstprivate(a,b,c,d,i,j,k,l,f,jd)
       do j1 = 1,a
        do j3 = 1,c
         do j4 = 1,d
          id(1) = j1
          !id(2) = j2
          id(3) = j3
          id(4) = j4
          iax = -1+b*(j1-1)
          ibx = b*(id(k)-1+jd(k)*(id(j)-1+jd(j)*(id(i)-1)))
!$omp simd
          do j2 = 1,b
           ia = j4+d*(j3-1+c*(j2+iax))
           ib = j2+ibx
           s(ib) = s(ib) + u(ia) * f
          enddo
         enddo
        enddo
       enddo
!$omp end parallel do
      else ! l.eq.1
!$omp  parallel do collapse(2)
!$omp& private(j1,j2,j3,j4,iax,ibx,ia,ib,id) shared(s,u)
!$omp& firstprivate(a,b,c,d,i,j,k,l,f,jd)
       do j2 = 1,b
        do j3 = 1,c
         do j4 = 1,d
          !id(1) = j1
          id(2) = j2
          id(3) = j3
          id(4) = j4
          ibx = a*(id(k)-1+jd(k)*(id(j)-1+jd(j)*(id(i)-1)))
!$omp simd
          do j1 = 1,a
           ia = j4+d*(j3-1+c*(j2-1+b*(j1-1)))
           ib = j1+ibx
           s(ib) = s(ib) + u(ia) * f
          enddo
         enddo
        enddo
       enddo
!$omp end parallel do
      endif
      return
      end
