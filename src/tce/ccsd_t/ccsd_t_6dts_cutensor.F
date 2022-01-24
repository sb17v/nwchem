      module tce_ccsd_t_cutensor

      use cudafor
      implicit none

      contains

      SUBROUTINE ccsd_t_6dts_cutensor(d_t1,k_t1_offset,d_t2,k_t2_offset,
     &            d_v2,k_v2_offset,energy1,energy2,size_t1,xmem)
      use cudafor
      implicit none
#include "global.fh"
#include "mafdecls.fh"
#include "util.fh"
#include "errquit.fh"
#include "tce.fh"
#include "tce_main.fh"
      double precision, intent(inout) :: energy1,energy2
      integer, intent(in) :: d_t1, d_t2, d_v2
      integer, intent(in) :: k_t1_offset, k_t2_offset, k_v2_offset
      integer, intent(in) :: xmem, size_t1
      integer :: t_h1b, t_h1, t_h2b, t_h2, t_h3b, t_h3
      integer :: t_p4b, t_p4, t_p5b, t_p5, t_p6b, t_p6
      double precision, allocatable :: f_singles(:)
      double precision, allocatable :: f_doubles1(:)
      double precision, allocatable :: f_doubles2(:)
      double precision, allocatable :: t1_local(:)
      integer :: alloc_error,cuda_err
      integer :: size,i
      integer :: nxtask
      integer :: next
      integer :: nprocs
      integer :: count
      integer :: offset_p4,offset_p5,offset_p6
      integer :: offset_h1,offset_h2,offset_h3
      integer :: range_p4,range_p5,range_p6
      integer :: range_h1,range_h2,range_h3
      integer :: spin_p4b,spin_p5b,spin_p6b
      integer :: spin_h1b,spin_h2b,spin_h3b
      integer :: sym_p4b,sym_p5b,sym_p6b
      integer :: sym_h1b,sym_h2b,sym_h3b
      integer :: dp4,dp5,ii,jj,istart,istop,jstart,jstop
      integer :: mdp4,mdp5
      integer :: maxp4,maxp5
      integer :: slice_dp4,slice_dp5,qp4,qp5
      integer :: xlocal
! ---------------------------------------
      integer :: maxh, maxp,h,p
! ---------------------------------------
      double precision :: energy(2)
      double precision :: factor,denom
      double precision :: e_s, e_d, e_d1, e_d2
      double precision :: denom_p4,denom_p5,denom_p6
      double precision :: denom_h1,denom_h2,denom_h3
      integer(kind=cuda_stream_kind) :: stream(3)
      integer :: nevals
      double precision, allocatable :: evals(:)
      external nxtask

      do p=1,3
        cuda_err = cudaStreamCreate(stream(p))
        if (cuda_err.ne.0) then
          call errquit('cudaStreamCreate',cuda_err,UNKNOWN_ERR)
        endif
      enddo
!
! Here we determine the maximum tile dimension
! for p and h tiles so we can allocate buffers
! up front once rather than in a loop.
!
! As a lose upper bound, we can use tile_dim instead,
! but for occupied tiles, this may be a very loose bound.
!
      maxp = 0
      do p = noab+1,noab+nvab
         maxp = max(maxp,int_mb(k_range+p-1))
      enddo
      maxh = 0
      do h = 1,noab
         maxh = max(maxh,int_mb(k_range+h-1))
      enddo

! - T1/X1 LOCALIZATION ----------
      allocate(t1_local(1:size_t1), stat=alloc_error)
      if (alloc_error.ne.0) then
        call errquit('ccsd_t_6dts: allocate failed',
     &               alloc_error, MA_ERR)
      endif
      t1_local = 0.0d0
      call get_block(d_t1,t1_local,size_t1,0)
! -------------------------------

      block
        integer :: x, y, ymin, ymax
        ymin = huge(x)
        ymax = 0
        do x = 1,noab+nvab
           !y = k_evl_sorted+int_mb(k_offset+x-1)-1
           y = int_mb(k_offset+x-1)-1
           ymin = min(y,ymin)
           ymax = max(y,ymax)
        enddo
        write(6,'(a4,i14,1x,a4,i14,1x,a5,i15)') 
     &          'min=',ymin,'max=',ymax,'diff=',ymax-ymin
      end block

      nevals = nmo(1)-nfv(1)-nfc(1)+nmo(ipol)-nfv(ipol)-nfc(ipol)
      allocate(evals(1:nevals), stat=alloc_error)
      if (alloc_error.ne.0) then
        call errquit('ccsd_t_6dts: allocate failed',
     &               alloc_error, MA_ERR)
      endif
      evals(1:nevals) = dbl_mb(k_evl_sorted:k_evl_sorted+nevals)

      nprocs = GA_NNODES()
      count = 0
      next = nxtask(nprocs,1)
      energy1 = 0.0d0
      energy2 = 0.0d0
      do t_p4b = noab+1,noab+nvab
       range_p4 = int_mb(k_range+t_p4b-1)
       offset_p4 = k_evl_sorted+int_mb(k_offset+t_p4b-1)-1
       spin_p4b=int_mb(k_spin+t_p4b-1)
       sym_p4b=int_mb(k_sym+t_p4b-1)
       do t_p5b = t_p4b,noab+nvab
        range_p5 = int_mb(k_range+t_p5b-1)
        offset_p5 = k_evl_sorted+int_mb(k_offset+t_p5b-1)-1
        spin_p5b=int_mb(k_spin+t_p5b-1)
        sym_p5b=int_mb(k_sym+t_p5b-1)
        do t_p6b = t_p5b,noab+nvab
         range_p6 = int_mb(k_range+t_p6b-1)
         offset_p6 = k_evl_sorted+int_mb(k_offset+t_p6b-1)-1
         spin_p6b=int_mb(k_spin+t_p6b-1)
         sym_p6b=int_mb(k_sym+t_p6b-1)
         do t_h1b = 1,noab
          range_h1 = int_mb(k_range+t_h1b-1)
          offset_h1 = k_evl_sorted+int_mb(k_offset+t_h1b-1)-1
          spin_h1b=int_mb(k_spin+t_h1b-1)
          sym_h1b=int_mb(k_sym+t_h1b-1)
          do t_h2b = t_h1b,noab
           range_h2 = int_mb(k_range+t_h2b-1)
           offset_h2 = k_evl_sorted+int_mb(k_offset+t_h2b-1)-1
           spin_h2b=int_mb(k_spin+t_h2b-1)
           sym_h2b=int_mb(k_sym+t_h2b-1)
           do t_h3b = t_h2b,noab
            range_h3 = int_mb(k_range+t_h3b-1)
            offset_h3 = k_evl_sorted+int_mb(k_offset+t_h3b-1)-1
            spin_h3b=int_mb(k_spin+t_h3b-1)
            sym_h3b=int_mb(k_sym+t_h3b-1)
            if (spin_p4b+spin_p5b+spin_p6b
     &      .eq.spin_h1b+spin_h2b+spin_h3b) then
            if ((.not.restricted).or.
     &         (spin_p4b+spin_p5b+spin_p6b+
     &          spin_h1b+spin_h2b+spin_h3b.le.8)) then
            if (ieor(sym_p4b,ieor(sym_p5b,
     &          ieor(sym_p6b,ieor(sym_h1b,
     &          ieor(sym_h2b,sym_h3b))))).eq.0) then
            if (next.eq.count) then
! -- 6DTS --
            dp4=1
            dp5=1
  300       continue
            xlocal=((int_mb(k_range+t_p4b-1)/dp4)
     &           *  (int_mb(k_range+t_p5b-1)/dp5))
     &           * int_mb(k_range+t_p6b-1)
     &           * int_mb(k_range+t_h1b-1)
     &           * int_mb(k_range+t_h2b-1)
     &           * int_mb(k_range+t_h3b-1)
!
            slice_dp4=range_p4/dp4
            qp4=range_p4/slice_dp4
            slice_dp5=range_p5/dp5
            qp5=range_p5/slice_dp5
!
            if((xlocal)/((1024*1024)/8).gt.(xmem)) then
             if(dp4.lt.int_mb(k_range+t_p4b-1)) then
              dp4=dp4+1
             else
              dp5=dp5+1
             end if
             go to 300
            end if
!
            if(slice_dp4*qp4.ne.range_p4) then
              mdp4=qp4+1
            else
              mdp4=qp4
            end if
            if(slice_dp5*qp5.ne.range_p5) then
              mdp5=qp5+1
            else
              mdp5=qp5
            end if
!
           size = slice_dp4 * slice_dp5
     &          * range_p6  * range_h1
     &          * range_h2  * range_h3
!
            maxp4=slice_dp4
            maxp5=slice_dp5
!
            allocate(f_singles(1:size), f_doubles1(1:size),
     &               f_doubles2(1:size), stat=alloc_error)
            if (alloc_error.ne.0) then
              call errquit('ccsd_t_6dts_cutensor: allocate failed',
     &                     alloc_error, MA_ERR)
            endif
!  6DTS ----
           do ii=1,mdp4 !do ii ---
           do jj=1,mdp5 !do jj ---
            if(ii.ne.mdp4) then
             istart=(ii-1)*(slice_dp4)+1
             istop=ii*(slice_dp4)
            else
             istart=(ii-1)*(slice_dp4)+1
             istop=int_mb(k_range+t_p4b-1)
            end if
            if(jj.ne.mdp5) then
             jstart=(jj-1)*(slice_dp5)+1
             jstop=jj*(slice_dp5)
            else
             jstart=(jj-1)*(slice_dp5)+1
             jstop=int_mb(k_range+t_p5b-1)
            end if
            f_singles = 0.0d0
            f_doubles1 = 0.0d0
            f_doubles2 = 0.0d0
            call ccsd_t_singles_l_1_6dts_cutensor(f_singles,
     &        t1_local,d_v2,k_t1_offset,k_v2_offset,
     &        t_h1b,t_h2b,t_h3b,t_p4b,t_p5b,t_p6b,
     &        istart,istop,jstart,jstop,maxp4,maxp5,maxp,maxh,
     &        stream(1))
            call ccsd_t_6dts_l_1_cutensor(f_doubles1,
     &        d_t2,d_v2,k_t2_offset,k_v2_offset,
     &        t_h1b,t_h2b,t_h3b,t_p4b,t_p5b,t_p6b,
     &        istart,istop,jstart,jstop,maxp4,maxp5,maxp,maxh,
     &        stream(2))
            call ccsd_t_6dts_l_2_cutensor(f_doubles2,
     &        d_t2,d_v2,k_t2_offset,k_v2_offset,
     &        t_h1b,t_h2b,t_h3b,t_p4b,t_p5b,t_p6b,
     &        istart,istop,jstart,jstop,maxp4,maxp5,maxp,maxh,
     &        stream(3))
            if (restricted) then
              factor = 2.0d0
            else
              factor = 1.0d0
            endif
            if ((t_p4b.eq.t_p5b).and.(t_p5b.eq.t_p6b)) then
              factor = factor / 6.0d0
            else if ((t_p4b.eq.t_p5b).or.(t_p5b.eq.t_p6b)) then
              factor = factor / 2.0d0
            endif
            if ((t_h1b.eq.t_h2b).and.(t_h2b.eq.t_h3b)) then
              factor = factor / 6.0d0
            else if ((t_h1b.eq.t_h2b).or.(t_h2b.eq.t_h3b)) then
              factor = factor / 2.0d0
            endif
!
! factor = [ 1/36, 1/18, 1/12, 1/6, 1/4, 1/3, 1/2, 1, 2]
!
!            do t_p4 = 1,range_p4
!             denom_p4 = dbl_mb(offset_p4+t_p4)
!             print*,'index_p=',offset_p4+t_p4-k_evl_sorted
!            enddo
!            do t_h3 = 1,range_h3
!             denom_h3 = dbl_mb(offset_h3+t_h3)
!             print*,'index_h=',offset_h3+t_h3-k_evl_sorted
!            enddo

            i = 1
            print*,'SANDWITCH',util_wallsec()
#if 1
!$acc parallel loop collapse(6) reduction(+:energy1,energy2)
!$acc& copyin(i,factor,f_singles,f_doubles1,f_doubles2)
!$acc& copyin(k_evl_sorted,evals)
!$acc& copyin(offset_p4,offset_p5,offset_p6)
!$acc& copyin(offset_h1,offset_h2,offset_h3)
!$acc& copyin(t_p4,t_p5,t_p6,t_h1,t_h2,t_h3)
!$acc& copyin(istart,istop,range_p6,range_h1,range_h2,range_h3)
!$acc& private(denom_p4,denom_p5,denom_p6,denom_h1,denom_h2,denom_h3)
!$acc& private(denom,e_s,e_d1,e_d2,e_d)
#endif
            do t_p4 = istart,istop
             !denom_p4 = dbl_mb(offset_p4+t_p4)
             do t_p5 = jstart,jstop
              !denom_p5 = dbl_mb(offset_p5+t_p5)
              do t_p6 = 1, range_p6
               !denom_p6 = dbl_mb(offset_p6+t_p6)
               do t_h1 = 1, range_h1
                !denom_h1 = dbl_mb(offset_h1+t_h1)
                do t_h2 = 1, range_h2
                 !denom_h2 = dbl_mb(offset_h2+t_h2)
                 do t_h3 = 1, range_h3
                  !denom_h3 = dbl_mb(offset_h3+t_h3)
                  denom_p4 = evals(offset_p4+t_p4-k_evl_sorted+1)
                  denom_p5 = evals(offset_p5+t_p5-k_evl_sorted+1)
                  denom_p6 = evals(offset_p6+t_p6-k_evl_sorted+1)
                  denom_h1 = evals(offset_h1+t_h1-k_evl_sorted+1)
                  denom_h2 = evals(offset_h2+t_h2-k_evl_sorted+1)
                  denom_h3 = evals(offset_h3+t_h3-k_evl_sorted+1)
                  denom = 1.0d0 / ( ( denom_h1 + denom_h2 + denom_h3 )
     &                            - ( denom_p4 + denom_p5 + denom_p6 ) )
                  e_s  = f_singles(i)
                  e_d1 = f_doubles1(i)
                  e_d2 = f_doubles2(i)
                  e_d  = e_d1 + e_d2
                  energy1 = energy1 + factor*denom * e_d * e_d
                  energy2 = energy2 + factor*denom * e_d * (e_d + e_s)
                  i = i + 1
                 enddo
                enddo
               enddo
              enddo
             enddo
            enddo
!
            enddo !do ii ---
            enddo !do jj ---
!
            deallocate(f_singles, f_doubles1, f_doubles2,
     &                 stat=alloc_error)
            if (alloc_error.ne.0) then
              call errquit('ccst_t_6dts: deallocate failed',
     &                     alloc_error, MA_ERR)
            endif
!
            next = nxtask(nprocs,1)
            endif
            count = count + 1
!
            endif
            endif
            endif
           enddo
          enddo
         enddo
        enddo
       enddo
      enddo
      next = nxtask(-nprocs,1)
      energy(1) = energy1
      energy(2) = energy2
      call ga_dgop(mt_dbl,energy,2,'+')
      energy1 = energy(1)
      energy2 = energy(2)
! - T1/X1 LOCALIZATION ------
      deallocate(t1_local, stat=alloc_error)
      if (alloc_error.ne.0)  then
        call errquit('ccsd_t_6dts: deallocate failed',
     &               alloc_error, MA_ERR)
      endif
! ---------------------------
      do p=1,3
        cuda_err = cudaStreamDestroy(stream(p))
        if (cuda_err.ne.0) then
          call errquit('cudaStreamDestroy',cuda_err,UNKNOWN_ERR)
        endif
      end do
      end

      SUBROUTINE ccsd_t_singles_l_1_6dts_cutensor(f_t3,f_t1,
     &           d_v2,k_t1_offset,k_v2_offset,
     &           t_h1b,t_h2b,t_h3b,t_p4b,t_p5b,t_p6b,
     &           istart,istop,jstart,jstop,maxp4,maxp5,maxp,maxh,
     &           stream)
!     i0 ( p4 p5 p6 h1 h2 h3 )_vt + = 1 * P( 9 ) * t ( p4 h1 )_t * v ( p5 p6 h2 h3 )_v
!
! elimination of the transpositions (kowalski,august,2009)
!
      implicit none
#include "global.fh"
#include "mafdecls.fh"
#include "sym.fh"
#include "errquit.fh"
#include "tce.fh"
      double precision, intent(inout) :: f_t3(*)
      double precision, intent(in) :: f_t1(*)
      integer, intent(in) :: d_v2, k_t1_offset, k_v2_offset
      integer, intent(in) :: t_p4b, t_p5b, t_p6b, t_h1b, t_h2b, t_h3b
      integer, intent(in) :: istart,istop,jstart,jstop,maxp4,maxp5
      integer, intent(in) :: maxp,maxh
      integer(kind=cuda_stream_kind), intent(inout) :: stream
      integer :: p4b, p5b, p6b, h1b, h2b, h3b
      integer :: p4b_1, h1b_1, p5b_2, p6b_2, h2b_2, h3b_2
      integer :: dima, dimb
      integer :: a3(9,6)
      integer :: ia6,ja6
      double precision, allocatable :: f_t1_temp(:), f_t1_sort(:,:)
      double precision, allocatable :: f_v2_sort(:,:)
      integer :: dimhp, dimhhpp
      integer :: alloc_error
      integer :: range_p4b,range_p5b,range_p6b
      integer :: range_h1b,range_h2b,range_h3b
      integer :: spin_p4b,spin_p5b,spin_p6b
      integer :: spin_h1b,spin_h2b,spin_h3b
      integer :: sym_p4b,sym_p5b,sym_p6b
      integer :: sym_h1b,sym_h2b,sym_h3b
      integer :: cc,comm

      dimhp   = maxh*maxp
      dimhhpp = maxh*maxh*maxp*maxp

      allocate( f_t1_temp(1:dimhp), f_t1_sort(1:dimhp,9),
     &          f_v2_sort(1:dimhhpp,9), stat=alloc_error)
      if (alloc_error.ne.0) then
        call errquit("MA ccsd_t_singles_l_1_6dts_cutensor",
     &               2*dimhhpp,MA_ERR)
      endif
!
      a3(1,1)=t_p4b
      a3(1,2)=t_p5b
      a3(1,3)=t_p6b
      a3(1,4)=t_h1b
      a3(1,5)=t_h2b
      a3(1,6)=t_h3b
!
      a3(2,1)=t_p4b
      a3(2,2)=t_p5b
      a3(2,3)=t_p6b
      a3(2,4)=t_h2b
      a3(2,5)=t_h1b
      a3(2,6)=t_h3b
!
      a3(3,1)=t_p4b
      a3(3,2)=t_p5b
      a3(3,3)=t_p6b
      a3(3,4)=t_h3b
      a3(3,5)=t_h1b
      a3(3,6)=t_h2b
!
      a3(4,1)=t_p5b
      a3(4,2)=t_p4b
      a3(4,3)=t_p6b
      a3(4,4)=t_h1b
      a3(4,5)=t_h2b
      a3(4,6)=t_h3b
!
      a3(5,1)=t_p5b
      a3(5,2)=t_p4b
      a3(5,3)=t_p6b
      a3(5,4)=t_h2b
      a3(5,5)=t_h1b
      a3(5,6)=t_h3b
!
      a3(6,1)=t_p5b
      a3(6,2)=t_p4b
      a3(6,3)=t_p6b
      a3(6,4)=t_h3b
      a3(6,5)=t_h1b
      a3(6,6)=t_h2b
!
      a3(7,1)=t_p6b
      a3(7,2)=t_p4b
      a3(7,3)=t_p5b
      a3(7,4)=t_h1b
      a3(7,5)=t_h2b
      a3(7,6)=t_h3b
!
      a3(8,1)=t_p6b
      a3(8,2)=t_p4b
      a3(8,3)=t_p5b
      a3(8,4)=t_h2b
      a3(8,5)=t_h1b
      a3(8,6)=t_h3b
!
      a3(9,1)=t_p6b
      a3(9,2)=t_p4b
      a3(9,3)=t_p5b
      a3(9,4)=t_h3b
      a3(9,5)=t_h1b
      a3(9,6)=t_h2b
!
      do ia6=1,8
       if(a3(ia6,1).ne.0) then
       do ja6=ia6+1,9
        if((a3(ia6,1).eq.a3(ja6,1)).and.(a3(ia6,2).eq.a3(ja6,2))
     &   .and.(a3(ia6,3).eq.a3(ja6,3)).and.(a3(ia6,4).eq.a3(ja6,4))
     &   .and.(a3(ia6,5).eq.a3(ja6,5)).and.(a3(ia6,6).eq.a3(ja6,6)))
     &   then
         a3(ja6,1)=0
         a3(ja6,2)=0
         a3(ja6,3)=0
         a3(ja6,4)=0
         a3(ja6,5)=0
         a3(ja6,6)=0
        end if
       enddo 
       end if
      enddo 
!
      cc = 0
      comm = 0

      do ia6=1,9
          p4b=a3(ia6,1)
          p5b=a3(ia6,2)
          p6b=a3(ia6,3)
          h1b=a3(ia6,4)
          h2b=a3(ia6,5)
          h3b=a3(ia6,6)

      if((p5b.le.p6b).and.(h2b.le.h3b).and.(p4b.ne.0)) then

          spin_h1b=int_mb(k_spin+h1b-1)
          spin_h2b=int_mb(k_spin+h2b-1)
          spin_h3b=int_mb(k_spin+h3b-1)
          spin_p4b=int_mb(k_spin+p4b-1)
          spin_p5b=int_mb(k_spin+p5b-1)
          spin_p6b=int_mb(k_spin+p6b-1)

      IF ((.not.restricted).or.
     &    (spin_p4b+spin_p5b+spin_p6b+
     &     spin_h1b+spin_h2b+spin_h3b.ne.12)) THEN
      IF (spin_p4b+spin_p5b+spin_p6b .eq. 
     &    spin_h1b+spin_h2b+spin_h3b) THEN

          sym_h1b=int_mb(k_sym+h1b-1)
          sym_h2b=int_mb(k_sym+h2b-1)
          sym_h3b=int_mb(k_sym+h3b-1)
          sym_p4b=int_mb(k_sym+p4b-1)
          sym_p5b=int_mb(k_sym+p5b-1)
          sym_p6b=int_mb(k_sym+p6b-1)

      IF (ieor(sym_p4b,ieor(sym_p5b,ieor(sym_p6b,ieor(sym_h1b,
     &    ieor(sym_h2b,sym_h3b))))) .eq. ieor(irrep_v,irrep_t)) THEN

          range_h1b=int_mb(k_range+h1b-1)
          range_h2b=int_mb(k_range+h2b-1)
          range_h3b=int_mb(k_range+h3b-1)
          range_p4b=int_mb(k_range+p4b-1)
          range_p5b=int_mb(k_range+p5b-1)
          range_p6b=int_mb(k_range+p6b-1)

      IF (spin_p4b .eq. spin_h1b) THEN
      IF (ieor(sym_p4b,sym_h1b) .eq. irrep_t) THEN
      CALL TCE_RESTRICTED_2(p4b,h1b,p4b_1,h1b_1)
      CALL TCE_RESTRICTED_4(p5b,p6b,h2b,h3b,p5b_2,p6b_2,h2b_2,h3b_2)
      dima = range_p4b * range_h1b
      dimb = range_p5b * range_p6b * range_h2b * range_h3b
      IF ((dima .gt. 0) .and. (dimb .gt. 0)) THEN
      CALL GET_HASH_BLOCK_MA(f_t1,f_t1_temp,dima, int_mb(k_t1_offset),
     &                       (h1b_1 - 1 + noab * (p4b_1 - noab - 1)))
      comm = comm + 1
      CALL TCE_SORT_2(f_t1_temp,f_t1_sort(:,ia6),
     &     range_p4b,range_h1b,2,1,1.0d0)

      if(.not.intorb) then
      CALL GET_HASH_BLOCK(d_v2,f_v2_sort(:,ia6),
     &     dimb,int_mb(k_v2_offset),
     &     (h3b_2 - 1 + (noab+nvab) * (h2b_2 - 1 + (noab+nvab) *
     &      (p6b_2 - 1 + (noab+nvab) * (p5b_2 - 1)))))
      comm = comm + 1
      else
      CALL GET_HASH_BLOCK_I(d_v2,f_v2_sort(:,ia6),
     &     dimb,int_mb(k_v2_offset),
     &     (h3b_2 - 1 + (noab+nvab) * (h2b_2 - 1 + (noab+nvab) *
     &      (p6b_2 - 1 + (noab+nvab) * (p5b_2 - 1)))),
     &     h3b_2,h2b_2,p6b_2,p5b_2)
      comm = comm + 1
      end if ! .not.intorb

      END IF ! dima & dimb
      END IF ! sym 4 & 1 & irrep_t
      END IF ! spin 4 & 1
!
      END IF ! sym
      END IF ! spin x6
      END IF ! .not.restricted ...
!
      END IF ! (p5b.le.p6b).and.(h2b.le.h3b).and.(p4b.ne.0)
      end do ! ia6

      do ia6=1,9
          p4b=a3(ia6,1)
          p5b=a3(ia6,2)
          p6b=a3(ia6,3)
          h1b=a3(ia6,4)
          h2b=a3(ia6,5)
          h3b=a3(ia6,6)

      if((p5b.le.p6b).and.(h2b.le.h3b).and.(p4b.ne.0)) then

          spin_h1b=int_mb(k_spin+h1b-1)
          spin_h2b=int_mb(k_spin+h2b-1)
          spin_h3b=int_mb(k_spin+h3b-1)
          spin_p4b=int_mb(k_spin+p4b-1)
          spin_p5b=int_mb(k_spin+p5b-1)
          spin_p6b=int_mb(k_spin+p6b-1)

      IF ((.not.restricted).or.
     &    (spin_p4b+spin_p5b+spin_p6b+
     &     spin_h1b+spin_h2b+spin_h3b.ne.12)) THEN
      IF (spin_p4b+spin_p5b+spin_p6b .eq. 
     &    spin_h1b+spin_h2b+spin_h3b) THEN

          sym_h1b=int_mb(k_sym+h1b-1)
          sym_h2b=int_mb(k_sym+h2b-1)
          sym_h3b=int_mb(k_sym+h3b-1)
          sym_p4b=int_mb(k_sym+p4b-1)
          sym_p5b=int_mb(k_sym+p5b-1)
          sym_p6b=int_mb(k_sym+p6b-1)

      IF (ieor(sym_p4b,ieor(sym_p5b,ieor(sym_p6b,ieor(sym_h1b,
     &    ieor(sym_h2b,sym_h3b))))) .eq. ieor(irrep_v,irrep_t)) THEN

          range_h1b=int_mb(k_range+h1b-1)
          range_h2b=int_mb(k_range+h2b-1)
          range_h3b=int_mb(k_range+h3b-1)
          range_p4b=int_mb(k_range+p4b-1)
          range_p5b=int_mb(k_range+p5b-1)
          range_p6b=int_mb(k_range+p6b-1)

      IF (spin_p4b .eq. spin_h1b) THEN
      IF (ieor(sym_p4b,sym_h1b) .eq. irrep_t) THEN

      IF ((t_p4b .eq. p4b) .and. (t_p5b .eq. p5b) .and. 
     &    (t_p6b .eq. p6b) .and. (t_h1b .eq. h1b) .and. 
     &    (t_h2b .eq. h2b) .and. (t_h3b .eq. h3b)) THEN

      cc = cc + 1
      call acc_sd_6ds_1(range_h3b,range_h2b,
     &                  range_h1b,range_p6b,
     &                  range_p5b,range_p4b,
     &                  f_t3,f_t1_sort(:,ia6),f_v2_sort(:,ia6),
     &                  istart,istop,jstart,jstop,
     &                  maxp4,maxp5)

      END IF
      if (   (t_p4b .eq. p4b).and.(t_p5b .eq. p5b)
     & .and. (t_p6b .eq. p6b).and.(t_h1b .eq. h2b)
     & .and. (t_h2b .eq. h1b).and.(t_h3b .eq. h3b)) then

      cc = cc + 1
      call acc_sd_6ds_2(range_h3b,range_h2b,range_h1b,range_p6b,
     &                  range_p5b,range_p4b,
     &                  f_t3,f_t1_sort(:,ia6),f_v2_sort(:,ia6),
     &                  istart,istop,jstart,jstop,
     &                  maxp4,maxp5)

      END IF
      if (   (t_p4b .eq. p4b).and.(t_p5b .eq. p5b)
     & .and. (t_p6b .eq. p6b).and.(t_h1b .eq. h2b)
     & .and. (t_h2b .eq. h3b).and.(t_h3b .eq. h1b)) then

      cc = cc + 1
      call acc_sd_6ds_3(range_h3b,range_h2b,range_h1b,range_p6b,
     &                  range_p5b,range_p4b,
     &                  f_t3,f_t1_sort(:,ia6),f_v2_sort(:,ia6),
     &                  istart,istop,jstart,jstop,
     &                  maxp4,maxp5)

      END IF
      if (   (t_p4b .eq. p5b).and.(t_p5b .eq. p4b)
     & .and. (t_p6b .eq. p6b).and.(t_h1b .eq. h1b)
     & .and. (t_h2b .eq. h2b).and.(t_h3b .eq. h3b)) then

      cc = cc + 1
      call acc_sd_6ds_4(range_h3b,range_h2b,range_h1b,range_p6b,
     &                  range_p5b,range_p4b,
     &                  f_t3,f_t1_sort(:,ia6),f_v2_sort(:,ia6),
     &                  istart,istop,jstart,jstop,
     &                  maxp4,maxp5)

      END IF
      if (   (t_p4b .eq. p5b).and.(t_p5b .eq. p4b)
     & .and. (t_p6b .eq. p6b).and.(t_h1b .eq. h2b)
     & .and. (t_h2b .eq. h1b).and.(t_h3b .eq. h3b)) then

      cc = cc + 1
      call acc_sd_6ds_5(range_h3b,range_h2b,range_h1b,range_p6b,
     &                  range_p5b,range_p4b,
     &                  f_t3,f_t1_sort(:,ia6),f_v2_sort(:,ia6),
     &                  istart,istop,jstart,jstop,
     &                  maxp4,maxp5)

      END IF
      if (   (t_p4b .eq. p5b).and.(t_p5b .eq. p4b)
     & .and. (t_p6b .eq. p6b).and.(t_h1b .eq. h2b)
     & .and. (t_h2b .eq. h3b).and.(t_h3b .eq. h1b)) then

      cc = cc + 1
      call acc_sd_6ds_6(range_h3b,range_h2b,range_h1b,range_p6b,
     &                  range_p5b,range_p4b,
     &                  f_t3,f_t1_sort(:,ia6),f_v2_sort(:,ia6),
     &                  istart,istop,jstart,jstop,
     &                  maxp4,maxp5)

      END IF
      if (   (t_p4b .eq. p5b).and.(t_p5b .eq. p6b)
     & .and. (t_p6b .eq. p4b).and.(t_h1b .eq. h1b)
     & .and. (t_h2b .eq. h2b).and.(t_h3b .eq. h3b)) then

      cc = cc + 1
      call acc_sd_6ds_7(range_h3b,range_h2b,range_h1b,range_p6b,
     &                  range_p5b,range_p4b,
     &                  f_t3,f_t1_sort(:,ia6),f_v2_sort(:,ia6),
     &                  istart,istop,jstart,jstop,
     &                  maxp4,maxp5)

      END IF
      if (   (t_p4b .eq. p5b).and.(t_p5b .eq. p6b)
     & .and. (t_p6b .eq. p4b).and.(t_h1b .eq. h2b)
     & .and. (t_h2b .eq. h1b).and.(t_h3b .eq. h3b)) then

      cc = cc + 1
      call acc_sd_6ds_8(range_h3b,range_h2b,range_h1b,range_p6b,
     &                  range_p5b,range_p4b,
     &                  f_t3,f_t1_sort(:,ia6),f_v2_sort(:,ia6),
     &                  istart,istop,jstart,jstop,
     &                  maxp4,maxp5)

      END IF
      if (   (t_p4b .eq. p5b).and.(t_p5b .eq. p6b)
     & .and. (t_p6b .eq. p4b).and.(t_h1b .eq. h2b)
     & .and. (t_h2b .eq. h3b).and.(t_h3b .eq. h1b)) then

      cc = cc + 1
      call acc_sd_6ds_9(range_h3b,range_h2b,range_h1b,range_p6b,
     &                  range_p5b,range_p4b,
     &                  f_t3,f_t1_sort(:,ia6),f_v2_sort(:,ia6),
     &                  istart,istop,jstart,jstop,
     &                  maxp4,maxp5)

      END IF
!
      END IF ! sym 4 & 1 & irrep_t
      END IF ! spin 4 & 1
!
      END IF ! sym
      END IF ! spin x6
      END IF ! .not.restricted ...
!
      END IF ! (p5b.le.p6b).and.(h2b.le.h3b).and.(p4b.ne.0)
!
      end do ! ia6
!      write(6,'(a3,1x,i2,1x,a5,1x,i2)') 'cc=',cc,'comm=',comm
      END

      SUBROUTINE ccsd_t_6dts_l_1_cutensor(f_t3,
     &           d_t2,d_v2,k_t2_offset,k_v2_offset,
     &           t_h1b,t_h2b,t_h3b,t_p4b,t_p5b,t_p6b,
     &           istart,istop,jstart,jstop,maxp4,maxp5,maxp,maxh,
     &           stream)
!     i0 ( p4 p5 p6 h1 h2 h3 )_vt + = -1 * P( 9 ) * Sum ( h7 ) * t ( p4 p5 h1 h7 )_t * v ( h7 p6 h2 h3 )_v
!
! elimination of the transpositions (kowalski,august,2009)
!
      implicit none
#include "global.fh"
#include "mafdecls.fh"
#include "sym.fh"
#include "errquit.fh"
#include "tce.fh"
      double precision, intent(inout) :: f_t3(*)
      integer, intent(in) :: d_t2, d_v2
      integer, intent(in) :: k_t2_offset, k_v2_offset
      integer, intent(in) :: t_p4b, t_p5b, t_p6b, t_h1b, t_h2b, t_h3b
      integer, intent(in) :: istart,istop,jstart,jstop,maxp4,maxp5
      integer, intent(in) :: maxp,maxh
      integer(kind=cuda_stream_kind), intent(inout) :: stream
      integer :: p4b, p5b, p6b, h1b, h2b, h3b, h7b
      integer :: p4b_1, p5b_1, h1b_1, h7b_1, p6b_2, h7b_2, h2b_2, h3b_2
      integer :: dima, dimb
      integer :: a3(9,6)
      integer :: ia6,ja6
      double precision, allocatable :: f_t2_temp(:), f_t2_sort(:)
      double precision, allocatable :: f_v2_sort(:)
      integer :: dimhhpp, dimhhhp
      integer :: alloc_error
      integer :: range_p4b,range_p5b,range_p6b,range_h7b
      integer :: range_h1b,range_h2b,range_h3b
      integer :: spin_p4b,spin_p5b,spin_p6b,spin_h7b
      integer :: spin_h1b,spin_h2b,spin_h3b
      integer :: sym_p4b,sym_p5b,sym_p6b,sym_h7b
      integer :: sym_h1b,sym_h2b,sym_h3b
      integer :: cc,comm

      dimhhhp = maxh*maxh*maxh*maxp
      dimhhpp = maxh*maxh*maxp*maxp

      allocate( f_t2_temp(1:dimhhpp), f_t2_sort(1:dimhhpp), 
     &          f_v2_sort(1:dimhhhp),
     &          stat=alloc_error)
      if (alloc_error.ne.0) then
        call errquit("MA ccsd_t_6dts_l_1_cutensor",
     &               2*dimhhpp+dimhhhp,MA_ERR)
      endif
!
      a3(1,1)=t_p4b
      a3(1,2)=t_p5b
      a3(1,3)=t_p6b
      a3(1,4)=t_h1b
      a3(1,5)=t_h2b
      a3(1,6)=t_h3b
!
      a3(2,1)=t_p4b
      a3(2,2)=t_p5b
      a3(2,3)=t_p6b
      a3(2,4)=t_h2b
      a3(2,5)=t_h1b
      a3(2,6)=t_h3b
!
      a3(3,1)=t_p4b
      a3(3,2)=t_p5b
      a3(3,3)=t_p6b
      a3(3,4)=t_h3b
      a3(3,5)=t_h1b
      a3(3,6)=t_h2b
!
      a3(4,1)=t_p5b
      a3(4,2)=t_p6b
      a3(4,3)=t_p4b
      a3(4,4)=t_h1b
      a3(4,5)=t_h2b
      a3(4,6)=t_h3b
!
      a3(5,1)=t_p5b
      a3(5,2)=t_p6b
      a3(5,3)=t_p4b
      a3(5,4)=t_h2b
      a3(5,5)=t_h1b
      a3(5,6)=t_h3b
!
      a3(6,1)=t_p5b
      a3(6,2)=t_p6b
      a3(6,3)=t_p4b
      a3(6,4)=t_h3b
      a3(6,5)=t_h1b
      a3(6,6)=t_h2b
!
      a3(7,1)=t_p4b
      a3(7,2)=t_p6b
      a3(7,3)=t_p5b
      a3(7,4)=t_h1b
      a3(7,5)=t_h2b
      a3(7,6)=t_h3b
!
      a3(8,1)=t_p4b
      a3(8,2)=t_p6b
      a3(8,3)=t_p5b
      a3(8,4)=t_h2b
      a3(8,5)=t_h1b
      a3(8,6)=t_h3b
!
      a3(9,1)=t_p4b
      a3(9,2)=t_p6b
      a3(9,3)=t_p5b
      a3(9,4)=t_h3b
      a3(9,5)=t_h1b
      a3(9,6)=t_h2b
!
      do ia6=1,8
       if(a3(ia6,1).ne.0) then
       do ja6=ia6+1,9
        if((a3(ia6,1).eq.a3(ja6,1)).and.(a3(ia6,2).eq.a3(ja6,2))
     &   .and.(a3(ia6,3).eq.a3(ja6,3)).and.(a3(ia6,4).eq.a3(ja6,4))
     &   .and.(a3(ia6,5).eq.a3(ja6,5)).and.(a3(ia6,6).eq.a3(ja6,6)))
     &   then
         a3(ja6,1)=0
         a3(ja6,2)=0
         a3(ja6,3)=0
         a3(ja6,4)=0
         a3(ja6,5)=0
         a3(ja6,6)=0
        end if
       enddo 
       end if
      enddo 
!
      DO h7b = 1,noab
          spin_h7b=int_mb(k_spin+h7b-1)
          sym_h7b=int_mb(k_sym+h7b-1)
          range_h7b=int_mb(k_range+h7b-1)

      cc = 0
      comm = 0

      do ia6=1,9
          p4b=a3(ia6,1)
          p5b=a3(ia6,2)
          p6b=a3(ia6,3)
          h1b=a3(ia6,4)
          h2b=a3(ia6,5)
          h3b=a3(ia6,6)

      if((p4b.le.p5b).and.(h2b.le.h3b).and.(p4b.ne.0)) then

          spin_h1b=int_mb(k_spin+h1b-1)
          spin_h2b=int_mb(k_spin+h2b-1)
          spin_h3b=int_mb(k_spin+h3b-1)
          spin_p4b=int_mb(k_spin+p4b-1)
          spin_p5b=int_mb(k_spin+p5b-1)
          spin_p6b=int_mb(k_spin+p6b-1)

      IF ((.not.restricted).or.
     &    (spin_p4b+spin_p5b+spin_p6b+spin_h1b+
     &     spin_h2b+spin_h3b.ne.12) ) THEN
      IF (spin_p4b+spin_p5b+spin_p6b .eq. 
     &    spin_h1b+spin_h2b+spin_h3b) THEN

          sym_h1b=int_mb(k_sym+h1b-1)
          sym_h2b=int_mb(k_sym+h2b-1)
          sym_h3b=int_mb(k_sym+h3b-1)
          sym_p4b=int_mb(k_sym+p4b-1)
          sym_p5b=int_mb(k_sym+p5b-1)
          sym_p6b=int_mb(k_sym+p6b-1)

      IF (ieor(sym_p4b,ieor(sym_p5b,
     &    ieor(sym_p6b,ieor(sym_h1b,
     &    ieor(sym_h2b,sym_h3b))))) 
     &   .eq. ieor(irrep_v,irrep_t)) THEN

          range_h1b=int_mb(k_range+h1b-1)
          range_h2b=int_mb(k_range+h2b-1)
          range_h3b=int_mb(k_range+h3b-1)
          range_p4b=int_mb(k_range+p4b-1)
          range_p5b=int_mb(k_range+p5b-1)
          range_p6b=int_mb(k_range+p6b-1)

      !DO h7b = 1,noab
      !    spin_h7b=int_mb(k_spin+h7b-1)
      !    sym_h7b=int_mb(k_sym+h7b-1)
      !    range_h7b=int_mb(k_range+h7b-1)

      IF (spin_p4b+spin_p5b .eq. spin_h1b+spin_h7b) THEN
      IF (ieor(sym_p4b,ieor(sym_p5b,ieor(sym_h1b,sym_h7b))) 
     &    .eq. irrep_t) THEN
      CALL TCE_RESTRICTED_4(p4b,p5b,h1b,h7b,p4b_1,p5b_1,h1b_1,h7b_1)
      CALL TCE_RESTRICTED_4(p6b,h7b,h2b,h3b,p6b_2,h7b_2,h2b_2,h3b_2)
      dima = range_h7b * range_p4b * range_p5b * range_h1b
      dimb = range_h7b * range_p6b * range_h2b * range_h3b
      IF ((dima .gt. 0) .and. (dimb .gt. 0)) THEN
      IF ((h7b .lt. h1b)) THEN
      CALL GET_HASH_BLOCK(d_t2,f_t2_temp,dima,int_mb(k_t2_offset),
     &      (h1b_1 - 1 + noab * (h7b_1 - 1 + noab * 
     &       (p5b_1 - noab - 1 + nvab * (p4b_1 - noab - 1)))))
      comm = comm + 1
      CALL TCE_SORT_4(f_t2_temp,f_t2_sort,
     &                range_p4b,range_p5b,range_h7b,range_h1b,
     &                4,2,1,3,-1.0d0)
      !END IF
      else
      !IF ((h1b .le. h7b)) THEN
      CALL GET_HASH_BLOCK(d_t2,f_t2_temp,dima,int_mb(k_t2_offset),
     &      (h7b_1 - 1 + noab * (h1b_1 - 1 + noab * 
     &       (p5b_1 - noab - 1 + nvab * (p4b_1 - noab - 1)))))
      comm = comm + 1
      CALL TCE_SORT_4(f_t2_temp,f_t2_sort,
     &                range_p4b,range_p5b,range_h1b,range_h7b,
     &                3,2,1,4,1.0d0)
      END IF
      IF ((h7b .le. p6b)) THEN
      if(.not.intorb) then
      CALL GET_HASH_BLOCK(d_v2,f_v2_sort,dimb,int_mb(k_v2_offset),
     &     (h3b_2 - 1 + (noab+nvab) * (h2b_2 - 1 + (noab+nvab) * 
     &     (p6b_2 - 1 + (noab+nvab) * (h7b_2 - 1)))))
      comm = comm + 1
      else
      CALL GET_HASH_BLOCK_I(d_v2,f_v2_sort,dimb,int_mb(k_v2_offset),
     &     (h3b_2 - 1 + (noab+nvab) * (h2b_2 - 1 + (noab+nvab) *
     &     (p6b_2 - 1 + (noab+nvab) * (h7b_2 - 1)))),
     &     h3b_2,h2b_2,p6b_2,h7b_2)
      comm = comm + 1
      end if
      END IF
!
!  no need to zero d_c - it has already been done in ccsd_t
!  k_c_sort is not used here
!
! contributions are calculated here
!
      if (   (t_p4b .eq. p4b).and.(t_p5b .eq. p5b)
     & .and. (t_p6b .eq. p6b).and.(t_h1b .eq. h1b)
     & .and. (t_h2b .eq. h2b).and.(t_h3b .eq. h3b)) then

      cc = cc + 1
      call acc_sd_tx1_1(range_h3b,range_h2b,range_h1b,range_p6b,
     &               range_p5b,range_p4b,range_h7b,
     &               f_t3,f_t2_sort,f_v2_sort,
     &                  istart,istop,jstart,jstop,maxp4,maxp5)

      END IF
      if (   (t_p4b .eq. p4b).and.(t_p5b .eq. p5b)
     & .and. (t_p6b .eq. p6b).and.(t_h1b .eq. h2b)
     & .and. (t_h2b .eq. h1b).and.(t_h3b .eq. h3b)) then

      cc = cc + 1
      call acc_sd_tx1_2(range_h3b,range_h2b,range_h1b,range_p6b,
     &               range_p5b,range_p4b,range_h7b,
     &               f_t3,f_t2_sort,f_v2_sort,
     &                  istart,istop,jstart,jstop,maxp4,maxp5)

      END IF
      if (   (t_p4b .eq. p4b).and.(t_p5b .eq. p5b)
     & .and. (t_p6b .eq. p6b).and.(t_h1b .eq. h2b)
     & .and. (t_h2b .eq. h3b).and.(t_h3b .eq. h1b)) then

      cc = cc + 1
      call acc_sd_tx1_3(range_h3b,range_h2b,range_h1b,range_p6b,
     &               range_p5b,range_p4b,range_h7b,
     &               f_t3,f_t2_sort,f_v2_sort,
     &                  istart,istop,jstart,jstop,maxp4,maxp5)

      END IF
      if (   (t_p4b .eq. p6b).and.(t_p5b .eq. p4b)
     & .and. (t_p6b .eq. p5b).and.(t_h1b .eq. h1b)
     & .and. (t_h2b .eq. h2b).and.(t_h3b .eq. h3b)) then

      cc = cc + 1
      call acc_sd_tx1_4(range_h3b,range_h2b,range_h1b,range_p6b,
     &               range_p5b,range_p4b,range_h7b,
     &               f_t3,f_t2_sort,f_v2_sort,
     &                  istart,istop,jstart,jstop,maxp4,maxp5)

      END IF
      if (   (t_p4b .eq. p6b).and.(t_p5b .eq. p4b)
     & .and. (t_p6b .eq. p5b).and.(t_h1b .eq. h2b)
     & .and. (t_h2b .eq. h1b).and.(t_h3b .eq. h3b)) then

      cc = cc + 1
      call acc_sd_tx1_5(range_h3b,range_h2b,range_h1b,range_p6b,
     &               range_p5b,range_p4b,range_h7b,
     &               f_t3,f_t2_sort,f_v2_sort,
     &                  istart,istop,jstart,jstop,maxp4,maxp5)

      END IF
      if (   (t_p4b .eq. p6b).and.(t_p5b .eq. p4b)
     & .and. (t_p6b .eq. p5b).and.(t_h1b .eq. h2b)
     & .and. (t_h2b .eq. h3b).and.(t_h3b .eq. h1b)) then

      cc = cc + 1
      call acc_sd_tx1_6(range_h3b,range_h2b,range_h1b,range_p6b,
     &               range_p5b,range_p4b,range_h7b,
     &               f_t3,f_t2_sort,f_v2_sort,
     &                  istart,istop,jstart,jstop,maxp4,maxp5)

      END IF
      if (   (t_p4b .eq. p4b).and.(t_p5b .eq. p6b)
     & .and. (t_p6b .eq. p5b).and.(t_h1b .eq. h1b)
     & .and. (t_h2b .eq. h2b).and.(t_h3b .eq. h3b)) then

      cc = cc + 1
      call acc_sd_tx1_7(range_h3b,range_h2b,range_h1b,range_p6b,
     &               range_p5b,range_p4b,range_h7b,
     &               f_t3,f_t2_sort,f_v2_sort,
     &                  istart,istop,jstart,jstop,maxp4,maxp5)

      END IF
      if (   (t_p4b .eq. p4b).and.(t_p5b .eq. p6b)
     & .and. (t_p6b .eq. p5b).and.(t_h1b .eq. h2b)
     & .and. (t_h2b .eq. h1b).and.(t_h3b .eq. h3b)) then

      cc = cc + 1
      call acc_sd_tx1_8(range_h3b,range_h2b,range_h1b,range_p6b,
     &               range_p5b,range_p4b,range_h7b,
     &               f_t3,f_t2_sort,f_v2_sort,
     &                  istart,istop,jstart,jstop,maxp4,maxp5)

      END IF
      if (   (t_p4b .eq. p4b).and.(t_p5b .eq. p6b)
     & .and. (t_p6b .eq. p5b).and.(t_h1b .eq. h2b)
     & .and. (t_h2b .eq. h3b).and.(t_h3b .eq. h1b)) then

      cc = cc + 1
      call acc_sd_tx1_9(range_h3b,range_h2b,range_h1b,range_p6b,
     &               range_p5b,range_p4b,range_h7b,
     &               f_t3,f_t2_sort,f_v2_sort,
     &                  istart,istop,jstart,jstop,maxp4,maxp5)

      END IF
!
      END IF
      END IF
      END IF
      !END DO !h7b loop
!
      END IF
      END IF
      END IF
!
      END IF
      END DO ! ia6
!      write(6,'(a4,i2,1x,a3,i2,1x,a5,i2)') 
!     &        'h7b=',h7b,'cc=',cc,'comm=',comm
      END DO !h7b loop
      END

      SUBROUTINE ccsd_t_6dts_l_2_cutensor(f_t3,
     &           d_t2,d_v2,k_t2_offset,k_v2_offset,
     &           t_h1b,t_h2b,t_h3b,t_p4b,t_p5b,t_p6b,
     &           istart,istop,jstart,jstop,maxp4,maxp5,maxp,maxh,
     &           stream)
!     i0 ( p4 p5 p6 h1 h2 h3 )_vt + = -1 * P( 9 ) * Sum ( p7 ) * t ( p4 p7 h1 h2 )_t * v ( p5 p6 h3 p7 )_v
!
! elimination of the transpositions (kowalski,august,2009)
!
      implicit none
#include "global.fh"
#include "mafdecls.fh"
#include "sym.fh"
#include "errquit.fh"
#include "tce.fh"
      double precision, intent(inout) :: f_t3(*)
      integer, intent(in) :: d_t2, d_v2
      integer, intent(in) :: k_t2_offset, k_v2_offset
      integer, intent(in) :: t_p4b, t_p5b, t_p6b, t_h1b, t_h2b, t_h3b
      integer, intent(in) :: istart,istop,jstart,jstop,maxp4,maxp5
      integer, intent(in) :: maxp,maxh
      integer(kind=cuda_stream_kind), intent(inout) :: stream
      integer :: p4b, p5b, p6b, h1b, h2b, h3b, p7b
      integer :: p4b_1, p7b_1, h1b_1, h2b_1, p5b_2, p6b_2, h3b_2, p7b_2
      integer :: dima, dimb
      integer :: a3(9,6)
      integer :: ia6,ja6
      double precision, allocatable :: f_t2_temp(:), f_t2_sort(:)
      double precision, allocatable :: f_v2_sort(:)
      integer :: dimhhpp, dimhppp
      integer :: alloc_error
      integer :: range_p4b,range_p5b,range_p6b,range_p7b
      integer :: range_h1b,range_h2b,range_h3b
      integer :: spin_p4b,spin_p5b,spin_p6b,spin_p7b
      integer :: spin_h1b,spin_h2b,spin_h3b
      integer :: sym_p4b,sym_p5b,sym_p6b,sym_p7b
      integer :: sym_h1b,sym_h2b,sym_h3b
      integer :: cc,comm

      dimhhpp = maxh*maxh*maxp*maxp
      dimhppp = maxh*maxp*maxp*maxp

      allocate( f_t2_temp(1:dimhhpp), f_t2_sort(1:dimhhpp), 
     &          f_v2_sort(1:dimhppp),
     &          stat=alloc_error)
      if (alloc_error.ne.0) then
        call errquit("MA ccsd_t_6dts_l_2_cutensor",
     &               2*dimhppp+dimhppp,MA_ERR)
      endif
!
      a3(1,1)=t_p4b
      a3(1,2)=t_p5b
      a3(1,3)=t_p6b
      a3(1,4)=t_h1b
      a3(1,5)=t_h2b
      a3(1,6)=t_h3b
!
      a3(2,1)=t_p4b
      a3(2,2)=t_p5b
      a3(2,3)=t_p6b
      a3(2,4)=t_h2b
      a3(2,5)=t_h3b
      a3(2,6)=t_h1b
!
      a3(3,1)=t_p4b
      a3(3,2)=t_p5b
      a3(3,3)=t_p6b
      a3(3,4)=t_h1b
      a3(3,5)=t_h3b
      a3(3,6)=t_h2b
!
      a3(4,1)=t_p5b
      a3(4,2)=t_p4b
      a3(4,3)=t_p6b
      a3(4,4)=t_h1b
      a3(4,5)=t_h2b
      a3(4,6)=t_h3b
!
      a3(5,1)=t_p5b
      a3(5,2)=t_p4b
      a3(5,3)=t_p6b
      a3(5,4)=t_h2b
      a3(5,5)=t_h3b
      a3(5,6)=t_h1b
!
      a3(6,1)=t_p5b
      a3(6,2)=t_p4b
      a3(6,3)=t_p6b
      a3(6,4)=t_h1b
      a3(6,5)=t_h3b
      a3(6,6)=t_h2b
!
      a3(7,1)=t_p6b
      a3(7,2)=t_p4b
      a3(7,3)=t_p5b
      a3(7,4)=t_h1b
      a3(7,5)=t_h2b
      a3(7,6)=t_h3b
!
      a3(8,1)=t_p6b
      a3(8,2)=t_p4b
      a3(8,3)=t_p5b
      a3(8,4)=t_h2b
      a3(8,5)=t_h3b
      a3(8,6)=t_h1b
!
      a3(9,1)=t_p6b
      a3(9,2)=t_p4b
      a3(9,3)=t_p5b
      a3(9,4)=t_h1b
      a3(9,5)=t_h3b
      a3(9,6)=t_h2b
!
      do ia6=1,8
       if(a3(ia6,1).ne.0) then
       do ja6=ia6+1,9
        if((a3(ia6,1).eq.a3(ja6,1)).and.(a3(ia6,2).eq.a3(ja6,2))
     &   .and.(a3(ia6,3).eq.a3(ja6,3)).and.(a3(ia6,4).eq.a3(ja6,4))
     &   .and.(a3(ia6,5).eq.a3(ja6,5)).and.(a3(ia6,6).eq.a3(ja6,6)))
     &   then
         a3(ja6,1)=0
         a3(ja6,2)=0
         a3(ja6,3)=0
         a3(ja6,4)=0
         a3(ja6,5)=0
         a3(ja6,6)=0
        end if
       enddo 
       end if
      enddo 
!
      DO p7b = noab+1,noab+nvab
          spin_p7b=int_mb(k_spin+p7b-1)
          sym_p7b=int_mb(k_sym+p7b-1)
          range_p7b=int_mb(k_range+p7b-1)

      cc = 0
      comm = 0

      do ia6=1,9
          p4b=a3(ia6,1)
          p5b=a3(ia6,2)
          p6b=a3(ia6,3)
          h1b=a3(ia6,4)
          h2b=a3(ia6,5)
          h3b=a3(ia6,6)

      if((p5b.le.p6b).and.(h1b.le.h2b).and.(p4b.ne.0)) then

          spin_h1b=int_mb(k_spin+h1b-1)
          spin_h2b=int_mb(k_spin+h2b-1)
          spin_h3b=int_mb(k_spin+h3b-1)
          spin_p4b=int_mb(k_spin+p4b-1)
          spin_p5b=int_mb(k_spin+p5b-1)
          spin_p6b=int_mb(k_spin+p6b-1)

      IF ((.not.restricted).or.
     &    (spin_p4b+spin_p5b+spin_p6b+spin_h1b+
     &     spin_h2b+spin_h3b.ne.12)) THEN
      IF (spin_p4b+spin_p5b+spin_p6b .eq. 
     &    spin_h1b+spin_h2b+spin_h3b) THEN

          sym_h1b=int_mb(k_sym+h1b-1)
          sym_h2b=int_mb(k_sym+h2b-1)
          sym_h3b=int_mb(k_sym+h3b-1)
          sym_p4b=int_mb(k_sym+p4b-1)
          sym_p5b=int_mb(k_sym+p5b-1)
          sym_p6b=int_mb(k_sym+p6b-1)

      IF (ieor(sym_p4b,ieor(sym_p5b,ieor(sym_p6b,ieor(sym_h1b,
     &    ieor(sym_h2b,sym_h3b))))) .eq. ieor(irrep_v,irrep_t)) THEN

          range_h1b=int_mb(k_range+h1b-1)
          range_h2b=int_mb(k_range+h2b-1)
          range_h3b=int_mb(k_range+h3b-1)
          range_p4b=int_mb(k_range+p4b-1)
          range_p5b=int_mb(k_range+p5b-1)
          range_p6b=int_mb(k_range+p6b-1)

      ! OLD STRUCTURE
      !DO p7b = noab+1,noab+nvab
      !    spin_p7b=int_mb(k_spin+p7b-1)
      !    sym_p7b=int_mb(k_sym+p7b-1)
      !    range_p7b=int_mb(k_range+p7b-1)

      IF (spin_p4b+spin_p7b .eq. spin_h1b+spin_h2b) THEN
      IF (ieor(sym_p4b,ieor(sym_p7b,ieor(sym_h1b,sym_h2b))) 
     &    .eq. irrep_t) THEN
      CALL TCE_RESTRICTED_4(p4b,p7b,h1b,h2b,p4b_1,p7b_1,h1b_1,h2b_1)
      CALL TCE_RESTRICTED_4(p5b,p6b,h3b,p7b,p5b_2,p6b_2,h3b_2,p7b_2)
      dima = range_p7b * range_p4b * range_h1b * range_h2b
      dimb = range_p7b * range_p5b * range_p6b * range_h3b
      IF ((dima .gt. 0) .and. (dimb .gt. 0)) THEN
      IF ((p7b .lt. p4b)) THEN
      CALL GET_HASH_BLOCK(d_t2,f_t2_temp,dima,int_mb(k_t2_offset),
     &     (h2b_1 - 1 + noab * (h1b_1 - 1 + noab * 
     &     (p4b_1 - noab - 1 + nvab * (p7b_1 - noab - 1)))))
      comm = comm + 1
      CALL TCE_SORT_4(f_t2_temp,f_t2_sort,
     &                range_p7b,range_p4b,range_h1b,range_h2b,
     &                4,3,2,1,-1.0d0)
      !END IF
      else
      !IF ((p4b .le. p7b)) THEN
      CALL GET_HASH_BLOCK(d_t2,f_t2_temp,dima,int_mb(k_t2_offset),
     &     (h2b_1 - 1 + noab * (h1b_1 - 1 + noab * 
     &     (p7b_1 - noab - 1 + nvab * (p4b_1 - noab - 1)))))
      comm = comm + 1
      CALL TCE_SORT_4(f_t2_temp,f_t2_sort,
     &                range_p4b,range_p7b,range_h1b,range_h2b,
     &                4,3,1,2,1.0d0)
      END IF
      IF ((h3b .le. p7b)) THEN
      if(.not.intorb) then
      CALL GET_HASH_BLOCK(d_v2,f_v2_sort,dimb,int_mb(k_v2_offset),
     &     (p7b_2 - 1 + (noab+nvab) * (h3b_2 - 1 + (noab+nvab) 
     &   * (p6b_2 - 1 + (noab+nvab) * (p5b_2 - 1)))))
      comm = comm + 1
      else
      CALL GET_HASH_BLOCK_I(d_v2,f_v2_sort,dimb,int_mb(k_v2_offset),
     &     (p7b_2 - 1 + (noab+nvab) * (h3b_2 - 1 + (noab+nvab) 
     &   * (p6b_2 - 1 + (noab+nvab) * (p5b_2 - 1)))),
     &     p7b_2,h3b_2,p6b_2,p5b_2)
      comm = comm + 1
      end if
      END IF
!
!  no need to zero d_c - it has already been done in ccsd_t
!  k_c_sort is not used here
!
! contributions are calculated here
!
      if (   (t_p4b .eq. p4b).and.(t_p5b .eq. p5b)
     & .and. (t_p6b .eq. p6b).and.(t_h1b .eq. h1b)
     & .and. (t_h2b .eq. h2b).and.(t_h3b .eq. h3b)) then

      cc = cc + 1
      call acc_sd_ty1_1(range_h3b,range_h2b,range_h1b,range_p6b,
     &                  range_p5b,range_p4b,range_p7b,
     &                  f_t3,f_t2_sort,f_v2_sort,
     &                  istart,istop,jstart,jstop,maxp4,maxp5)

      END IF
      if (   (t_p4b .eq. p4b).and.(t_p5b .eq. p5b)
     & .and. (t_p6b .eq. p6b).and.(t_h1b .eq. h3b)
     & .and. (t_h2b .eq. h1b).and.(t_h3b .eq. h2b)) then

      cc = cc + 1
      call acc_sd_ty1_2(range_h3b,range_h2b,range_h1b,range_p6b,
     &                  range_p5b,range_p4b,range_p7b,
     &                  f_t3,f_t2_sort,f_v2_sort,
     &                  istart,istop,jstart,jstop,maxp4,maxp5)

      END IF
      if (   (t_p4b .eq. p4b).and.(t_p5b .eq. p5b)
     & .and. (t_p6b .eq. p6b).and.(t_h1b .eq. h1b)
     & .and. (t_h2b .eq. h3b).and.(t_h3b .eq. h2b)) then

      cc = cc + 1
      call acc_sd_ty1_3(range_h3b,range_h2b,range_h1b,range_p6b,
     &                  range_p5b,range_p4b,range_p7b,
     &                  f_t3,f_t2_sort,f_v2_sort,
     &                  istart,istop,jstart,jstop,maxp4,maxp5)

      END IF
      if (   (t_p4b .eq. p5b).and.(t_p5b .eq. p4b)
     & .and. (t_p6b .eq. p6b).and.(t_h1b .eq. h1b)
     & .and. (t_h2b .eq. h2b).and.(t_h3b .eq. h3b)) then

      cc = cc + 1
      call acc_sd_ty1_4(range_h3b,range_h2b,range_h1b,range_p6b,
     &                  range_p5b,range_p4b, range_p7b,
     &                  f_t3,f_t2_sort,f_v2_sort,
     &                  istart,istop,jstart,jstop,maxp4,maxp5)

      END IF
      if (   (t_p4b .eq. p5b).and.(t_p5b .eq. p4b)
     & .and. (t_p6b .eq. p6b).and.(t_h1b .eq. h3b)
     & .and. (t_h2b .eq. h1b).and.(t_h3b .eq. h2b)) then

      cc = cc + 1
      call acc_sd_ty1_5(range_h3b,range_h2b,range_h1b,range_p6b,
     &                  range_p5b,range_p4b,range_p7b,
     &                  f_t3,f_t2_sort,f_v2_sort,
     &                  istart,istop,jstart,jstop,maxp4,maxp5)

      END IF
      if (   (t_p4b .eq. p5b).and.(t_p5b .eq. p4b)
     & .and. (t_p6b .eq. p6b).and.(t_h1b .eq. h1b)
     & .and. (t_h2b .eq. h3b).and.(t_h3b .eq. h2b)) then

      cc = cc + 1
      call acc_sd_ty1_6(range_h3b,range_h2b,range_h1b,range_p6b,
     &                  range_p5b,range_p4b,range_p7b,
     &                  f_t3,f_t2_sort,f_v2_sort,
     &                  istart,istop,jstart,jstop,maxp4,maxp5)

      END IF
      if (   (t_p4b .eq. p5b).and.(t_p5b .eq. p6b)
     & .and. (t_p6b .eq. p4b).and.(t_h1b .eq. h1b)
     & .and. (t_h2b .eq. h2b).and.(t_h3b .eq. h3b)) then

      cc = cc + 1
      call acc_sd_ty1_7(range_h3b,range_h2b,range_h1b,range_p6b,
     &                  range_p5b,range_p4b,range_p7b,
     &                  f_t3,f_t2_sort,f_v2_sort,
     &                  istart,istop,jstart,jstop,maxp4,maxp5)

      END IF
      if (   (t_p4b .eq. p5b).and.(t_p5b .eq. p6b)
     & .and. (t_p6b .eq. p4b).and.(t_h1b .eq. h3b)
     & .and. (t_h2b .eq. h1b).and.(t_h3b .eq. h2b)) then

      cc = cc + 1
      call acc_sd_ty1_8(range_h3b,range_h2b,range_h1b,range_p6b,
     &                  range_p5b,range_p4b,range_p7b,
     &                  f_t3,f_t2_sort,f_v2_sort,
     &                  istart,istop,jstart,jstop,maxp4,maxp5)

      END IF
      if (   (t_p4b .eq. p5b).and.(t_p5b .eq. p6b)
     & .and. (t_p6b .eq. p4b).and.(t_h1b .eq. h1b)
     & .and. (t_h2b .eq. h3b).and.(t_h3b .eq. h2b)) then

      cc = cc + 1
      call acc_sd_ty1_9(range_h3b,range_h2b,range_h1b,range_p6b,
     &                  range_p5b,range_p4b,range_p7b,
     &                  f_t3,f_t2_sort,f_v2_sort,
     &                  istart,istop,jstart,jstop,maxp4,maxp5)

      END IF

      END IF
      END IF
      END IF
      !END DO ! p7b loop
!
      END IF
      END IF
      END IF
!
      END IF
      END DO ! ia6
!      write(6,'(a4,i2,1x,a3,i2,1x,a5,i2)') 
!     &        'p7b=',p7b,'cc=',cc,'comm=',comm
      END DO ! p7b loop
      END

      subroutine acc_sd_6ds_1(h3d,h2d,h1d,p6d,p5d,p4d,
     &               triplesx,t1sub,v2sub,
     &                  istart,istop,jstart,jstop,maxp4,maxp5)
      implicit none
      integer, intent(in) :: istart,istop,jstart,jstop,maxp4,maxp5
      integer, intent(in) :: h3d,h2d,h1d,p6d,p5d,p4d
      integer :: h3h2,h1,p6,p5,p4
      double precision :: triplesx(h3d*h2d,h1d,p6d,maxp5,maxp4)
      double precision, intent(in) :: t1sub(p4d,h1d)
      double precision, intent(in) :: v2sub(h3d*h2d,p6d,p5d)
!$acc parallel loop collapse(4)
      do p4=istart,istop
      do p5=jstart,jstop
      do p6=1,p6d
      do h1=1,h1d
      do h3h2=1,h2d*h3d
       triplesx(h3h2,h1,p6,p5-jstart+1,p4-istart+1)=
     &     triplesx(h3h2,h1,p6,p5-jstart+1,p4-istart+1)
     &   + t1sub(p4,h1)*v2sub(h3h2,p6,p5)
      enddo
      enddo
      enddo
      enddo
      enddo
      end
!
      subroutine acc_sd_6ds_2(h3d,h2d,h1d,p6d,p5d,p4d,
     &               triplesx,t1sub,v2sub,
     &                  istart,istop,jstart,jstop,maxp4,maxp5)
      implicit none
      integer, intent(in) :: istart,istop,jstart,jstop,maxp4,maxp5
      integer, intent(in) :: h3d,h2d,h1d,p6d,p5d,p4d
      integer :: h3,h2,h1,p6,p5,p4
      double precision :: triplesx(h3d,h1d,h2d,p6d,maxp5,maxp4)
      double precision, intent(in) :: t1sub(p4d,h1d)
      double precision, intent(in) :: v2sub(h3d,h2d,p6d,p5d)
!$acc parallel loop collapse(4)
      do p4=istart,istop
      do p5=jstart,jstop
      do p6=1,p6d
      do h2=1,h2d
      do h1=1,h1d
      do h3=1,h3d
       triplesx(h3,h1,h2,p6,p5-jstart+1,p4-istart+1)=
     &     triplesx(h3,h1,h2,p6,p5-jstart+1,p4-istart+1)
     &   - t1sub(p4,h1)*v2sub(h3,h2,p6,p5)
      enddo
      enddo
      enddo
      enddo
      enddo
      enddo
      return
      end
!
      subroutine acc_sd_6ds_3(h3d,h2d,h1d,p6d,p5d,p4d,
     &               triplesx,t1sub,v2sub,
     &                  istart,istop,jstart,jstop,maxp4,maxp5)
      implicit none
      integer, intent(in) :: istart,istop,jstart,jstop,maxp4,maxp5
      integer, intent(in) :: h3d,h2d,h1d,p6d,p5d,p4d
      integer :: h3,h2,h1,p6,p5,p4
      double precision :: triplesx(h1d,h3d,h2d,p6d,maxp5,maxp4)
      double precision, intent(in) :: t1sub(p4d,h1d)
      double precision, intent(in) :: v2sub(h3d,h2d,p6d,p5d)
!$acc parallel loop collapse(4)
      do p4=istart,istop
      do p5=jstart,jstop
      do p6=1,p6d
      do h2=1,h2d
      do h3=1,h3d
      do h1=1,h1d
       triplesx(h1,h3,h2,p6,p5-jstart+1,p4-istart+1)=
     &     triplesx(h1,h3,h2,p6,p5-jstart+1,p4-istart+1)
     &   + t1sub(p4,h1)*v2sub(h3,h2,p6,p5)
      enddo
      enddo
      enddo
      enddo
      enddo
      enddo
      return
      end
!
      subroutine acc_sd_6ds_4(h3d,h2d,h1d,p6d,p5d,p4d,
     &               triplesx,t1sub,v2sub,
     &                  istart,istop,jstart,jstop,maxp4,maxp5)
      implicit none
      integer, intent(in) :: istart,istop,jstart,jstop,maxp4,maxp5
      integer, intent(in) :: h3d,h2d,h1d,p6d,p5d,p4d
      integer :: h3h2,h1,p6,p5,p4
      double precision :: triplesx(h3d*h2d,h1d,p6d,maxp5,maxp4)
      double precision, intent(in) :: t1sub(p4d,h1d)
      double precision, intent(in) :: v2sub(h3d*h2d,p6d,p5d)
!$acc parallel loop collapse(4)
      do p5=istart,istop
      do p4=jstart,jstop
      do p6=1,p6d
      do h1=1,h1d
      do h3h2=1,h3d*h2d
       triplesx(h3h2,h1,p6,p4-jstart+1,p5-istart+1)=
     &     triplesx(h3h2,h1,p6,p4-jstart+1,p5-istart+1)
     &   - t1sub(p4,h1)*v2sub(h3h2,p6,p5)
      enddo
      enddo
      enddo
      enddo
      enddo
      return
      end
!
      subroutine acc_sd_6ds_5(h3d,h2d,h1d,p6d,p5d,p4d,
     &               triplesx,t1sub,v2sub,
     &                  istart,istop,jstart,jstop,maxp4,maxp5)
      implicit none
      integer, intent(in) :: istart,istop,jstart,jstop,maxp4,maxp5
      integer, intent(in) :: h3d,h2d,h1d,p6d,p5d,p4d
      integer :: h3,h2,h1,p6,p5,p4
      double precision :: triplesx(h3d,h1d,h2d,p6d,maxp5,maxp4)
      double precision, intent(in) :: t1sub(p4d,h1d)
      double precision, intent(in) :: v2sub(h3d,h2d,p6d,p5d)
!$acc parallel loop collapse(4)
      do p5=istart,istop
      do p4=jstart,jstop
      do p6=1,p6d
      do h2=1,h2d
      do h1=1,h1d
      do h3=1,h3d
       triplesx(h3,h1,h2,p6,p4-jstart+1,p5-istart+1)=
     &     triplesx(h3,h1,h2,p6,p4-jstart+1,p5-istart+1)
     &   + t1sub(p4,h1)*v2sub(h3,h2,p6,p5)
      enddo
      enddo
      enddo
      enddo
      enddo
      enddo
      return
      end
!
      subroutine acc_sd_6ds_6(h3d,h2d,h1d,p6d,p5d,p4d,
     &               triplesx,t1sub,v2sub,
     &                  istart,istop,jstart,jstop,maxp4,maxp5)
      implicit none
      integer, intent(in) :: istart,istop,jstart,jstop,maxp4,maxp5
      integer, intent(in) :: h3d,h2d,h1d,p6d,p5d,p4d
      integer :: h3,h2,h1,p6,p5,p4
      double precision :: triplesx(h1d,h3d,h2d,p6d,maxp5,maxp4)
      double precision, intent(in) :: t1sub(p4d,h1d)
      double precision, intent(in) :: v2sub(h3d,h2d,p6d,p5d)
!$acc parallel loop collapse(4)
      do p5=istart,istop
      do p4=jstart,jstop
      do p6=1,p6d
      do h2=1,h2d
      do h3=1,h3d
      do h1=1,h1d
       triplesx(h1,h3,h2,p6,p4-jstart+1,p5-istart+1)=
     &     triplesx(h1,h3,h2,p6,p4-jstart+1,p5-istart+1)
     &   - t1sub(p4,h1)*v2sub(h3,h2,p6,p5)
      enddo
      enddo
      enddo
      enddo
      enddo
      enddo
      return
      end
!
      subroutine acc_sd_6ds_7(h3d,h2d,h1d,p6d,p5d,p4d,
     &               triplesx,t1sub,v2sub,
     &                  istart,istop,jstart,jstop,maxp4,maxp5)
      implicit none
      integer, intent(in) :: istart,istop,jstart,jstop,maxp4,maxp5
      integer, intent(in) :: h3d,h2d,h1d,p6d,p5d,p4d
      integer :: h3h2,h1,p6,p5,p4
      double precision :: triplesx(h3d*h2d,h1d,p4d,maxp5,maxp4)
      double precision, intent(in) :: t1sub(p4d,h1d)
      double precision, intent(in) :: v2sub(h3d*h2d,p6d,p5d)
!$acc parallel loop collapse(4)
      do p5=istart,istop
      do p6=jstart,jstop
      do p4=1,p4d
      do h1=1,h1d
      do h3h2=1,h3d*h2d
       triplesx(h3h2,h1,p4,p6-jstart+1,p5-istart+1)=
     &     triplesx(h3h2,h1,p4,p6-jstart+1,p5-istart+1)
     &   + t1sub(p4,h1)*v2sub(h3h2,p6,p5)
      enddo
      enddo
      enddo
      enddo
      enddo
      return
      end
!
      subroutine acc_sd_6ds_8(h3d,h2d,h1d,p6d,p5d,p4d,
     &               triplesx,t1sub,v2sub,
     &                  istart,istop,jstart,jstop,maxp4,maxp5)
      implicit none
      integer, intent(in) :: istart,istop,jstart,jstop,maxp4,maxp5
      integer, intent(in) :: h3d,h2d,h1d,p6d,p5d,p4d
      integer :: h3,h2,h1,p6,p5,p4
      double precision :: triplesx(h3d,h1d,h2d,p4d,maxp5,maxp4)
      double precision, intent(in) :: t1sub(p4d,h1d)
      double precision, intent(in) :: v2sub(h3d,h2d,p6d,p5d)
!$acc parallel loop collapse(4)
      do p5=istart,istop
      do p6=jstart,jstop
      do p4=1,p4d
      do h2=1,h2d
      do h1=1,h1d
      do h3=1,h3d
       triplesx(h3,h1,h2,p4,p6-jstart+1,p5-istart+1)=
     &     triplesx(h3,h1,h2,p4,p6-jstart+1,p5-istart+1)
     &   - t1sub(p4,h1)*v2sub(h3,h2,p6,p5)
      enddo
      enddo
      enddo
      enddo
      enddo
      enddo
      return
      end
!
      subroutine acc_sd_6ds_9(h3d,h2d,h1d,p6d,p5d,p4d,
     &               triplesx,t1sub,v2sub,
     &                  istart,istop,jstart,jstop,maxp4,maxp5)
      implicit none
      integer, intent(in) :: istart,istop,jstart,jstop,maxp4,maxp5
      integer, intent(in) :: h3d,h2d,h1d,p6d,p5d,p4d
      integer :: h3,h2,h1,p6,p5,p4
      double precision :: triplesx(h1d,h3d,h2d,p4d,maxp5,maxp4)
      double precision, intent(in) :: t1sub(p4d,h1d)
      double precision, intent(in) :: v2sub(h3d,h2d,p6d,p5d)
!$acc parallel loop collapse(4)
      do p5=istart,istop
      do p6=jstart,jstop
      do p4=1,p4d
      do h2=1,h2d
      do h3=1,h3d
      do h1=1,h1d
       triplesx(h1,h3,h2,p4,p6-jstart+1,p5-istart+1)=
     &     triplesx(h1,h3,h2,p4,p6-jstart+1,p5-istart+1)
     &   + t1sub(p4,h1)*v2sub(h3,h2,p6,p5)
      enddo
      enddo
      enddo
      enddo
      enddo
      enddo
      return
      end



      subroutine acc_sd_tx1_1(h3d,h2d,h1d,p6d,p5d,p4d,
     &               h7d,
     &               triplesx,t2sub,v2sub,
     &                  istart,istop,jstart,jstop,maxp4,maxp5)
      implicit none
      integer, intent(in) :: istart,istop,jstart,jstop,maxp4,maxp5
      integer, intent(in) :: h3d,h2d,h1d,p6d,p5d,p4d,h7d
      integer :: h3h2,h1,p6,p5,p4,h7
      double precision :: triplesx(h3d*h2d,h1d,p6d,maxp5,maxp4)
      double precision, intent(in) :: t2sub(h7d,p4d,p5d,h1d)
      double precision, intent(in) :: v2sub(h3d*h2d,p6d,h7d)
!$acc parallel loop collapse(4)
      do p4=istart,istop
      do p5=jstart,jstop
      do p6=1,p6d
      do h1=1,h1d
      do h7=1,h7d
      do h3h2=1,h2d*h3d
       triplesx(h3h2,h1,p6,p5-jstart+1,p4-istart+1)=
     &     triplesx(h3h2,h1,p6,p5-jstart+1,p4-istart+1)
     &   - t2sub(h7,p4,p5,h1)*v2sub(h3h2,p6,h7)
      enddo
      enddo
      enddo
      enddo
      enddo
      enddo
      return
      end
!
      subroutine acc_sd_tx1_2(h3d,h2d,h1d,p6d,p5d,p4d,
     &               h7d,
     &               triplesx,t2sub,v2sub,
     &                  istart,istop,jstart,jstop,maxp4,maxp5)
      implicit none
      integer, intent(in) :: istart,istop,jstart,jstop,maxp4,maxp5
      integer, intent(in) :: h3d,h2d,h1d,p6d,p5d,p4d,h7d
      integer :: h3,h2,h1,p6,p5,p4,h7
      double precision :: triplesx(h3d,h1d,h2d,p6d,maxp5,maxp4)
      double precision, intent(in) :: t2sub(h7d,p4d,p5d,h1d)
      double precision, intent(in) :: v2sub(h3d,h2d,p6d,h7d)
!$acc parallel loop collapse(4)
      do p4=istart,istop
      do p5=jstart,jstop
      do p6=1,p6d
      do h2=1,h2d
      do h1=1,h1d
      do h3=1,h3d
      do h7=1,h7d
       triplesx(h3,h1,h2,p6,p5-jstart+1,p4-istart+1)=
     &     triplesx(h3,h1,h2,p6,p5-jstart+1,p4-istart+1)
     &   + t2sub(h7,p4,p5,h1)*v2sub(h3,h2,p6,h7)
      enddo
      enddo
      enddo
      enddo
      enddo
      enddo
      enddo
      return
      end
!
      subroutine acc_sd_tx1_3(h3d,h2d,h1d,p6d,p5d,p4d,
     &               h7d,
     &               triplesx,t2sub,v2sub,
     &                  istart,istop,jstart,jstop,maxp4,maxp5)
      implicit none
      integer, intent(in) :: istart,istop,jstart,jstop,maxp4,maxp5
      integer, intent(in) :: h3d,h2d,h1d,p6d,p5d,p4d,h7d
      integer :: h3h2,h1,p6,p5,p4,h7
      double precision :: triplesx(h1d,h3d*h2d,p6d,maxp5,maxp4)
      double precision, intent(in) :: t2sub(h7d,p4d,p5d,h1d)
      double precision, intent(in) :: v2sub(h3d,h2d,p6d,h7d)
      double precision :: scratch(h7d,h3d*h2d,p6d)
      call cute_transpt2_7326(v2sub,scratch,h3d,h2d,p6d,h7d)
!$acc parallel loop collapse(4)
      do p4=istart,istop
      do p5=jstart,jstop
      do p6=1,p6d
      do h3h2=1,h2d*h3d
      do h1=1,h1d
      do h7=1,h7d
       triplesx(h1,h3h2,p6,p5-jstart+1,p4-istart+1)=
     &     triplesx(h1,h3h2,p6,p5-jstart+1,p4-istart+1)
     &    - t2sub(h7,p4,p5,h1)*scratch(h7,h3h2,p6)
!     &   - t2sub(h7,p4,p5,h1)*v2sub(h3,h2,p6,h7)
      enddo
      enddo
      enddo
      enddo
      enddo
      enddo
      return
      end
!
      subroutine acc_sd_tx1_4(h3d,h2d,h1d,p6d,p5d,p4d,
     &               h7d,
     &               triplesx,t2sub,v2sub,
     &                  istart,istop,jstart,jstop,maxp4,maxp5)
      implicit none
      integer, intent(in) :: istart,istop,jstart,jstop,maxp4,maxp5
      integer, intent(in) :: h3d,h2d,h1d,p6d,p5d,p4d,h7d
      integer :: h3h2,h1,p6,p5,p4,h7
      double precision :: triplesx(h3d*h2d,h1d,p5d,maxp5,maxp4)
      double precision, intent(in) :: t2sub(h7d,p4d,p5d,h1d)
      double precision, intent(in) :: v2sub(h3d*h2d,p6d,h7d)
!$acc parallel loop collapse(4)
      do p6=istart,istop
      do p4=jstart,jstop
      do p5=1,p5d
      do h1=1,h1d
      do h7=1,h7d
      do h3h2=1,h2d*h3d
       triplesx(h3h2,h1,p5,p4-jstart+1,p6-istart+1)=
     &     triplesx(h3h2,h1,p5,p4-jstart+1,p6-istart+1)
     &   - t2sub(h7,p4,p5,h1)*v2sub(h3h2,p6,h7)
      enddo
      enddo
      enddo
      enddo
      enddo
      enddo
      return
      end
!
      subroutine acc_sd_tx1_5(h3d,h2d,h1d,p6d,p5d,p4d,
     &               h7d,
     &               triplesx,t2sub,v2sub,
     &                  istart,istop,jstart,jstop,maxp4,maxp5)
      implicit none
      integer, intent(in) :: istart,istop,jstart,jstop,maxp4,maxp5
      integer, intent(in) :: h3d,h2d,h1d,p6d,p5d,p4d,h7d
      integer :: h3,h2,h1,p6,p5,p4,h7
      double precision :: triplesx(h3d,h1d,h2d,p5d,maxp5,maxp4)
      double precision, intent(in) :: t2sub(h7d,p4d,p5d,h1d)
      double precision, intent(in) :: v2sub(h3d,h2d,p6d,h7d)
      double precision :: scratch(h7d,h3d,h2d,p6d)
      call cute_transpt2_7326(v2sub,scratch,h3d,h2d,p6d,h7d)
!$acc parallel loop collapse(4)
      do p6=istart,istop
      do p4=jstart,jstop
      do p5=1,p5d
      do h2=1,h2d
      do h1=1,h1d
      do h3=1,h3d
      do h7=1,h7d
       triplesx(h3,h1,h2,p5,p4-jstart+1,p6-istart+1)=
     &     triplesx(h3,h1,h2,p5,p4-jstart+1,p6-istart+1)
     &   + t2sub(h7,p4,p5,h1)*scratch(h7,h3,h2,p6)
!     &   + t2sub(h7,p4,p5,h1)*v2sub(h3,h2,p6,h7)
      enddo
      enddo
      enddo
      enddo
      enddo
      enddo
      enddo
      return
      end
!
      subroutine acc_sd_tx1_6(h3d,h2d,h1d,p6d,p5d,p4d,
     &               h7d,
     &               triplesx,t2sub,v2sub,
     &                  istart,istop,jstart,jstop,maxp4,maxp5)
      implicit none
      integer, intent(in) :: istart,istop,jstart,jstop,maxp4,maxp5
      integer, intent(in) :: h3d,h2d,h1d,p6d,p5d,p4d,h7d
      integer :: h3h2,h1,p6,p5,p4,h7
      double precision :: triplesx(h1d,h3d*h2d,p5d,maxp5,maxp4)
      double precision, intent(in) :: t2sub(h7d,p4d,p5d,h1d)
      double precision, intent(in) :: v2sub(h3d,h2d,p6d,h7d)
      double precision :: scratch(h7d,h3d*h2d,p6d)
      call cute_transpt2_7326(v2sub,scratch,h3d,h2d,p6d,h7d)
!$acc parallel loop collapse(4)
      do p6=istart,istop
      do p4=jstart,jstop
      do p5=1,p5d
      do h3h2=1,h2d*h3d
      do h1=1,h1d
      do h7=1,h7d
       triplesx(h1,h3h2,p5,p4-jstart+1,p6-istart+1)=
     &     triplesx(h1,h3h2,p5,p4-jstart+1,p6-istart+1)
     &   - t2sub(h7,p4,p5,h1)*scratch(h7,h3h2,p6)
!     &   - t2sub(h7,p4,p5,h1)*v2sub(h3,h2,p6,h7)
      enddo
      enddo
      enddo
      enddo
      enddo
      enddo
      return
      end
!
      subroutine acc_sd_tx1_7(h3d,h2d,h1d,p6d,p5d,p4d,
     &               h7d,
     &               triplesx,t2sub,v2sub,
     &                  istart,istop,jstart,jstop,maxp4,maxp5)
      implicit none
      integer, intent(in) :: istart,istop,jstart,jstop,maxp4,maxp5
      integer, intent(in) :: h3d,h2d,h1d,p6d,p5d,p4d,h7d
      integer :: h3h2,h1,p6,p5,p4,h7
      double precision :: triplesx(h3d*h2d,h1d,p5d,maxp5,maxp4)
      double precision, intent(in) :: t2sub(h7d,p4d,p5d,h1d)
      double precision, intent(in) :: v2sub(h3d*h2d,p6d,h7d)
!$acc parallel loop collapse(4)
      do p4=istart,istop
      do p6=jstart,jstop
      do p5=1,p5d
      do h1=1,h1d
      do h7=1,h7d
      do h3h2=1,h2d*h3d
       triplesx(h3h2,h1,p5,p6-jstart+1,p4-istart+1)=
     &     triplesx(h3h2,h1,p5,p6-jstart+1,p4-istart+1)
     &   + t2sub(h7,p4,p5,h1)*v2sub(h3h2,p6,h7)
      enddo
      enddo
      enddo
      enddo
      enddo
      enddo
      return
      end
!
      subroutine acc_sd_tx1_8(h3d,h2d,h1d,p6d,p5d,p4d,
     &               h7d,
     &               triplesx,t2sub,v2sub,
     &                  istart,istop,jstart,jstop,maxp4,maxp5)
      implicit none
      integer, intent(in) :: istart,istop,jstart,jstop,maxp4,maxp5
      integer, intent(in) :: h3d,h2d,h1d,p6d,p5d,p4d,h7d
      integer :: h3,h2,h1,p6,p5,p4,h7
      double precision :: triplesx(h3d,h1d,h2d,p5d,maxp5,maxp4)
      double precision, intent(in) :: t2sub(h7d,p4d,p5d,h1d)
      double precision, intent(in) :: v2sub(h3d,h2d,p6d,h7d)
      double precision :: scratch(h7d,h3d,h2d,p6d)
      call cute_transpt2_7326(v2sub,scratch,h3d,h2d,p6d,h7d)
!$acc parallel loop collapse(4)
      do p4=istart,istop
      do p6=jstart,jstop
      do p5=1,p5d
      do h2=1,h2d
      do h1=1,h1d
      do h3=1,h3d
      do h7=1,h7d
       triplesx(h3,h1,h2,p5,p6-jstart+1,p4-istart+1)=
     &     triplesx(h3,h1,h2,p5,p6-jstart+1,p4-istart+1)
     &   - t2sub(h7,p4,p5,h1)*scratch(h7,h3,h2,p6)
!     &   - t2sub(h7,p4,p5,h1)*v2sub(h3,h2,p6,h7)
      enddo
      enddo
      enddo
      enddo
      enddo
      enddo
      enddo
      return
      end
!
      subroutine acc_sd_tx1_9(h3d,h2d,h1d,p6d,p5d,p4d,
     &               h7d,
     &               triplesx,t2sub,v2sub,
     &                  istart,istop,jstart,jstop,maxp4,maxp5)
      implicit none
      integer, intent(in) :: istart,istop,jstart,jstop,maxp4,maxp5
      integer, intent(in) :: h3d,h2d,h1d,p6d,p5d,p4d,h7d
      integer :: h3h2,h1,p6,p5,p4,h7
      double precision :: triplesx(h1d,h3d*h2d,p5d,maxp5,maxp4)
      double precision, intent(in) :: t2sub(h7d,p4d,p5d,h1d)
      double precision, intent(in) :: v2sub(h3d,h2d,p6d,h7d)
      double precision :: scratch(h7d,h3d*h2d,p6d)
      call cute_transpt2_7326(v2sub,scratch,h3d,h2d,p6d,h7d)
!$acc parallel loop collapse(4)
      do p4=istart,istop
      do p6=jstart,jstop
      do p5=1,p5d
      do h3h2=1,h2d*h3d
      do h1=1,h1d
      do h7=1,h7d
       triplesx(h1,h3h2,p5,p6-jstart+1,p4-istart+1)=
     &     triplesx(h1,h3h2,p5,p6-jstart+1,p4-istart+1)
     &   + t2sub(h7,p4,p5,h1)*scratch(h7,h3h2,p6)
!     &   + t2sub(h7,p4,p5,h1)*v2sub(h3,h2,p6,h7)
      enddo
      enddo
      enddo
      enddo
      enddo
      enddo
      return
      end
!
      subroutine acc_sd_ty1_1(h3d,h2d,h1d,p6d,p5d,p4d,
     &               p7d,
     &               triplesx,t2sub,v2sub,
     &                  istart,istop,jstart,jstop,maxp4,maxp5)
      implicit none
      integer, intent(in) :: istart,istop,jstart,jstop,maxp4,maxp5
      integer, intent(in) :: h3d,h2d,h1d,p6d,p5d,p4d,p7d
      integer :: h3,h2,h1,p6,p5,p4,p7
      double precision :: triplesx(h3d,h2d,h1d,p6d,maxp5,maxp4)
      double precision, intent(in) :: t2sub(p7d,p4d,h1d,h2d)
      double precision, intent(in) :: v2sub(p7d,h3d,p6d,p5d)
      double precision :: scratch(p7d,h2d,h1d,p4d)
      call cute_transpt2_7124(t2sub,scratch,p4d,h1d,h2d,p7d)
!$acc parallel loop collapse(4)
      do p4=istart,istop
      do p5=jstart,jstop
      do p6=1,p6d
      do h1=1,h1d
      do h2=1,h2d
      do h3=1,h3d
      do p7=1,p7d
       triplesx(h3,h2,h1,p6,p5-jstart+1,p4-istart+1)=
     &     triplesx(h3,h2,h1,p6,p5-jstart+1,p4-istart+1)
     &   - scratch(p7,h2,h1,p4)*v2sub(p7,h3,p6,p5)
!     &   - t2sub(p7,p4,h1,h2)*v2sub(p7,h3,p6,p5)
      enddo
      enddo
      enddo
      enddo
      enddo
      enddo
      enddo
      return
      end
!
      subroutine acc_sd_ty1_2(h3d,h2d,h1d,p6d,p5d,p4d,
     &               p7d,
     &               triplesx,t2sub,v2sub,
     &                  istart,istop,jstart,jstop,maxp4,maxp5)
      implicit none
      integer, intent(in) :: istart,istop,jstart,jstop,maxp4,maxp5
      integer, intent(in) :: h3d,h2d,h1d,p6d,p5d,p4d,p7d
      integer :: h3,h2h1,p6,p5,p4,p7
      double precision :: triplesx(h2d*h1d,h3d,p6d,maxp5,maxp4)
      double precision, intent(in) :: t2sub(p7d,p4d,h1d,h2d)
      double precision, intent(in) :: v2sub(p7d,h3d,p6d,p5d)
      double precision :: scratch(h2d*h1d,p4d,p7d)
      call cute_transpt2(t2sub,scratch,h2d,h1d,p4d,p7d)
!$acc parallel loop collapse(4)
      do p4=istart,istop
      do p5=jstart,jstop
      do p6=1,p6d
      do h3=1,h3d
      do p7=1,p7d
      do h2h1=1,h1d*h2d
       triplesx(h2h1,h3,p6,p5-jstart+1,p4-istart+1)=
     &     triplesx(h2h1,h3,p6,p5-jstart+1,p4-istart+1)
     &   - scratch(h2h1,p4,p7)*v2sub(p7,h3,p6,p5)
!     &   - t2sub(p7,p4,h1,h2)*v2sub(p7,h3,p6,p5)
      enddo
      enddo
      enddo
      enddo
      enddo
      enddo
      return
      end
!
      subroutine acc_sd_ty1_3(h3d,h2d,h1d,p6d,p5d,p4d,
     &               p7d,
     &               triplesx,t2sub,v2sub,
     &                  istart,istop,jstart,jstop,maxp4,maxp5)
      implicit none
      integer, intent(in) :: istart,istop,jstart,jstop,maxp4,maxp5
      integer, intent(in) :: h3d,h2d,h1d,p6d,p5d,p4d,p7d
      integer :: h3,h2,h1,p6,p5,p4,p7
      double precision :: triplesx(h2d,h3d,h1d,p6d,maxp5,maxp4)
      double precision, intent(in) :: t2sub(p7d,p4d,h1d,h2d)
      double precision, intent(in) :: v2sub(p7d,h3d,p6d,p5d)
      double precision :: scratch(p7d,h2d,h1d,p4d)
      call cute_transpt2_7124(t2sub,scratch,p4d,h1d,h2d,p7d)
!$acc parallel loop collapse(4)
      do p4=istart,istop
      do p5=jstart,jstop
      do p6=1,p6d
      do h1=1,h1d
      do h3=1,h3d
      do h2=1,h2d
      do p7=1,p7d
       triplesx(h2,h3,h1,p6,p5-jstart+1,p4-istart+1)=
     &     triplesx(h2,h3,h1,p6,p5-jstart+1,p4-istart+1)
     &   + scratch(p7,h2,h1,p4)*v2sub(p7,h3,p6,p5)
!     &   + t2sub(p7,p4,h1,h2)*v2sub(p7,h3,p6,p5)
      enddo
      enddo
      enddo
      enddo
      enddo
      enddo
      enddo
      return
      end
!
      subroutine acc_sd_ty1_4(h3d,h2d,h1d,p6d,p5d,p4d,
     &               p7d,
     &               triplesx,t2sub,v2sub,
     &                  istart,istop,jstart,jstop,maxp4,maxp5)
      implicit none
      integer, intent(in) :: istart,istop,jstart,jstop,maxp4,maxp5
      integer, intent(in) :: h3d,h2d,h1d,p6d,p5d,p4d,p7d
      integer :: h3,h2,h1,p6,p5,p4,p7
      double precision :: triplesx(h3d,h2d,h1d,p6d,maxp5,maxp4)
      double precision, intent(in) :: t2sub(p7d,p4d,h1d,h2d)
      double precision, intent(in) :: v2sub(p7d,h3d,p6d,p5d)
      double precision :: scratch(p7d,h2d,h1d,p4d)
      call cute_transpt2_7124(t2sub,scratch,p4d,h1d,h2d,p7d)
!$acc parallel loop collapse(4)
      do p5=istart,istop
      do p4=jstart,jstop
      do p6=1,p6d
      do h1=1,h1d
      do h2=1,h2d
      do h3=1,h3d
      do p7=1,p7d
       triplesx(h3,h2,h1,p6,p4-jstart+1,p5-istart+1)=
     &     triplesx(h3,h2,h1,p6,p4-jstart+1,p5-istart+1)
     &   + scratch(p7,h2,h1,p4)*v2sub(p7,h3,p6,p5)
!     &   + t2sub(p7,p4,h1,h2)*v2sub(p7,h3,p6,p5)
      enddo
      enddo
      enddo
      enddo
      enddo
      enddo
      enddo
      return
      end
!
      subroutine acc_sd_ty1_5(h3d,h2d,h1d,p6d,p5d,p4d,
     &               p7d,
     &               triplesx,t2sub,v2sub,
     &                  istart,istop,jstart,jstop,maxp4,maxp5)
      implicit none
      integer, intent(in) :: istart,istop,jstart,jstop,maxp4,maxp5
      integer, intent(in) :: h3d,h2d,h1d,p6d,p5d,p4d,p7d
      integer :: h3,h2h1,p6,p5,p4,p7
      double precision :: triplesx(h2d*h1d,h3d,p6d,maxp5,maxp4)
      double precision, intent(in) :: t2sub(p7d,p4d,h1d,h2d)
      double precision, intent(in) :: v2sub(p7d,h3d,p6d,p5d)
      double precision :: scratch(h2d*h1d,p4d,p7d)
      call cute_transpt2(t2sub,scratch,h2d,h1d,p4d,p7d)
!$acc parallel loop collapse(4)
      do p5=istart,istop
      do p4=jstart,jstop
      do p6=1,p6d
      do h3=1,h3d
      do p7=1,p7d
      do h2h1=1,h2d*h1d
       triplesx(h2h1,h3,p6,p4-jstart+1,p5-istart+1)=
     &     triplesx(h2h1,h3,p6,p4-jstart+1,p5-istart+1)
     &   + scratch(h2h1,p4,p7)*v2sub(p7,h3,p6,p5)
!     &   + t2sub(p7,p4,h1,h2)*v2sub(p7,h3,p6,p5)
      enddo
      enddo
      enddo
      enddo
      enddo
      enddo
      return
      end
!
      subroutine acc_sd_ty1_6(h3d,h2d,h1d,p6d,p5d,p4d,
     &               p7d,
     &               triplesx,t2sub,v2sub,
     &                  istart,istop,jstart,jstop,maxp4,maxp5)
      implicit none
      integer, intent(in) :: istart,istop,jstart,jstop,maxp4,maxp5
      integer, intent(in) :: h3d,h2d,h1d,p6d,p5d,p4d,p7d
      integer :: h3,h2,h1,p6,p5,p4,p7
      double precision :: triplesx(h2d,h3d,h1d,p6d,maxp5,maxp4)
      double precision, intent(in) :: t2sub(p7d,p4d,h1d,h2d)
      double precision, intent(in) :: v2sub(p7d,h3d,p6d,p5d)
      double precision :: scratch(p7d,h2d,h1d,p4d)
      call cute_transpt2_7124(t2sub,scratch,p4d,h1d,h2d,p7d)
!$acc parallel loop collapse(4)
      do p5=istart,istop
      do p4=jstart,jstop
      do p6=1,p6d
      do h1=1,h1d
      do h3=1,h3d
      do h2=1,h2d
      do p7=1,p7d
       triplesx(h2,h3,h1,p6,p4-jstart+1,p5-istart+1)=
     &     triplesx(h2,h3,h1,p6,p4-jstart+1,p5-istart+1)
     &   - scratch(p7,h2,h1,p4)*v2sub(p7,h3,p6,p5)
!     &   - t2sub(p7,p4,h1,h2)*v2sub(p7,h3,p6,p5)
      enddo
      enddo
      enddo
      enddo
      enddo
      enddo
      enddo
      return
      end
!
      subroutine acc_sd_ty1_7(h3d,h2d,h1d,p6d,p5d,p4d,
     &               p7d,
     &               triplesx,t2sub,v2sub,
     &                  istart,istop,jstart,jstop,maxp4,maxp5)
      implicit none
      integer, intent(in) :: istart,istop,jstart,jstop,maxp4,maxp5
      integer, intent(in) :: h3d,h2d,h1d,p6d,p5d,p4d,p7d
      integer :: h3,h2,h1,p6,p5,p4,p7
      double precision :: triplesx(h3d,h2d,h1d,p4d,maxp5,maxp4)
      double precision, intent(in) :: t2sub(p7d,p4d,h1d,h2d)
      double precision, intent(in) :: v2sub(p7d,h3d,p6d,p5d)
      double precision :: scratch(p7d,h2d,h1d,p4d)
      call cute_transpt2_7124(t2sub,scratch,p4d,h1d,h2d,p7d)
!$acc parallel loop collapse(4)
      do p5=istart,istop
      do p6=jstart,jstop
      do p4=1,p4d
      do h1=1,h1d
      do h2=1,h2d
      do h3=1,h3d
      do p7=1,p7d
       triplesx(h3,h2,h1,p4,p6-jstart+1,p5-istart+1)=
     &     triplesx(h3,h2,h1,p4,p6-jstart+1,p5-istart+1)
     &   - scratch(p7,h2,h1,p4)*v2sub(p7,h3,p6,p5)
!     &   - t2sub(p7,p4,h1,h2)*v2sub(p7,h3,p6,p5)
      enddo
      enddo
      enddo
      enddo
      enddo
      enddo
      enddo
      return
      end
!
      subroutine acc_sd_ty1_8(h3d,h2d,h1d,p6d,p5d,p4d,p7d,
     &               triplesx,t2sub,v2sub,
     &                  istart,istop,jstart,jstop,maxp4,maxp5)
      implicit none
      integer, intent(in) :: istart,istop,jstart,jstop,maxp4,maxp5
      integer, intent(in) :: h3d,h2d,h1d,p6d,p5d,p4d,p7d
      double precision :: triplesx(h2d*h1d,h3d,p4d,maxp5,maxp4)
      double precision, intent(in) :: t2sub(p7d,p4d,h1d,h2d)
      double precision, intent(in) :: v2sub(p7d,h3d,p6d,p5d)
      double precision :: scratch(h2d*h1d,p4d,p7d)
      integer :: h3,h2h1,p6,p5,p4,p7
      call cute_transpt2(t2sub,scratch,h2d,h1d,p4d,p7d)
!$acc parallel loop collapse(4)
      do p5=istart,istop
      do p6=jstart,jstop
      do p4=1,p4d
      do h3=1,h3d
      do p7=1,p7d
      do h2h1=1,h1d*h2d
       triplesx(h2h1,h3,p4,p6-jstart+1,p5-istart+1)=
     &     triplesx(h2h1,h3,p4,p6-jstart+1,p5-istart+1)
     &   - scratch(h2h1,p4,p7)*v2sub(p7,h3,p6,p5)
!     &   - t2sub(p7,p4,h1,h2)*v2sub(p7,h3,p6,p5)
      enddo
      enddo
      enddo
      enddo
      enddo
      enddo
      return
      end
!
      subroutine acc_sd_ty1_9(h3d,h2d,h1d,p6d,p5d,p4d,p7d,
     &               triplesx,t2sub,v2sub,
     &                  istart,istop,jstart,jstop,maxp4,maxp5)
      implicit none
      integer, intent(in) :: istart,istop,jstart,jstop,maxp4,maxp5
      integer, intent(in) :: h3d,h2d,h1d,p6d,p5d,p4d,p7d
      double precision :: triplesx(h2d,h3d,h1d,p4d,maxp5,maxp4)
      double precision, intent(in) :: t2sub(p7d,p4d,h1d,h2d)
      double precision, intent(in) :: v2sub(p7d,h3d,p6d,p5d)
      double precision :: scratch(p7d,h2d,h1d,p4d)
      integer :: h3,h2,h1,p6,p5,p4,p7
      call cute_transpt2_7124(t2sub,scratch,p4d,h1d,h2d,p7d)
!$acc parallel loop collapse(4)
      do p5=istart,istop
      do p6=jstart,jstop
      do p4=1,p4d
      do h1=1,h1d
      do h3=1,h3d
      do h2=1,h2d
      do p7=1,p7d
       triplesx(h2,h3,h1,p4,p6-jstart+1,p5-istart+1)=
     &     triplesx(h2,h3,h1,p4,p6-jstart+1,p5-istart+1)
     &   + scratch(p7,h2,h1,p4)*v2sub(p7,h3,p6,p5)
!     &   + t2sub(p7,p4,h1,h2)*v2sub(p7,h3,p6,p5)
      enddo
      enddo
      enddo
      enddo
      enddo
      enddo
      enddo
      return
      end

      subroutine cute_transpt2(t2sub,scratch,h1d,p5d,p4d,h7d)
      implicit none
      integer, intent(in) :: h1d,p5d,p4d,h7d
      double precision, intent(in) :: t2sub(h7d,p4d,p5d,h1d)
      double precision :: scratch(h1d,p5d,p4d,h7d)
      integer :: h1,p5,p4,h7,pp5,hh1
      do h7=1,h7d
         do p4=1,p4d
            do p5=1,p5d
               do h1=1,h1d
                  scratch(h1,p5,p4,h7)=t2sub(h7,p4,p5,h1)
               enddo
            enddo
         enddo
      enddo
      end

      subroutine cute_transpt2_7124(t2sub,scratch,p4d,h1d,h2d,p7d)
      implicit none
      integer, intent(in) :: p4d,h1d,h2d,p7d
      double precision, intent(in) :: t2sub(p7d,p4d,h1d,h2d)
      double precision :: scratch(p7d,h2d,h1d,p4d)
      integer :: p4,h1,h2,p7
      do p4=1,p4d
         do h1=1,h1d
            do h2=1,h2d
               do p7=1,p7d
                  scratch(p7,h2,h1,p4)=t2sub(p7,p4,h1,h2)
               enddo
            enddo
         enddo
      enddo
      end


      subroutine cute_transpt2_7326(v2sub,scratch,h3d,h2d,p6d,h7d)
      implicit none
      integer, intent(in) :: h3d,h2d,p6d,h7d
      double precision, intent(in) :: v2sub(h3d,h2d,p6d,h7d)
      double precision :: scratch(h7d,h3d,h2d,p6d)
      integer :: h3,h2,p6,h7
      do p6=1,p6d
         do h2=1,h2d
            do h3=1,h3d
               do h7=1,h7d
                  scratch(h7,h3,h2,p6)=v2sub(h3,h2,p6,h7)
               enddo
            enddo
         enddo
      enddo
      end

      end module tce_ccsd_t_cutensor
