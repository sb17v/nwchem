      module tce_ccsd_t_cutensor

      use cutensorex
      implicit none

      type(cutensorHandle) :: cut_handle 
      type(cutensorContractionFind) :: cut_find
      type(cudaDataType), parameter :: cut_double = 
     &                                 cudaDataType(CUDA_R_64F)
      type(cutensorComputeType), parameter :: cut_compF64 = 
     &                                        CUTENSOR_R_MIN_64F
      integer(kind=cuda_stream_kind) :: stream

      ! because 2/4/6 are default integers, we need fixed-width versions
      integer(4), parameter :: two = 2
      integer(4), parameter :: four = 4
      integer(4), parameter :: six = 6

      contains

      subroutine ccsd_t_cutensor(d_t1,k_t1_offset,d_t2,k_t2_offset,
     &                           d_v2,k_v2_offset,energy1,energy2,
     &                           size_t1)
      use cudafor
      use cutensorex
      use openacc
      implicit none
#include "global.fh"
#include "mafdecls.fh"
#include "util.fh"
#include "errquit.fh"
#include "tce.fh"
#include "tce_main.fh"
      integer, intent(in) :: d_t1, k_t1_offset
      integer, intent(in) :: d_t2, k_t2_offset
      integer, intent(in) :: d_v2, k_v2_offset
      double precision, intent(inout) :: energy1,energy2
      integer, intent(in) :: size_t1
      integer :: t_h1b, t_h1, t_h2b, t_h2, t_h3b, t_h3
      integer :: t_p4b, t_p4, t_p5b, t_p5, t_p6b, t_p6
      double precision, allocatable, managed :: f_singles(:)
      double precision, allocatable, managed :: f_doubles(:)
      double precision, allocatable, managed :: t1_local(:)
      integer :: alloc_error
      integer :: size_t3
      integer :: next
      integer :: nprocs
      integer :: count
      integer :: offset_p4,offset_p5,offset_p6
      integer :: offset_h1,offset_h2,offset_h3
      integer :: range_p4,range_p5,range_p6
      integer :: range_h1,range_h2,range_h3
      double precision :: energy(2)
      integer, allocatable :: f_aux(:), f_list(:)
      integer :: tot_task,w_thresh,task_thresh
      integer :: k,iptr,k_grain
      double precision :: wall
      integer :: tasks_skip
      integer :: pct_progr
      integer, parameter :: n_progr=20
      logical :: i_progr(n_progr)
      integer(4) :: cuda_err
      type(cutensorStatus) :: cut_status
      integer :: nxtask0
      external nxtask0

      cuda_err = cudaStreamCreate(stream)
      if (cuda_err.ne.0) then
        write(6,*) cudaGetErrorString(cuda_err)
        call errquit('cudaStreamCreate',cuda_err,UNKNOWN_ERR)
      endif
      ! init the library
      cut_status = cutensorInit(cut_handle)
      if (cut_status .ne. CUTENSOR_STATUS_SUCCESS) then
        call errquit('cutensorInit',int(cut_status%stat),UNKNOWN_ERR)
      endif
      ! init the find algorithm to default
      cut_status = cutensorInitContractionFind(cut_handle,cut_find,
     &                                         CUTENSOR_ALGO_DEFAULT)
      if (cut_status .ne. CUTENSOR_STATUS_SUCCESS) then
        call errquit('cutensorInitContractionFind',
     &               int(cut_status%stat),UNKNOWN_ERR)
      endif
! - T1/X1 LOCALIZATION ----------
      allocate(t1_local(1:size_t1), stat=alloc_error)
      if (alloc_error.ne.0) then
        call errquit('ccsd_t_6dts: allocate failed',
     &               alloc_error, MA_ERR)
      endif
      t1_local = 0.0d0
      call get_block(d_t1,t1_local,size_t1,0)
! -------------------------------
      nprocs = GA_NNODES()
      energy1 = 0.0d0
      energy2 = 0.0d0

!     estimate triplesx size
      range_p4=0
      do t_p4b = noab+1,noab+nvab
         range_p4 = max(range_p4,int_mb(k_range+t_p4b-1))
      enddo
      range_h1=0
      do t_h1b = 1,noab
         range_h1 = max(range_h1,int_mb(k_range+t_h1b-1))
      enddo
      size_t3=(range_p4**3)*(range_h1**3)

      allocate( f_singles(1:size_t3), f_doubles(1:size_t3), 
     &          stat=alloc_error)
      if (alloc_error.ne.0) then
        call errquit('ccsd_t_cutensor: MA error f_t3',0,MA_ERR)
      endif
      wall=-util_wallsec()
      tot_task= ccsd_t_6tasks_cutensor(restricted,noab,nvab,
     &                        int_mb(k_spin),int_mb(k_sym))
      allocate( f_list(7*tot_task), f_aux(7*tot_task),
     &          stat=alloc_error)
      if (alloc_error.ne.0) then
        call errquit('ccsd_t_cutensor: MA error task list/aux',0,MA_ERR)
      endif
!
!     get first task with weight lt ? 8
!
!      w_thresh=10
      w_thresh=0
      w_thresh=w_thresh**6
      call ccsd_t_neword_cutensor(tot_task, w_thresh,task_thresh,
     &     restricted,noab,nvab,
     &     int_mb(k_spin),int_mb(k_sym),
     &     int_mb(k_range), f_aux, f_list)
      deallocate( f_aux, stat=alloc_error)
      if (alloc_error.ne.0) then
        call errquit('ccsd_t_cutensor',112,MA_ERR)
      endif

      count = 0
      k_grain=1
      tasks_skip=0
      next = nxtask0(nprocs,k_grain,tasks_skip)
      do k=1,n_progr
         i_progr(k)=.true.
      enddo
!     stagger start of loop
      call util_mpinap(100)
      if(task_thresh.gt.1) then
      do k=1,task_thresh-1
         if (next.eq.count) then
            call ccsd_t_loop_cutensor(k,energy1,energy2,
     &              f_list,
     &              int_mb(k_range),int_mb(k_offset),
     &              f_singles,f_doubles,
     &              t1_local,k_t1_offset,
     &              d_t2,d_v2,k_t2_offset,k_v2_offset,
     &              restricted,k_evl_sorted,size_t3)
            if(ga_nodeid().eq.2) then
               pct_progr=(k*n_progr)/tot_task
               if(i_progr(pct_progr)) then
                  i_progr(pct_progr)=.false.
                  write(6,'(a,i5,a,i4,a,f15.1,a,f9.1)')
     &                  '0task ',k,'  done ',
     &                  int((k*100d0)/tot_task),'%  at',
     &                  wall+util_wallsec(),' sec, (size)^1/6= ',
     &                  (size_t3)**(1d0/6d0)
                  call util_flush(6)
               endif
            endif
            next = nxtask0(nprocs,k_grain,tasks_skip)
         endif
         count = count + 1
      enddo
      endif
      if(task_thresh.le.tot_task) then
      next = nxtask0(-nprocs,k_grain,tasks_skip)
      if(.true.) then
         next = nxtask0(nprocs,k_grain,tasks_skip)
         count=0
         do k=task_thresh,tot_task
            if (next.eq.count) then
               call ccsd_t_loop_cutensor(k,energy1,energy2,
     &                 f_list,
     &                 int_mb(k_range),int_mb(k_offset),
     &                 f_singles,f_doubles,
     &                 t1_local,k_t1_offset,
     &                 d_t2,d_v2,k_t2_offset,k_v2_offset,
     &                 restricted,k_evl_sorted,size_t3)
               next = nxtask0(nprocs,k_grain,tasks_skip)
               if(ga_nodeid().eq.2) then
                  pct_progr=(k*n_progr)/tot_task
                  if(i_progr(pct_progr)) then
                     i_progr(pct_progr)=.false.
                     write(6,'(a,i8,a,i4,a,f15.1,a,f9.1)')
     &                     ' task',k,'  done ',
     &                     int((k*100d0)/tot_task),'%  at',
     &                     wall+util_wallsec(),' sec, (size)^1/6= ',
     &                     (size_t3)**(1d0/6d0)
                     call util_flush(6)
                  endif
               endif
            endif
            count = count + 1
         enddo
      endif
      endif

      deallocate( f_list, f_singles, f_doubles, stat=alloc_error)
      if (alloc_error.ne.0) then
        call errquit('ccsd_t_cutensor: deallocat failed',
     &               alloc_error,MA_ERR)
      endif
      next = nxtask0(-nprocs,k_grain,tasks_skip)
!      call ga_sync()
      energy(1) = energy1
      energy(2) = energy2
      call ga_mask_sync(.false.,.true.)
      call ga_dgop(1975,energy,2,'+')
      energy1 = energy(1)
      energy2 = energy(2)
! - T1/X1 LOCALIZATION ------
      deallocate(t1_local, stat=alloc_error)
      if (alloc_error.ne.0)  then
        call errquit('ccsd_t_cutensor: deallocate failed',
     &               alloc_error, MA_ERR)
      endif
! ---------------------------
      cuda_err = cudaStreamDestroy(stream)
      if (cuda_err.ne.0) then
        write(6,*) cudaGetErrorString(cuda_err)
        call errquit('cudaStreamDestroy',cuda_err,UNKNOWN_ERR)
      endif
      end

      subroutine ccsd_t_dot_cutensor(a_singles, a_doubles,
     &                               restricted,
     &                               h1b,h2b,h3b,p4b,p5b,p6b,
     &                               o_h1,o_h2,o_h3,o_p4,o_p5,o_p6,
     &                               r_h1,r_h2,r_h3,r_p4,r_p5,r_p6,
     &                               energy1,energy2)
      implicit none
      double precision, managed :: a_singles(r_h3,r_h2,r_h1,r_p6,r_p5,r_p4)
      double precision, managed :: a_doubles(r_h3,r_h2,r_h1,r_p6,r_p5,r_p4)
      logical, intent(in) :: restricted
      integer, intent(in) :: h1b, h2b, h3b, p4b, p5b, p6b
      integer, intent(in) :: r_h1,r_h2,r_h3,r_p4,r_p5,r_p6
      double precision, intent(in) :: o_h1(r_h1),o_h2(r_h2),o_h3(r_h3)
      double precision, intent(in) :: o_p4(r_p4),o_p5(r_p5),o_p6(r_p6)
      double precision :: energy1,energy2
      double precision :: factor,denom,sing,doub
      double precision :: d_h1,d_h2,d_h3,d_p4,d_p5,d_p6
      integer :: h1, h2, h3, p4, p5, p6
!
! factor = [ 1/36, 1/18, 1/12, 1/6, 1/4, 1/3, 1/2, 1, 2]
!
      if (restricted) then
        factor = 2.0d0
      else
        factor = 1.0d0
      endif
      if ((p4b.eq.p5b).and.(p5b.eq.p6b)) then
        factor = factor / 6.0d0
      else if ((p4b.eq.p5b).or.(p5b.eq.p6b)) then
        factor = factor / 2.0d0
      endif
      if ((h1b.eq.h2b).and.(h2b.eq.h3b)) then
        factor = factor / 6.0d0
      else if ((h1b.eq.h2b).or.(h2b.eq.h3b)) then
        factor = factor / 2.0d0
      endif
!$acc parallel loop collapse(6) reduction(+:energy1,energy2)
      do p4 = 1, r_p4
       do p5 = 1, r_p5
        do p6 = 1, r_p6
         do h1 = 1, r_h1
          do h2 = 1, r_h2
           do h3 = 1, r_h3
            d_p4 = o_p4(p4)
            d_p5 = o_p5(p5)
            d_p6 = o_p6(p6)
            d_h1 = o_h1(h1)
            d_h2 = o_h2(h2)
            d_h3 = o_h3(h3)
            sing = a_singles(h3,h2,h1,p6,p5,p4)
            doub = a_doubles(h3,h2,h1,p6,p5,p4)
            denom = doub * factor / (d_h1+d_h2+d_h3-d_p4-d_p5-d_p6)
            energy1 = energy1 + denom*doub
            energy2 = energy2 + denom*(doub+sing)
           enddo
          enddo
         enddo
        enddo
       enddo
      enddo
      end

      subroutine ccsd_t_loop_cutensor(k,energy1,energy2,
     &                                k_list,k_range,k_offset,
     &                                a_singles,a_doubles,
     &                                t1_local,k_t1_offset,
     &                                d_t2,d_v2,k_t2_offset,k_v2_offset,
     &                                restricted,k_evl_sorted,size_t3)
      implicit none
#include "mafdecls.fh"
      integer, intent(in) :: k
      double precision, intent(inout) :: energy1,energy2
      integer, intent(in) :: k_list(7,*),k_range(*),k_offset(*)
      double precision, intent(inout), managed :: a_singles(:)
      double precision, intent(inout), managed :: a_doubles(:)
      double precision, intent(in) :: t1_local(:)
      logical, intent(in) :: restricted
      integer, intent(in) :: k_evl_sorted
      integer, intent(in) :: k_t1_offset
      integer, intent(in) :: d_t2,d_v2
      integer, intent(in) :: k_t2_offset,k_v2_offset
      integer, intent(inout) :: size_t3
      integer :: t_p4b,t_p5b,t_p6b,t_h1b,t_h2b,t_h3b
      integer :: range_p4,range_p5,range_p6
      integer :: range_h1,range_h2,range_h3
      integer :: offset_p4,offset_p5,offset_p6
      integer :: offset_h1,offset_h2,offset_h3
      t_p4b=k_list(1,k)
      t_p5b=k_list(2,k)
      t_p6b=k_list(3,k)
      t_h1b=k_list(4,k)
      t_h2b=k_list(5,k)
      t_h3b=k_list(6,k)
      range_p4 = k_range(t_p4b)
      range_p5 = k_range(t_p5b)
      range_p6 = k_range(t_p6b)
      range_h1 = k_range(t_h1b)
      range_h2 = k_range(t_h2b)
      range_h3 = k_range(t_h3b)
      offset_p4 = k_evl_sorted+k_offset(t_p4b)-1
      offset_p5 = k_evl_sorted+k_offset(t_p5b)-1
      offset_p6 = k_evl_sorted+k_offset(t_p6b)-1
      offset_h1 = k_evl_sorted+k_offset(t_h1b)-1
      offset_h2 = k_evl_sorted+k_offset(t_h2b)-1
      offset_h3 = k_evl_sorted+k_offset(t_h3b)-1
      size_t3 = range_p4 * range_p5 * range_p6
     &        * range_h1 * range_h2 * range_h3
      a_singles = 0.0d0
      a_doubles = 0.0d0
      call ccsd_t_singles_cutensor(a_singles,
     &                 t1_local,d_v2,k_t1_offset,k_v2_offset,
     &                 t_h1b,t_h2b,t_h3b,t_p4b,t_p5b,t_p6b)
      call ccsd_t_doubles_cutensor(a_doubles,
     &                 d_t2,d_v2,k_t2_offset,k_v2_offset,
     &                 t_h1b,t_h2b,t_h3b,t_p4b,t_p5b,t_p6b)
      call ccsd_t_dot_cutensor(a_singles,a_doubles,restricted,
     &                 t_h1b,t_h2b,t_h3b,t_p4b,t_p5b,t_p6b,
     &                 dbl_mb(offset_h1+1),dbl_mb(offset_h2+1),
     &                 dbl_mb(offset_h3+1),dbl_mb(offset_p4+1),
     &                 dbl_mb(offset_p5+1),dbl_mb(offset_p6+1),
     &                 range_h1,range_h2,range_h3,
     &                 range_p4,range_p5,range_p6,
     &                 energy1,energy2)
      end

      integer function ccsd_t_6tasks_cutensor(restricted,
     &                                        noab,nvab,kspin,ksym)
      implicit none
      logical, intent(in) :: restricted
      integer, intent(in) :: noab,nvab
      integer, intent(in) :: kspin(*),ksym(*)
      integer :: ccsd_t_6tasks
      integer :: t_p4b,t_p5b,t_p6b
      integer :: t_h1b,t_h2b,t_h3b
      ccsd_t_6tasks=0
      do t_p4b = noab+1,noab+nvab
       do t_p5b = t_p4b,noab+nvab
        do t_p6b = t_p5b,noab+nvab
         do t_h1b = 1,noab
          do t_h2b = t_h1b,noab
           do t_h3b = t_h2b,noab
            if (kspin(t_p4b)+kspin(t_p5b)+kspin(t_p6b).eq.
     &          kspin(t_h1b)+kspin(t_h2b)+kspin(t_h3b)) then
            if ((.not.restricted).or.
     &          (kspin(t_p4b)+kspin(t_p5b)+kspin(t_p6b)+
     &         kspin(t_h1b)+kspin(t_h2b)+kspin(t_h3b).le.8)) then
            if (ieor(ksym(t_p4b),ieor(ksym(t_p5b),ieor(ksym(t_p6b),
     &          ieor(ksym(t_h1b),ieor(ksym(t_h2b),ksym(t_h3b)
     &              ))))).eq.0) then
               ccsd_t_6tasks = ccsd_t_6tasks + 1
            endif
            endif
            endif
           enddo
          enddo
         enddo
        enddo
       enddo
      enddo
      ccsd_t_6tasks_cutensor = ccsd_t_6tasks
      end

      subroutine ccsd_t_neword_cutensor(tot_task, 
     &                                  w_thresh,task_thresh,
     &                                  restricted,noab,nvab,
     &                                  kspin,ksym,
     &                                  krange,kaux,klist)
      implicit none
#include "errquit.fh"
#include "mafdecls.fh"
#include "global.fh"
      logical :: restricted
      integer :: w_thresh,task_thresh
      integer :: tot_task
      integer :: noab,nvab
      integer :: kspin(*),ksym(*)
      integer :: krange(*)
      integer :: kaux(7,*),klist(7,*)
      integer :: t_p4b,t_p5b,t_p6b
      integer :: t_h1b,t_h2b,t_h3b
      integer :: i,j,m
      integer :: wl_max,wl_min
      integer(8) :: range_p4,range_p5,range_p6
      integer(8) :: range_h1,range_h2,range_h3
      integer :: nsplits,ii,w_in
      task_thresh=tot_task
      m=0
      do t_p4b = noab+1,noab+nvab
       do t_p5b = t_p4b,noab+nvab
        do t_p6b = t_p5b,noab+nvab
         do t_h1b = 1,noab
          do t_h2b = t_h1b,noab
           do t_h3b = t_h2b,noab
            if (kspin(t_p4b)+kspin(t_p5b)+kspin(t_p6b).eq.
     &          kspin(t_h1b)+kspin(t_h2b)+kspin(t_h3b)) then
            if ((.not.restricted).or.
     &          (kspin(t_p4b)+kspin(t_p5b)+kspin(t_p6b)+
     &         kspin(t_h1b)+kspin(t_h2b)+kspin(t_h3b).le.8)) then
            if (ieor(ksym(t_p4b),ieor(ksym(t_p5b),ieor(ksym(t_p6b),
     &          ieor(ksym(t_h1b),ieor(ksym(t_h2b),ksym(t_h3b)
     &              ))))).eq.0) then
            range_p4 = krange(t_p4b)
            range_p5 = krange(t_p5b)
            range_p6 = krange(t_p6b)
            range_h1 = krange(t_h1b)
            range_h2 = krange(t_h2b)
            range_h3 = krange(t_h3b)
            m=m+1
            kaux(1,m)=t_p4b
            kaux(2,m)=t_p5b
            kaux(3,m)=t_p6b
            kaux(4,m)=t_h1b
            kaux(5,m)=t_h2b
            kaux(6,m)=t_h3b
            kaux(7,m)=range_p4*range_p5*range_p6
     &               *range_h1*range_h2*range_h3
            endif
            endif
            endif
           enddo
          enddo
         enddo
        enddo
       enddo
      enddo
      if(m.ne.tot_task)  then
         write(6,*) ' m ',m,' tot_task ',tot_task
         call errquit("incompatibility_task",6,MA_ERR)
      endif
      wl_max=0
      do m=1,tot_task
        wl_max=max(kaux(7,m),wl_max)
      enddo
      wl_min=wl_max
      do m=1,tot_task
        wl_min=min(kaux(7,m),wl_min)
      enddo
!     all tiles the same. no ordering needed
      if(((wl_max-wl_min)*100d0)/wl_max.lt.1d0) then 
      do m=1,tot_task
         do j=1,7
            klist(j,m)=kaux(j,m)
         enddo
      enddo
      endif
!
! simple sorting
!
      do m=1,tot_task
         do j=1,7
            klist(j,m)=-1
         enddo
      enddo
      m=0
      if(ga_nodeid().eq.0) then
         write(6,33) ' total no. of tasks ',tot_task
         write(6,33) ' total no. of tasks / no. procs ',
     &        tot_task/ga_nnodes()
         write(6,33) ' wl_min ',wl_min,(wl_min)**(1d0/6d0)
         write(6,33) ' wl_max ',wl_max,(wl_max)**(1d0/6d0)
      endif
   33 format(a32,i12,f14.1)

      nsplits=16
      do ii=nsplits,1,-1
         w_in=wl_min+((wl_max-wl_min)*(ii-1))/nsplits
         call ccsd_sillysort_cutensor(w_in,kaux,klist,tot_task,m)
       enddo
      call ccsd_sillysort_cutensor(0,kaux,klist,tot_task,m)
!     check that everybody has been invalidated
      do i=1,tot_task
         if(kaux(7,i).ge.0) then
            write(6,*) ' kaux element ',i,
     &           'not sorted. val = ',kaux(7,i)
            call errquit('neworder error ',0,0)
            endif
      enddo
!
!     figure out task with w lt w_thresh
!
!      w_thresh=w_thresh**6

      do i=1,tot_task
         if(klist(7,i).lt.w_thresh) then
            task_thresh=i
            goto 1973
         endif
      enddo
!
!      
 1973 continue
!baad
!     task_thresh must be gt ga_nnodes because of the static nxtask for the first task
!baad      task_thresh=max(task_thresh,ga_nnodes()+1) 
      if(ga_nodeid().eq.0) then
         if(task_thresh.lt.ga_nnodes()+1) then
            write(6,*) ' warning: CCSD(T) oversubscribed '
            write(6,'(a,i8)') '  warning: tip: use nproc <',task_thresh
         endif
         write(6,33) ' thresh for no. of tasks ',task_thresh
      endif
      end



      ! very simple sorting
      subroutine ccsd_sillysort_cutensor(val,kaux,klist,n,found)
      implicit none
      integer, intent(in) :: val
      integer, intent(inout) :: kaux(7,*)
      integer, intent(inout) :: klist(7,*)
      integer, intent(in) :: n
      integer, intent(inout) :: found
      integer :: j,m
      do m=1,n
         if(kaux(7,m).gt.val) then
            found=found+1
            do j=1,7
               klist(j,found)=kaux(j,m)
            enddo
            ! invalidate
            kaux(7,m)=-99
         endif
      enddo
      end


      subroutine ccsd_t_singles_cutensor(a_t3,
     &                                   f_t1,d_v2,
     &                                   k_t1_offset,k_v2_offset,
     &                                   t_h1b,t_h2b,t_h3b,
     &                                   t_p4b,t_p5b,t_p6b)
      implicit none
#include "global.fh"
#include "mafdecls.fh"
#include "sym.fh"
#include "errquit.fh"
#include "tce.fh"
      double precision, intent(inout) :: a_t3(:)
      double precision, intent(in) :: f_t1(:)
      integer, intent(in) :: k_t1_offset
      integer, intent(in) :: d_v2,k_v2_offset
      integer, intent(in) :: t_p4b, t_p5b, t_p6b, t_h1b, t_h2b, t_h3b
      integer :: h1b, h2b, h3b, p4b, p5b, p6b
      integer :: k_c_sort, l_c_sort
      integer :: h1b_1, h2b_2, h3b_2, p4b_1, p5b_2, p6b_2
      integer :: dima, dimb, dima_sort, dimb_sort
      integer :: k_a_sort, l_a_sort, k_b_sort, l_b_sort
      integer :: k_a, l_a, k_b, l_b
      integer :: a3(9,6)
      integer :: ia6,ja6

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
      do ia6=1,9
        p4b=a3(ia6,1)
        p5b=a3(ia6,2)
        p6b=a3(ia6,3)
        h1b=a3(ia6,4)
        h2b=a3(ia6,5)
        h3b=a3(ia6,6)
        if((p5b.le.p6b).and.(h2b.le.h3b).and.(p4b.ne.0)) then
        if ((.not.restricted).or.
     &       (int_mb(k_spin+p4b-1)+int_mb(k_spin+p5b-1)+
     &        int_mb(k_spin+p6b-1)+int_mb(k_spin+h1b-1)+
     &        int_mb(k_spin+h2b-1)+int_mb(k_spin+h3b-1).ne.12)) then
        if (int_mb(k_spin+p4b-1)+int_mb(k_spin+p5b-1)+
     &      int_mb(k_spin+p6b-1).eq. 
     &      int_mb(k_spin+h1b-1)+int_mb(k_spin+h2b-1)+
     &      int_mb(k_spin+h3b-1)) then
        if (ieor(int_mb(k_sym+p4b-1),ieor(int_mb(k_sym+p5b-1),
     &      ieor(int_mb(k_sym+p6b-1),ieor(int_mb(k_sym+h1b-1),
     &      ieor(int_mb(k_sym+h2b-1),int_mb(k_sym+h3b-1)))))) 
     &     .eq.ieor(irrep_v,irrep_t)) then
        if (int_mb(k_spin+p4b-1).eq.int_mb(k_spin+h1b-1)) then
        if (ieor(int_mb(k_sym+p4b-1),int_mb(k_sym+h1b-1)).eq.irrep_t) then
        CALL TCE_RESTRICTED_2(p4b,h1b,p4b_1,h1b_1)
        CALL TCE_RESTRICTED_4(p5b,p6b,h2b,h3b,p5b_2,p6b_2,h2b_2,h3b_2)
        dima_sort = int_mb(k_range+p4b-1) * int_mb(k_range+h1b-1)
        dima = dima_sort
        dimb_sort = int_mb(k_range+p5b-1) * int_mb(k_range+p6b-1) * 
     &              int_mb(k_range+h2b-1) * int_mb(k_range+h3b-1)
        dimb = dimb_sort
        if ((dima .gt. 0).and.(dimb .gt. 0)) then
        if (.not.MA_PUSH_GET(mt_dbl,dima,'noname',l_a_sort,k_a_sort)) 
     &             CALL ERRQUIT('ccsd_t_singles_1',1,MA_ERR)
        if (.not.MA_PUSH_GET(mt_dbl,dima,'noname',l_a,k_a)) 
     &             CALL ERRQUIT('ccsd_t_singles_1',2,MA_ERR)
        CALL GET_HASH_BLOCK_MA(f_t1,dbl_mb(k_a),dima,
     &                         int_mb(k_t1_offset),
     &                         (h1b_1-1+noab*(p4b_1-noab-1)))
        CALL TCE_SORT_2(dbl_mb(k_a),dbl_mb(k_a_sort),
     &                  int_mb(k_range+p4b-1),int_mb(k_range+h1b-1),
     &                  2,1,1.0d0)
        if (.not.MA_POP_STACK(l_a)) 
     &             CALL ERRQUIT('ccsd_t_singles_1',3,MA_ERR)
        if (.not.MA_PUSH_GET(mt_dbl,dimb,'noname',l_b_sort,k_b_sort)) 
     &             CALL ERRQUIT('ccsd_t_singles_1',4,MA_ERR)
        if(.not.intorb) then
        CALL GET_HASH_BLOCK(d_v2,dbl_mb(k_b_sort),dimb,
     &                      int_mb(k_v2_offset),
     &                      (h3b_2-1+(noab+nvab)*(h2b_2-1+(noab+nvab)*
     &                      (p6b_2-1+(noab+nvab)*(p5b_2-1)))))
        else
        CALL GET_HASH_BLOCK_I(d_v2,dbl_mb(k_b_sort),dimb,
     &                        int_mb(k_v2_offset),(h3b_2-1+(noab+nvab)*
     &                        (h2b_2-1+(noab+nvab)*(p6b_2-1+(noab+nvab)*
     &                        (p5b_2-1)))),h3b_2,h2b_2,p6b_2,p5b_2)
        end if
        if ((t_p4b.eq.p4b).and.(t_p5b.eq.p5b).and.(t_p6b.eq.p6b)
     &  .and.(t_h1b.eq.h1b).and.(t_h2b.eq.h2b).and.(t_h3b .eq. h3b)) then
        call sd_t_s1_1(int_mb(k_range+h3b-1),int_mb(k_range+h2b-1),
     &                 int_mb(k_range+h1b-1),int_mb(k_range+p6b-1),
     &                 int_mb(k_range+p5b-1),int_mb(k_range+p4b-1),
     &                 a_t3,dbl_mb(k_a_sort),dbl_mb(k_b_sort))
        
        end if
        if ((t_p4b.eq.p4b).and.(t_p5b.eq.p5b).and.(t_p6b.eq.p6b)
     &  .and.(t_h1b.eq.h2b).and.(t_h2b.eq.h1b).and.(t_h3b .eq. h3b)) then
        call sd_t_s1_2(int_mb(k_range+h3b-1),int_mb(k_range+h2b-1),
     &                 int_mb(k_range+h1b-1),int_mb(k_range+p6b-1),
     &                 int_mb(k_range+p5b-1),int_mb(k_range+p4b-1),
     &                 a_t3,dbl_mb(k_a_sort),dbl_mb(k_b_sort))
        
        end if
        if ((t_p4b.eq.p4b).and.(t_p5b.eq.p5b).and.(t_p6b.eq.p6b)
     &  .and.(t_h1b.eq.h2b).and.(t_h2b.eq.h3b).and.(t_h3b .eq. h1b)) then
        call sd_t_s1_3(int_mb(k_range+h3b-1),int_mb(k_range+h2b-1),
     &                 int_mb(k_range+h1b-1),int_mb(k_range+p6b-1),
     &                 int_mb(k_range+p5b-1),int_mb(k_range+p4b-1),
     &                 a_t3,dbl_mb(k_a_sort),dbl_mb(k_b_sort))
        
        end if
        if ((t_p4b.eq.p5b).and.(t_p5b.eq.p4b).and.(t_p6b.eq.p6b)
     &  .and.(t_h1b.eq.h1b).and.(t_h2b.eq.h2b).and.(t_h3b .eq. h3b)) then
        call sd_t_s1_4(int_mb(k_range+h3b-1),int_mb(k_range+h2b-1),
     &                 int_mb(k_range+h1b-1),int_mb(k_range+p6b-1),
     &                 int_mb(k_range+p5b-1),int_mb(k_range+p4b-1),
     &                 a_t3,dbl_mb(k_a_sort),dbl_mb(k_b_sort))
        
        end if
        if ((t_p4b.eq.p5b).and.(t_p5b.eq.p4b).and.(t_p6b.eq.p6b)
     &  .and.(t_h1b.eq.h2b).and.(t_h2b.eq.h1b).and.(t_h3b .eq. h3b)) then
        call sd_t_s1_5(int_mb(k_range+h3b-1),int_mb(k_range+h2b-1),
     &                 int_mb(k_range+h1b-1),int_mb(k_range+p6b-1),
     &                 int_mb(k_range+p5b-1),int_mb(k_range+p4b-1),
     &                 a_t3,dbl_mb(k_a_sort),dbl_mb(k_b_sort))
        
        end if
        if ((t_p4b.eq.p5b).and.(t_p5b.eq.p4b).and.(t_p6b.eq.p6b)
     &  .and.(t_h1b.eq.h2b).and.(t_h2b.eq.h3b).and.(t_h3b .eq. h1b)) then
        call sd_t_s1_6(int_mb(k_range+h3b-1),int_mb(k_range+h2b-1),
     &                 int_mb(k_range+h1b-1),int_mb(k_range+p6b-1),
     &                 int_mb(k_range+p5b-1),int_mb(k_range+p4b-1),
     &                 a_t3,dbl_mb(k_a_sort),dbl_mb(k_b_sort))
        
        end if
        if ((t_p4b.eq.p5b).and.(t_p5b.eq.p6b).and.(t_p6b.eq.p4b)
     &  .and.(t_h1b.eq.h1b).and.(t_h2b.eq.h2b).and.(t_h3b .eq. h3b)) then
        call sd_t_s1_7(int_mb(k_range+h3b-1),int_mb(k_range+h2b-1),
     &                 int_mb(k_range+h1b-1),int_mb(k_range+p6b-1),
     &                 int_mb(k_range+p5b-1),int_mb(k_range+p4b-1),
     &                 a_t3,dbl_mb(k_a_sort),dbl_mb(k_b_sort))
        
        end if
        if ((t_p4b.eq.p5b).and.(t_p5b.eq.p6b).and.(t_p6b.eq.p4b)
     &  .and.(t_h1b.eq.h2b).and.(t_h2b.eq.h1b).and.(t_h3b .eq. h3b)) then
        call sd_t_s1_8(int_mb(k_range+h3b-1),int_mb(k_range+h2b-1),
     &                 int_mb(k_range+h1b-1),int_mb(k_range+p6b-1),
     &                 int_mb(k_range+p5b-1),int_mb(k_range+p4b-1),
     &                 a_t3,dbl_mb(k_a_sort),dbl_mb(k_b_sort))
        
        end if
        if ((t_p4b.eq.p5b).and.(t_p5b.eq.p6b).and.(t_p6b.eq.p4b)
     &  .and.(t_h1b.eq.h2b).and.(t_h2b.eq.h3b).and.(t_h3b .eq. h1b)) then
        call sd_t_s1_9(int_mb(k_range+h3b-1),int_mb(k_range+h2b-1),
     &                 int_mb(k_range+h1b-1),int_mb(k_range+p6b-1),
     &                 int_mb(k_range+p5b-1),int_mb(k_range+p4b-1),
     &                 a_t3,dbl_mb(k_a_sort),dbl_mb(k_b_sort))
        
        end if
        if (.not.MA_POP_STACK(l_b_sort)) 
     &         CALL ERRQUIT('ccsd_t_singles_1',7,MA_ERR)
        if (.not.MA_POP_STACK(l_a_sort)) 
     &         CALL ERRQUIT('ccsd_t_singles_1',8,MA_ERR)
!       
        end if
        end if
        end if
!       
        end if
        end if
        end if
        end if
      end do
      end
      
      
      subroutine ccsd_t_doubles_cutensor(f_t3,d_t2,d_v2,
     &           k_t2_offset,k_v2_offset,
     &           t_h1b,t_h2b,t_h3b,t_p4b,t_p5b,t_p6b)
!     i0 ( p4 p5 p6 h1 h2 h3 )_vt + = -1 * P( 9 ) * Sum ( h7 ) * t ( p4 p5 h1 h7 )_t * v ( h7 p6 h2 h3 )_v
!     i0 ( p4 p5 p6 h1 h2 h3 )_vt + = -1 * P( 9 ) * Sum ( p7 ) * t ( p4 p7 h1 h2 )_t * v ( p5 p6 h3 p7 )_v
      implicit none
#include "global.fh"
#include "mafdecls.fh"
#include "util.fh"
#include "errquit.fh"
#include "tce.fh"
      double precision, intent(inout), managed :: f_t3(:)
      integer, intent(in) :: t_p4b, t_p5b, t_p6b, t_h1b, t_h2b, t_h3b
      integer :: d_t2, k_t2_offset
      integer :: d_v2, k_v2_offset
      double precision, allocatable, managed :: d_t2sub(:)
      double precision, allocatable, managed :: d_v2sub(:)
      double precision, allocatable, managed :: d_scratch(:)
      integer :: alloc_error
      integer :: range_p4,range_h1
      integer :: maxp,maxh
      integer :: t2size,v2size
      call tce_maxph(maxp,maxh)
      t2size=(maxp**2)*(maxh**2)
      v2size=max((maxp)*(maxh**3),(maxp**3)*(maxh))
      allocate( d_t2sub(1:t2size), d_v2sub(1:v2size),
     &          d_scratch(1:max(t2size,v2size)), stat=alloc_error)
      if (alloc_error.ne.0) then
        call errquit('ccsd_t_doubles_cutensor',alloc_error,MA_ERR)
      endif
      call ccsd_t_doubles_cutensor_12(d_t2,
     &     k_t2_offset,d_v2,k_v2_offset,
     &     f_t3,t_p4b,t_p5b,t_p6b,t_h1b,t_h2b,t_h3b,
     &     d_v2sub,d_t2sub,d_scratch)
      deallocate( d_t2sub, d_v2sub, d_scratch, stat=alloc_error)
      if (alloc_error.ne.0) then
        call errquit('ccsd_t_doubles_cutensor',alloc_error,MA_ERR)
      endif
      end

      ! compute v2sub and t2sub max length
      subroutine tce_maxph(p,h)
      implicit none
#include "mafdecls.fh"
#include "tce.fh"
      integer, intent(out) :: p,h
      integer :: i
      h=0
      do i = 1,noab
         h = max(h,int_mb(k_range+i-1))
      enddo
      p=0
      do i = noab+1,noab+nvab
         p = max(p,int_mb(k_range+i-1))
      enddo
      end

      subroutine ccsd_t_doubles_cutensor_12(d_a,k_a_offset,
     &                                      d_b,k_b_offset,
     &                                      triplesx,
     &                                      t_p4b,t_p5b,t_p6b,
     &                                      t_h1b,t_h2b,t_h3b,
     &                                      v2sub,t2sub,scratch)
!     i0 ( p4 p5 p6 h1 h2 h3 )_vt + = -1 * P( 9 ) * Sum ( h7 ) * t ( p4 p5 h1 h7 )_t * v ( h7 p6 h2 h3 )_v
      use iso_c_binding
      use cudafor
      use cutensorex
      implicit none
#include "global.fh"
#include "mafdecls.fh"
#include "sym.fh"
#include "errquit.fh"
#include "tce.fh"
#include "util.fh"
      integer, intent(in) :: d_a, k_a_offset
      integer, intent(in) :: d_b, k_b_offset
      double precision, intent(inout), managed :: triplesx(:)
      integer, intent(in) :: t_p4b, t_p5b, t_p6b, t_h1b, t_h2b, t_h3b
      double precision, intent(in), managed :: t2sub(:),v2sub(:)
      double precision, intent(in), managed :: scratch(:)
      integer :: h1b, h2b, h3b, p4b, p5b, p6b, h7b,p7b
      integer :: p5b_2,p7b_1,h2b_1,p7b_2
      integer :: p4b_1, p5b_1, h1b_1, h7b_1
      integer :: p6b_2, h7b_2, h2b_2, h3b_2
      integer :: dima, dimb, dima_sort, dimb_sort
      integer :: k_c_sort, l_c_sort
      integer :: k_a, l_a
      integer :: k_b, l_b
      integer :: a3(9,6)
      integer :: ia6,ja6
      integer :: range_h1b, range_h2b, range_h3b, range_p7b
      integer :: range_p4b, range_p5b, range_p6b, range_h7b
      integer :: k_a_srt_len
      integer :: k_b_srt_len
      logical :: tce_hashnsort,tce_hashnsort_2, lresult
      external tce_hashnsort, tce_hashnsort_2
      integer :: n_h7b
      integer :: n_p7b
      integer :: h7b_in,p7b_in

      type(cutensorStatus) :: cut_status
      type(cutensorDescriptor) :: dT2, dV2, dT3
      type(cutensorContractionDescriptor) :: dX
      type(cutensorContractionPlan) :: pX
      double precision :: alpha
      double precision, parameter :: beta = 1.0d0

      integer(8), dimension(4) :: eT2 ! extent
      integer(8), dimension(4) :: sT2 ! stride
      integer(4), dimension(4) :: mT2 ! modes
      integer(8), dimension(4) :: eV2 ! extent
      integer(8), dimension(4) :: sV2 ! stride
      integer(4), dimension(4) :: mV2 ! modes
      integer(8), dimension(6) :: eT3
      integer(4), dimension(6) :: mT3
      integer(4) :: aT2, aV2, aT3

      integer :: alloc_error
      integer(4) :: cuda_err
      integer(8) :: workspaceSize
      type(c_devptr) :: workspace

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
      do ia6=1,9
      p4b=a3(ia6,1)
      p5b=a3(ia6,2)
      p6b=a3(ia6,3)
      h1b=a3(ia6,4)
      h2b=a3(ia6,5)
      h3b=a3(ia6,6)
      if(((p4b.le.p5b).and.(h2b.le.h3b).and.(p4b.ne.0)).and.
     &   ((.not.restricted).or.
     &    (int_mb(k_spin+p4b-1)+int_mb(k_spin+p5b-1)+
     &     int_mb(k_spin+p6b-1)+int_mb(k_spin+h1b-1)+
     &     int_mb(k_spin+h2b-1)+int_mb(k_spin+h3b-1).ne.12))
     &   .and.
     &   (int_mb(k_spin+p4b-1)+int_mb(k_spin+p5b-1)+
     &    int_mb(k_spin+p6b-1) .eq. int_mb(k_spin+h1b-1)+
     &    int_mb(k_spin+h2b-1)+int_mb(k_spin+h3b-1)).and.
     &   (ieor(int_mb(k_sym+p4b-1),ieor(int_mb(k_sym+p5b-1),
     &    ieor(int_mb(k_sym+p6b-1),ieor(int_mb(k_sym+h1b-1),
     &    ieor(int_mb(k_sym+h2b-1),int_mb(k_sym+h3b-1))))))
     &    .eq. ieor(irrep_v,irrep_t))) then
      range_h3b = int_mb(k_range+h3b-1)
      range_h2b = int_mb(k_range+h2b-1)
      range_h1b = int_mb(k_range+h1b-1)
      range_p6b = int_mb(k_range+p6b-1)
      range_p5b = int_mb(k_range+p5b-1)
      range_p4b = int_mb(k_range+p4b-1)
!     prep. work
!     figure out  cycles to work on
      n_h7b=0
      do h7b = 1,noab
         if(tce_hashnsort(.true., d_a, d_b,
     &                    k_a_offset,k_b_offset,
     &                    p4b,p5b,h1b,h7b, p6b,h2b,h3b,
     &                    t2sub,v2sub)) then
            n_h7b=n_h7b+1
         endif
      enddo

      if(n_h7b.gt.0) then

      ! all d1 contractions use
      ! t2sub(h7d,p4d,p5d,h1d)
      ! v2sub(h3d,h2d,p6d,h7d)

         do h7b_in=ga_nodeid()+1,ga_nodeid()+noab
            h7b=mod(h7b_in,noab)+1
            if(tce_hashnsort(.false., d_a, d_b,
     &                       k_a_offset,k_b_offset,
     &                       p4b,p5b,h1b,h7b,p6b,h2b,h3b,
     &                       t2sub,v2sub)) then
                range_h7b = int_mb(k_range+h7b-1)

                if ((t_p4b .eq. p4b) .and. (t_p5b .eq. p5b) .and. 
     &              (t_p6b .eq. p6b) .and. (t_h1b .eq. h1b) .and.
     &              (t_h2b .eq. h2b) .and. (t_h3b .eq. h3b)) then

      ! t2sub(h7d,p4d,p5d,h1d)
      eT2 = [range_h7b,range_p4b,range_p5b,range_h1b]
      cut_status = cutensorInitTensorDescriptor(cut_handle,dT2,
     &                               four,eT2,c_null_ptr,
     &                               cut_double,
     &                               CUTENSOR_OP_IDENTITY)
      if (cut_status .ne. CUTENSOR_STATUS_SUCCESS) then
        call errquit('cutensorInitTensorDescriptor T2',
     &               int(cut_status%stat),UNKNOWN_ERR)
      endif
      cut_status = cutensorGetAlignmentRequirement(cut_handle, 
     &                        t2sub, dT2, aT2)
      if (cut_status .ne. CUTENSOR_STATUS_SUCCESS) then
        call errquit('cutensorGetAlignmentRequirement T2',
     &               int(cut_status%stat),UNKNOWN_ERR)
      endif

      ! v2sub(h3d,h2d,p6d,h7d)
      eV2 = [range_h3b,range_h2b,range_p6b,range_h7b] ! 3267
      cut_status = cutensorInitTensorDescriptor(cut_handle,dV2,
     &                               four,eV2,c_null_ptr,
     &                               cut_double,
     &                               CUTENSOR_OP_IDENTITY)
      if (cut_status .ne. CUTENSOR_STATUS_SUCCESS) then
        call errquit('cutensorInitTensorDescriptor V2',
     &               int(cut_status%stat),UNKNOWN_ERR)
      endif
      cut_status = cutensorGetAlignmentRequirement(cut_handle, 
     &                        v2sub, dV2, aV2)
      if (cut_status .ne. CUTENSOR_STATUS_SUCCESS) then
        call errquit('cutensorGetAlignmentRequirement V2',
     &               int(cut_status%stat),UNKNOWN_ERR)
      endif

      ! no stride here
      eT3 = [range_h3b,range_h2b,range_h1b,range_p6b,range_p5b,range_p4b] ! 321654
      cut_status = cutensorInitTensorDescriptor(cut_handle,dT3,
     &                               six,eT3,c_null_ptr,
     &                               cut_double,
     &                               CUTENSOR_OP_IDENTITY)
      if (cut_status .ne. CUTENSOR_STATUS_SUCCESS) then
        call errquit('cutensorInitTensorDescriptor T3',
     &               int(cut_status%stat),UNKNOWN_ERR)
      endif
      cut_status = cutensorGetAlignmentRequirement(cut_handle, 
     &                                 triplesx, dT3, aT3)
      if (cut_status .ne. CUTENSOR_STATUS_SUCCESS) then
        call errquit('cutensorGetAlignmentRequirement T3',
     &               int(cut_status%stat),UNKNOWN_ERR)
      endif

      mT2 = [7,4,5,1]
      mV2 = [3,2,6,7]
      mT3 = [3,2,1,6,5,4]
      cut_status = cutensorInitContractionDescriptor(cut_handle,dX,
     &                         dT2,mT2,aT2, dV2,mV2,aV2,
     &                         dT3,mT3,aT3, dT3,mT3,aT3, cut_compF64)
      if (cut_status .ne. CUTENSOR_STATUS_SUCCESS) then
        call errquit('cutensorInitContractionDescriptor D1_1',
     &               int(cut_status%stat),UNKNOWN_ERR)
      endif
      workspaceSize = 0
      cut_status = cutensorContractionGetWorkspace(cut_handle,
     &                                dX, cut_find,
     &                                CUTENSOR_WORKSPACE_RECOMMENDED,
     &                                workspaceSize)
      if (cut_status .ne. CUTENSOR_STATUS_SUCCESS) then
        call errquit('cutensorContractionGetWorkspace D1_1',
     &               int(cut_status%stat),UNKNOWN_ERR)
      endif
#if ASYNC
      alloc_error = cudaMallocAsync(workspace, workspaceSize, stream)
      if (alloc_error.ne.0) then
        call errquit('cudaMallocAsync D1_1',alloc_error,MA_ERR)
      endif
      cuda_err = cudaStreamSynchronize( stream )
      if (cuda_err.ne.0) then
        write(6,*) cudaGetErrorString(cuda_err)
        call errquit('cudaStreamSynchronize',cuda_err,UNKNOWN_ERR)
      endif
#else
      alloc_error = cudaMalloc(workspace, workspaceSize)
      if (alloc_error.ne.0) then
        call errquit('cudaMalloc D1_1',alloc_error,MA_ERR)
      endif
#endif
      cut_status = cutensorInitContractionPlan(cut_handle, pX, dX,
     &                         cut_find, workspaceSize)
      if (cut_status .ne. CUTENSOR_STATUS_SUCCESS) then
        call errquit('cutensorInitContractionPlan D1_1',
     &               int(cut_status%stat),UNKNOWN_ERR)
      endif
      print*,'cutensorContraction actually called'
      alpha = -1.0d0
      cut_status = cutensorContraction(cut_handle, pX,
     &                     alpha, t2sub, v2sub,
     &                     beta, triplesx, triplesx,
     &                     workspace, workspaceSize, stream)
      if (cut_status .ne. CUTENSOR_STATUS_SUCCESS) then
        call errquit('cutensorContraction D1_1',
     &               int(cut_status%stat),UNKNOWN_ERR)
      endif
      cuda_err = cudaStreamSynchronize( stream )
      if (cuda_err.ne.0) then
        write(6,*) cudaGetErrorString(cuda_err)
        call errquit('cudaStreamSynchronize',cuda_err,UNKNOWN_ERR)
      endif
#if ASYNC
      alloc_error = cudaFreeAsync(workspace, stream)
      if (alloc_error.ne.0) then
        call errquit('cudaFreeAsync workspace D1_1',
     &               alloc_error,MA_ERR)
      endif
      cuda_err = cudaStreamSynchronize( stream )
      if (cuda_err.ne.0) then
        write(6,*) cudaGetErrorString(cuda_err)
        call errquit('cudaStreamSynchronize',cuda_err,UNKNOWN_ERR)
      endif
#else
      alloc_error = cudaFree(workspace)
      if (alloc_error.ne.0) then
        call errquit('cudaFree workspace D1_1',alloc_error,MA_ERR)
      endif
#endif
!                    call cutensor_t_d1_1(range_h3b,range_h2b,
!     &                                   range_h1b,range_p6b,
!     &                                   range_p5b,range_p4b,
!     &                                   range_h7b,
!     &                                   triplesx,t2sub,v2sub)
                end if
                if ((t_p4b .eq. p4b) .and. (t_p5b .eq. p5b) .and. 
     &              (t_p6b .eq. p6b) .and. (t_h1b .eq. h2b) .and. 
     &              (t_h2b .eq. h1b) .and. (t_h3b .eq. h3b)) then
                    call sd_t_d1_2(range_h3b,range_h2b,
     &                                   range_h1b,range_p6b,
     &                                   range_p5b,range_p4b,
     &                                   range_h7b,
     &                                   triplesx,t2sub,v2sub,scratch)
                end if
                if ((t_p4b .eq. p4b) .and. (t_p5b .eq. p5b) .and.
     &              (t_p6b .eq. p6b) .and. (t_h1b .eq. h2b) .and. 
     &              (t_h2b .eq. h3b) .and. (t_h3b .eq. h1b)) then
                    call sd_t_d1_3(range_h3b,range_h2b,
     &                                   range_h1b,range_p6b,
     &                                   range_p5b,range_p4b,
     &                                   range_h7b,
     &                                   triplesx,t2sub,v2sub,scratch)
                end if
                if ((t_p4b .eq. p6b) .and. (t_p5b .eq. p4b) .and.
     &              (t_p6b .eq. p5b) .and. (t_h1b .eq. h1b) .and. 
     &              (t_h2b .eq. h2b) .and. (t_h3b .eq. h3b)) then
                    call sd_t_d1_4(range_h3b,range_h2b,
     &                                   range_h1b,range_p6b,
     &                                   range_p5b,range_p4b,
     &                                   range_h7b,
     &                                   triplesx,t2sub,v2sub)
                end if
                if ((t_p4b .eq. p6b) .and. (t_p5b .eq. p4b) .and.
     &              (t_p6b .eq. p5b) .and. (t_h1b .eq. h2b) .and.
     &              (t_h2b .eq. h1b) .and. (t_h3b .eq. h3b)) then
                    call sd_t_d1_5(range_h3b,range_h2b,
     &                                   range_h1b,range_p6b,
     &                                   range_p5b,range_p4b,
     &                                   range_h7b,
     &                                   triplesx,t2sub,v2sub,scratch)
                end if
                if ((t_p4b .eq. p6b) .and. (t_p5b .eq. p4b) .and.
     &              (t_p6b .eq. p5b) .and. (t_h1b .eq. h2b) .and. 
     &              (t_h2b .eq. h3b) .and. (t_h3b .eq. h1b)) then
                    call sd_t_d1_6(range_h3b,range_h2b,
     &                                   range_h1b,range_p6b,
     &                                   range_p5b,range_p4b,
     &                                   range_h7b,
     &                                   triplesx,t2sub,v2sub,scratch)
                end if
                if ((t_p4b .eq. p4b) .and. (t_p5b .eq. p6b) .and.
     &              (t_p6b .eq. p5b) .and. (t_h1b .eq. h1b) .and. 
     &              (t_h2b .eq. h2b) .and. (t_h3b .eq. h3b)) then
                    call sd_t_d1_7(range_h3b,range_h2b,
     &                                   range_h1b,range_p6b,
     &                                   range_p5b,range_p4b,
     &                                   range_h7b,
     &                                   triplesx,t2sub,v2sub)
                end if
                if ((t_p4b .eq. p4b) .and. (t_p5b .eq. p6b) .and.
     &              (t_p6b .eq. p5b) .and. (t_h1b .eq. h2b) .and.
     &              (t_h2b .eq. h1b) .and. (t_h3b .eq. h3b)) then
                    call sd_t_d1_8(range_h3b,range_h2b,
     &                                   range_h1b,range_p6b,
     &                                   range_p5b,range_p4b,
     &                                   range_h7b,
     &                                   triplesx,t2sub,v2sub,scratch)
                end if
                if ((t_p4b .eq. p4b) .and. (t_p5b .eq. p6b) .and. 
     &              (t_p6b .eq. p5b) .and. (t_h1b .eq. h2b) .and. 
     &              (t_h2b .eq. h3b) .and. (t_h3b .eq. h1b)) then
                    call sd_t_d1_9(range_h3b,range_h2b,
     &                                   range_h1b,range_p6b,
     &                                   range_p5b,range_p4b,
     &                                   range_h7b,
     &                                   triplesx,t2sub,v2sub,scratch)
                end if
            endif ! hashnsort
         end do !h7b loop
      endif ! n_h7b.ne.0
      end if
      end do !ia6 loop
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
     &     .and.(a3(ia6,3).eq.a3(ja6,3)).and.(a3(ia6,4).eq.a3(ja6,4))
     &     .and.(a3(ia6,5).eq.a3(ja6,5)).and.(a3(ia6,6).eq.a3(ja6,6)))
     &     then
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
      do ia6=1,9
      p4b=a3(ia6,1)
      p5b=a3(ia6,2)
      p6b=a3(ia6,3)
      h1b=a3(ia6,4)
      h2b=a3(ia6,5)
      h3b=a3(ia6,6)
      if(((p5b.le.p6b).and.(h1b.le.h2b).and.(p4b.ne.0)).and.
     &   ((.not.restricted).or.
     &    (int_mb(k_spin+p4b-1)+int_mb(k_spin+p5b-1)+
     &     int_mb(k_spin+p6b-1)+int_mb(k_spin+h1b-1)+
     &     int_mb(k_spin+h2b-1)+int_mb(k_spin+h3b-1).ne.12))
     &   .and.
     &   (int_mb(k_spin+p4b-1)+int_mb(k_spin+p5b-1)+
     &    int_mb(k_spin+p6b-1) .eq. int_mb(k_spin+h1b-1)+
     &    int_mb(k_spin+h2b-1)+int_mb(k_spin+h3b-1)).and.
     &   (ieor(int_mb(k_sym+p4b-1),ieor(int_mb(k_sym+p5b-1),
     &    ieor(int_mb(k_sym+p6b-1),ieor(int_mb(k_sym+h1b-1),
     &    ieor(int_mb(k_sym+h2b-1),int_mb(k_sym+h3b-1)))))) 
     &    .eq. ieor(irrep_v,irrep_t))) then
      range_h3b = int_mb(k_range+h3b-1)
      range_h2b = int_mb(k_range+h2b-1)
      range_h1b = int_mb(k_range+h1b-1)
      range_p6b = int_mb(k_range+p6b-1)
      range_p5b = int_mb(k_range+p5b-1)
      range_p4b = int_mb(k_range+p4b-1)
!     figure out  cycles to work on
      n_p7b=0
      do p7b = noab+1,noab+nvab
         if(tce_hashnsort_2(.true.,
     &        d_a, d_b,
     &        k_a_offset,k_b_offset,
     &        p4b,p7b,h1b,h2b,
     &        p5b,p6b,h3b,
     &        t2sub,v2sub
     &        )) then
            n_p7b=n_p7b+1
         endif
      enddo

      if(n_p7b.ne.0) then

      ! all d2 contractions use
      ! t2sub(p7d,p4d,h1d,h2d)
      ! v2sub(p7d,h3d,p6d,p5d)

      do p7b_in= ga_nodeid()+1,nvab+ga_nodeid()
         p7b=mod(p7b_in,nvab)+noab+1
         if(tce_hashnsort_2(.false., d_a, d_b,
     &                      k_a_offset,k_b_offset,
     &                      p4b,p7b,h1b,h2b,p5b,p6b,h3b,
     &                      t2sub,v2sub)) then
            range_p7b = int_mb(k_range+p7b-1)
            if ((t_p4b .eq. p4b) .and. (t_p5b .eq. p5b) .and. 
     &          (t_p6b .eq. p6b) .and. (t_h1b .eq. h1b) .and.
     &          (t_h2b .eq. h2b) .and. (t_h3b .eq .h3b)) then
                call sd_t_d2_1(range_h3b,range_h2b,
     &                               range_h1b,range_p6b,
     &                               range_p5b,range_p4b,
     &                               range_p7b,
     &                               triplesx,t2sub,v2sub,scratch)
            end if
            if ((t_p4b .eq. p4b) .and. (t_p5b .eq. p5b) .and. 
     &          (t_p6b .eq. p6b) .and. (t_h1b .eq. h3b) .and.
     &          (t_h2b .eq. h1b) .and. (t_h3b .eq. h2b)) then
                call sd_t_d2_2(range_h3b,range_h2b,
     &                               range_h1b,range_p6b,
     &                               range_p5b,range_p4b,
     &                               range_p7b,
     &                               triplesx,t2sub,v2sub,scratch)
            end if
            if ((t_p4b .eq. p4b) .and. (t_p5b .eq. p5b) .and. 
     &          (t_p6b .eq. p6b) .and. (t_h1b .eq. h1b) .and.
     &          (t_h2b .eq. h3b) .and. (t_h3b .eq. h2b)) then
                call sd_t_d2_3(range_h3b,range_h2b,
     &                               range_h1b,range_p6b,
     &                               range_p5b,range_p4b,
     &                               range_p7b,
     &                               triplesx,t2sub,v2sub)
            end if
            if ((t_p4b .eq. p5b) .and. (t_p5b .eq. p4b) .and. 
     &          (t_p6b .eq. p6b) .and. (t_h1b .eq. h1b) .and.
     &          (t_h2b .eq. h2b) .and. (t_h3b .eq .h3b)) then
                call sd_t_d2_4(range_h3b,range_h2b,
     &                               range_h1b,range_p6b,
     &                               range_p5b,range_p4b,
     &                               range_p7b,
     &                               triplesx,t2sub,v2sub,scratch)
            end if
            if ((t_p4b .eq. p5b) .and. (t_p5b .eq. p4b) .and. 
     &          (t_p6b .eq. p6b) .and. (t_h1b .eq. h3b) .and.
     &          (t_h2b .eq. h1b) .and. (t_h3b .eq .h2b)) then
           
                call sd_t_d2_5(range_h3b,range_h2b,
     &                               range_h1b,range_p6b,
     &                               range_p5b,range_p4b,
     &                               range_p7b,
     &                               triplesx,t2sub,v2sub,scratch)
            end if
            if ((t_p4b .eq. p5b) .and. (t_p5b .eq. p4b) .and. 
     &          (t_p6b .eq. p6b) .and. (t_h1b .eq. h1b) .and.
     &          (t_h2b .eq. h3b) .and. (t_h3b .eq .h2b)) then
                call sd_t_d2_6(range_h3b,range_h2b,
     &                               range_h1b,range_p6b,
     &                               range_p5b,range_p4b,
     &                               range_p7b,
     &                               triplesx,t2sub,v2sub,scratch)
            end if
            if ((t_p4b .eq. p5b) .and. (t_p5b .eq. p6b) .and. 
     &          (t_p6b .eq. p4b) .and. (t_h1b .eq. h1b) .and.
     &          (t_h2b .eq. h2b) .and. (t_h3b .eq .h3b)) then
                call sd_t_d2_7(range_h3b,range_h2b,
     &                               range_h1b,range_p6b,
     &                               range_p5b,range_p4b,
     &                               range_p7b,
     &                               triplesx,t2sub,v2sub)
            end if
            if ((t_p4b .eq. p5b) .and. (t_p5b .eq. p6b) .and. 
     &          (t_p6b .eq. p4b) .and. (t_h1b .eq. h3b) .and.
     &          (t_h2b .eq. h1b) .and. (t_h3b .eq .h2b)) then
                call sd_t_d2_8(range_h3b,range_h2b,
     &                               range_h1b,range_p6b,
     &                               range_p5b,range_p4b,
     &                               range_p7b,
     &                               triplesx,t2sub,v2sub,scratch)
                     end if
            if ((t_p4b .eq. p5b) .and. (t_p5b .eq. p6b) .and. 
     &          (t_p6b .eq. p4b) .and. (t_h1b .eq. h1b) .and.
     &          (t_h2b .eq. h3b) .and. (t_h3b .eq .h2b)) then
                call sd_t_d2_9(range_h3b,range_h2b,
     &                               range_h1b,range_p6b,
     &                               range_p5b,range_p4b,
     &                               range_p7b,
     &                               triplesx,t2sub,v2sub,scratch)
            end if
         endif ! hashnsort
      end do ! p7b loop

      endif ! n_p7b.ne.0
!
!
      end if
      end do ! ia6 loop
      end

      end module
