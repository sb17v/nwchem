      module tce_ccsd_t_cutensor_nodts

      contains

      subroutine ccsd_t_cutensor(d_t1,k_t1_offset,d_t2,k_t2_offset,
     &            d_v2,k_v2_offset,energy1,energy2,size_t1)
      implicit none
#include "global.fh"
#include "mafdecls.fh"
#include "util.fh"
#include "errquit.fh"
#include "tce.fh"
#include "tce_main.fh"
#include "ccsd_t_ps.fh"
      integer :: d_t1
      integer :: k_t1_offset
      integer :: d_t2
      integer :: k_t2_offset
      integer :: d_v2
      integer :: k_v2_offset
      integer :: t_h1b, t_h1
      integer :: t_h2b, t_h2
      integer :: t_h3b, t_h3
      integer :: t_p4b, t_p4
      integer :: t_p5b, t_p5
      integer :: t_p6b, t_p6
      double precision, allocatable :: f_singles(:),f_doubles(:)
      integer :: alloc_error
      integer :: size_t3,i
      integer :: nxtask0
      integer :: next
      integer :: nprocs
      integer :: count
      integer :: offset_p4,offset_p5,offset_p6
      integer :: offset_h1,offset_h2,offset_h3
      integer :: range_p4,range_p5,range_p6
      integer :: range_h1,range_h2,range_h3
! - T1/X1 LOCALIZATION -------------------
      integer :: l_t1_local,k_t1_local
      integer :: size_t1
! ---------------------------------------
      double precision :: energy(2)
      double precision :: energy1,energy2
      integer :: k_aux,k_list,l_aux,l_list
      integer :: tot_task,w_thresh,task_thresh
      integer :: k,iptr,k_grain
      double precision :: wall
      integer :: tasks_skip
      integer :: n_progr,pct_progr
      parameter(n_progr=20)
      logical :: i_progr(n_progr)
      external nxtask0
!
! - T1/X1 LOCALIZATION ----------
!    opening l_t1_local and l_x1_local
        if (.not.MA_PUSH_GET(mt_dbl,size_t1,'t1_local',
     &      l_t1_local,k_t1_local))
     &      call errquit('ccsd_t: t1_local size=',size_t1,MA_ERR)
        call ma_zero(dbl_mb(k_t1_local),size_t1)
!    copy d_t1 ==> l_t1_local
!cx        call ga_get(d_t1,1,size_t1,1,1,dbl_mb(k_t1_local),1)
      call get_block(d_t1,dbl_mb(k_t1_local),size_t1,0)
! -------------------------------
!
      nprocs = GA_NNODES()
      energy(1)=0.0d0
      energy(2)=0.0d0
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
      if (.not.ma_push_get(mt_int,7*tot_task,"list.task",
     &  l_list,k_list)) call errquit("k_list",1,MA_ERR)
      if (.not.ma_push_get(mt_int,7*tot_task,"auxtask",
     &  l_aux,k_aux)) call errquit("k_aux",2,MA_ERR)
!
!     get first task with weight lt ? 8
!
!      w_thresh=10
      w_thresh=0
      w_thresh=w_thresh**6
      call ccsd_t_neword_cutensor(tot_task, w_thresh,task_thresh,
     &     restricted,noab,nvab,
     &     int_mb(k_spin),int_mb(k_sym),
     &     int_mb(k_range),
     &     int_mb(k_aux),int_mb(k_list))

      if (.not.MA_POP_STACK(l_aux))
     &     call errquit('ordering',3,MA_ERR)
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
!
         if (next.eq.count) then
            call ccsd_t_loop_cutensor(k,energy1,energy2,
     &              int_mb(k_list),int_mb(k_range),int_mb(k_offset),
     &              f_singles,f_doubles,
     &              k_t1_local,k_t1_offset,
     &              d_t2,d_v2,k_t2_offset,k_v2_offset,
     &              restricted,k_evl_sorted,size_t3)
!
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
     &                 int_mb(k_list),int_mb(k_range),int_mb(k_offset),
     &                 f_singles,f_doubles,
     &                 k_t1_local,k_t1_offset,
     &                 d_t2,d_v2,k_t2_offset,k_v2_offset,
     &                 restricted,k_evl_sorted,size_t3)
!
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

      if (.not.MA_POP_STACK(l_list))
     &     call errquit('ordering',3,MA_ERR)
      deallocate( f_doubles, stat=alloc_error)
      if (alloc_error.ne.0) then
        call errquit('ccsd_t doubles',3,MA_ERR)
      endif
      deallocate( f_singles, stat=alloc_error)
      if (alloc_error.ne.0) then
        call errquit('ccsd_t singles',4,MA_ERR)
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
         if(.not.MA_POP_STACK(l_t1_local))
     &      call errquit('ccsd_t: l_t1_local',4,MA_ERR)
! ---------------------------
      end

      subroutine ccsd_t_dot_cutensor(a_singles, a_doubles,
     &                      restricted,
     &                      h1b,h2b,h3b,p4b,p5b,p6b,
     &                      o_h1,o_h2,o_h3,
     &                      o_p4,o_p5,o_p6,
     &                      r_h1,r_h2,r_h3,
     &                      r_p4,r_p5,r_p6,
     &                      energy1,energy2)
      implicit none
      integer :: h1b, h2b, h3b, p4b, p5b, p6b
      integer :: r_h1,r_h2,r_h3
      integer :: r_p4,r_p5,r_p6
      double precision :: o_h1(r_h1),o_h2(r_h2),o_h3(r_h3)
      double precision :: o_p4(r_p4),o_p5(r_p5),o_p6(r_p6)
      double precision :: a_singles(r_h3,r_h2,r_h1,r_p6,r_p5,r_p4)
      double precision :: a_doubles(r_h3,r_h2,r_h1,r_p6,r_p5,r_p4)
      logical :: restricted
      double precision :: energy1,energy2
      integer :: h1, h2, h3, p4, p5, p6
      double precision :: factor,denom,sing,doub
      double precision :: d_p4,d_p5,d_p6
      double precision :: d_h1,d_h2,d_h3
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
     &     k_list,k_range,k_offset,a_singles,a_doubles,
     &     k_t1_local,k_t1_offset,d_t2,d_v2,k_t2_offset,k_v2_offset,
     &     restricted,k_evl_sorted,size_t3)
      implicit none
#include "mafdecls.fh"
#include "ccsd_t_ps.fh"
      integer, intent(in) :: k
      double precision, intent(inout) :: energy1,energy2
      integer, intent(in) :: k_list(7,*)
      integer, intent(in) :: k_range(*),k_offset(*)
      double precision, intent(inout) :: a_singles(:),a_doubles(:)
      logical, intent(in) :: restricted
      integer, intent(in) :: k_evl_sorted
      integer, intent(in) :: k_t1_local,k_t1_offset
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
     &                 k_t1_local,d_v2,k_t1_offset,k_v2_offset,
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

      integer function ccsd_t_6tasks_cutensor(
     &     restricted,noab,nvab,kspin,ksym)
      implicit none
      logical :: restricted
      integer :: noab,nvab
      integer :: kspin(*),ksym(*)
      integer :: ccsd_t_6tasks
!
      integer :: t_p4b,t_p5b,t_p6b
      integer :: t_h1b,t_h2b,t_h3b
!
      ccsd_t_6tasks=0
      do t_p4b = noab+1,noab+nvab
       do t_p5b = t_p4b,noab+nvab
        do t_p6b = t_p5b,noab+nvab
         do t_h1b = 1,noab
          do t_h2b = t_h1b,noab
           do t_h3b = t_h2b,noab
            if (kspin(t_p4b)+kspin(t_p5b)+kspin(t_p6b).eq.
     &             kspin(t_h1b)+kspin(t_h2b)+kspin(t_h3b)) then
            if ((.not.restricted).or.
     &              (kspin(t_p4b)+kspin(t_p5b)+kspin(t_p6b)+
     &         kspin(t_h1b)+kspin(t_h2b)+kspin(t_h3b).le.8)) then
            if (ieor(ksym(t_p4b),ieor(ksym(t_p5b),ieor(ksym(t_p6b),
     &          ieor(ksym(t_h1b),ieor(ksym(t_h2b),ksym(t_h3b)
     &              ))))).eq.0) then
               ccsd_t_6tasks=ccsd_t_6tasks + 1
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
!
      subroutine ccsd_t_neword_cutensor(tot_task, 
     &     w_thresh,task_thresh,
     &     restricted,noab,nvab,kspin,ksym,
     &     krange,kaux,klist)
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
      integer :: highindx
      parameter (highindx=7)
      task_thresh=tot_task
      m=0
      do t_p4b = noab+1,noab+nvab
       range_p4 = krange(t_p4b)
       do t_p5b = t_p4b,noab+nvab
        range_p5 = krange(t_p5b)
        do t_p6b = t_p5b,noab+nvab
         range_p6 = krange(t_p6b)
         do t_h1b = 1,noab
          range_h1 = krange(t_h1b)
          do t_h2b = t_h1b,noab
           range_h2 = krange(t_h2b)
           do t_h3b = t_h2b,noab
            range_h3 = krange(t_h3b)
            if (kspin(t_p4b)+kspin(t_p5b)+kspin(t_p6b).eq.
     &             kspin(t_h1b)+kspin(t_h2b)+kspin(t_h3b)) then
            if ((.not.restricted).or.
     &              (kspin(t_p4b)+kspin(t_p5b)+kspin(t_p6b)+
     &         kspin(t_h1b)+kspin(t_h2b)+kspin(t_h3b).le.8)) then
            if (ieor(ksym(t_p4b),ieor(ksym(t_p5b),ieor(ksym(t_p6b),
     &          ieor(ksym(t_h1b),ieor(ksym(t_h2b),ksym(t_h3b)
     &              ))))).eq.0) then
            m=m+1
            kaux(1,m)=t_p4b
            kaux(2,m)=t_p5b
            kaux(3,m)=t_p6b
            kaux(4,m)=t_h1b
            kaux(5,m)=t_h2b
            kaux(6,m)=t_h3b
            kaux(highindx,m)=range_p4*range_p5*range_p6
     &                      *range_h1*range_h2*range_h3
            endif
            endif
            endif
           enddo
          enddo
         enddo
        enddo
       enddo
      enddo
!
      if(m.ne.tot_task)  then
         write(6,*) ' m ',m,' tot_task ',tot_task
         call errquit("incompatibility_task",6,MA_ERR)
      endif
!
      wl_max=0
      do m=1,tot_task
        wl_max=max(kaux(highindx,m),wl_max)
      enddo
!
      wl_min=wl_max
      do m=1,tot_task
        wl_min=min(kaux(highindx,m),wl_min)
      enddo
!     all tiles the same. no ordering needed
      if(((wl_max-wl_min)*100d0)/wl_max.lt.1d0) then 
      do m=1,tot_task
         do j=1,highindx
            klist(j,m)=kaux(j,m)
         enddo
      enddo
      endif
!
! simple sorting
!
      do m=1,tot_task
         do j=1,highindx
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
          call ccsd_sillysort_cutensor(w_in,
     &     kaux,klist, tot_task,m,highindx)
       enddo
      call ccsd_sillysort_cutensor(0,
     &     kaux,klist, tot_task,m,highindx)
!     check that everybody has been invalidated
      do i=1,tot_task
         if(kaux(highindx,i).ge.0) then
            write(6,*) ' kaux element ',i,
     &           'not sorted. val = ',kaux(highindx,i)
            call errquit('neworder error ',0,0)
            endif
      enddo
!
!     figure out task with w lt w_thresh
!
!      w_thresh=w_thresh**6

      do i=1,tot_task
         if(klist(highindx,i).lt.w_thresh) then
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
      subroutine ccsd_sillysort_cutensor(value,
     &     kaux,klist,n,found,highindx )
      implicit none
!
! very simple sorting
!
      integer :: value
      integer :: n
      integer :: found
      integer :: highindx
      integer :: kaux(highindx,*)
      integer :: klist(highindx,*)
!
      integer :: i,j,m
!
      do m=1,n
         if(kaux(highindx,m).gt.value) then
            found=found+1
            do j=1,highindx
               klist(j,found)=kaux(j,m)
            enddo
!     invalidate
            kaux(highindx,m)=-99
         endif
      enddo
      end


      subroutine ccsd_t_singles_cutensor(a_t3,
     &                                   d_t1,d_v2,
     &                                   k_t1_offset,k_v2_offset,
     &                                   t_h1b,t_h2b,t_h3b,
     &                                   t_p4b,t_p5b,t_p6b)
      implicit none
#include "global.fh"
#include "mafdecls.fh"
#include "util.fh"
#include "errquit.fh"
#include "tce.fh"
      double precision :: a_t3(*)
      integer :: d_t1, d_v2
      integer :: k_t1_offset, k_v2_offset
      integer :: t_p4b, t_p5b, t_p6b, t_h1b, t_h2b, t_h3b
      CALL ccsd_t_singles_cutensor_1(d_t1,k_t1_offset,
     &                               d_v2,k_v2_offset,
     &                               a_t3,
     &                               t_p4b,t_p5b,t_p6b,
     &                               t_h1b,t_h2b,t_h3b)
      END



      subroutine ccsd_t_singles_cutensor_1(d_t1,k_t1_offset,
     &                              d_v2,k_v2_offset,
     &                              a_t3,
     &                              t_p4b,t_p5b,t_p6b,
     &                              t_h1b,t_h2b,t_h3b)
!
! elimination of the transpositions (kowalski,august,2009)
!
      implicit none
#include "global.fh"
#include "mafdecls.fh"
#include "sym.fh"
#include "errquit.fh"
#include "tce.fh"
      integer :: d_t1
      integer :: k_t1_offset
      integer :: d_v2
      integer :: k_v2_offset
      integer :: t_p4b
      integer :: t_p5b
      integer :: t_p6b
      integer :: t_h1b
      integer :: t_h2b
      integer :: t_h3b
      integer :: p4b
      integer :: p5b
      integer :: p6b
      integer :: h1b
      integer :: h2b
      integer :: h3b
      integer :: dimc
      integer :: l_c_sort
      integer :: k_c_sort
      integer :: p4b_1
      integer :: h1b_1
      integer :: p5b_2
      integer :: p6b_2
      integer :: h2b_2
      integer :: h3b_2
      integer :: dim_common
      integer :: dima_sort
      integer :: dima
      integer :: dimb_sort
      integer :: dimb
      integer :: l_a_sort
      integer :: k_a_sort
      integer :: l_a
      integer :: k_a
      integer :: l_b_sort
      integer :: k_b_sort
      integer :: l_b
      integer :: k_b
! -- peta-exa --
      integer :: a3(9,6)
      integer :: ia6,ja6
! --------------
      double precision :: a_t3(*)
      logical :: skipped

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
      if((p5b.le.p6b).and.(h2b.le.h3b).and.(p4b.ne.0)) then
      skipped = .false.
      IF (.not.skipped) THEN
      IF ((.not.restricted).or.
     &     (int_mb(k_spin+p4b-1)+int_mb(k_spin+p5b-1)+
     &      int_mb(k_spin+p6b-1)+int_mb(k_spin+h1b-1)+
     &      int_mb(k_spin+h2b-1)+int_mb(k_spin+h3b-1).ne.12)) THEN
      IF (int_mb(k_spin+p4b-1)+int_mb(k_spin+p5b-1)+
     &    int_mb(k_spin+p6b-1).eq. 
     &    int_mb(k_spin+h1b-1)+int_mb(k_spin+h2b-1)+
     &    int_mb(k_spin+h3b-1)) THEN
      IF (ieor(int_mb(k_sym+p4b-1),ieor(int_mb(k_sym+p5b-1),
     &    ieor(int_mb(k_sym+p6b-1),ieor(int_mb(k_sym+h1b-1),
     &    ieor(int_mb(k_sym+h2b-1),int_mb(k_sym+h3b-1)))))) 
     &   .eq.ieor(irrep_v,irrep_t)) THEN
      dimc = int_mb(k_range+p4b-1) * int_mb(k_range+p5b-1) * 
     &       int_mb(k_range+p6b-1) * int_mb(k_range+h1b-1) * 
     &       int_mb(k_range+h2b-1) * int_mb(k_range+h3b-1)
      IF (int_mb(k_spin+p4b-1).eq.int_mb(k_spin+h1b-1)) THEN
      IF (ieor(int_mb(k_sym+p4b-1),int_mb(k_sym+h1b-1))
     &               .eq.irrep_t) THEN
      CALL TCE_RESTRICTED_2(p4b,h1b,p4b_1,h1b_1)
      CALL TCE_RESTRICTED_4(p5b,p6b,h2b,h3b,p5b_2,p6b_2,h2b_2,h3b_2)
      dim_common = 1
      dima_sort = int_mb(k_range+p4b-1) * int_mb(k_range+h1b-1)
      dima = dim_common * dima_sort
      dimb_sort = int_mb(k_range+p5b-1) * int_mb(k_range+p6b-1) * 
     &            int_mb(k_range+h2b-1) * int_mb(k_range+h3b-1)
      dimb = dim_common * dimb_sort
      IF ((dima .gt. 0).and.(dimb .gt. 0)) THEN
      IF (.not.MA_PUSH_GET(mt_dbl,dima,'noname',l_a_sort,k_a_sort)) 
     &           CALL ERRQUIT('ccsd_t_singles_1',1,MA_ERR)
      IF (.not.MA_PUSH_GET(mt_dbl,dima,'noname',l_a,k_a)) 
     &           CALL ERRQUIT('ccsd_t_singles_1',2,MA_ERR)
      CALL GET_HASH_BLOCK_MA(dbl_mb(d_t1),dbl_mb(k_a),dima,
     &                       int_mb(k_t1_offset),
     &                       (h1b_1-1+noab*(p4b_1-noab-1)))
      CALL TCE_SORT_2(dbl_mb(k_a),dbl_mb(k_a_sort),
     &                int_mb(k_range+p4b-1),int_mb(k_range+h1b-1),
     &                2,1,1.0d0)
      IF (.not.MA_POP_STACK(l_a)) 
     &           CALL ERRQUIT('ccsd_t_singles_1',3,MA_ERR)
      IF (.not.MA_PUSH_GET(mt_dbl,dimb,'noname',l_b_sort,k_b_sort)) 
     &           CALL ERRQUIT('ccsd_t_singles_1',4,MA_ERR)
      if(.not.intorb) then
      CALL GET_HASH_BLOCK(d_v2,dbl_mb(k_b_sort),dimb,
     &                    int_mb(k_v2_offset),
     &                    (h3b_2-1+(noab+nvab)*(h2b_2-1+(noab+nvab)*
     &                    (p6b_2-1+(noab+nvab)*(p5b_2-1)))))
      else
      CALL GET_HASH_BLOCK_I(d_v2,dbl_mb(k_b_sort),dimb,
     &                      int_mb(k_v2_offset),(h3b_2-1+(noab+nvab)*
     &                      (h2b_2-1+(noab+nvab)*(p6b_2-1+(noab+nvab)*
     &                      (p5b_2-1)))),h3b_2,h2b_2,p6b_2,p5b_2)
      end if
      IF ((t_p4b.eq.p4b).and.(t_p5b.eq.p5b).and.(t_p6b.eq.p6b)
     &.and.(t_h1b.eq.h1b).and.(t_h2b.eq.h2b).and.(t_h3b.eq.h3b)) THEN
      call sd_t_s1_1(int_mb(k_range+h3b-1),int_mb(k_range+h2b-1),
     &               int_mb(k_range+h1b-1),int_mb(k_range+p6b-1),
     &               int_mb(k_range+p5b-1),int_mb(k_range+p4b-1),
     &               a_t3,dbl_mb(k_a_sort),dbl_mb(k_b_sort))

      END IF
      IF ((t_p4b.eq.p4b).and.(t_p5b.eq.p5b).and.(t_p6b.eq.p6b)
     &.and.(t_h1b.eq.h2b).and.(t_h2b.eq.h1b).and.(t_h3b.eq.h3b)) THEN
      call sd_t_s1_2(int_mb(k_range+h3b-1),int_mb(k_range+h2b-1),
     &               int_mb(k_range+h1b-1),int_mb(k_range+p6b-1),
     &               int_mb(k_range+p5b-1),int_mb(k_range+p4b-1),
     &               a_t3,dbl_mb(k_a_sort),dbl_mb(k_b_sort))

      END IF
      IF ((t_p4b.eq.p4b).and.(t_p5b.eq.p5b).and.(t_p6b.eq.p6b)
     &.and.(t_h1b.eq.h2b).and.(t_h2b.eq.h3b).and.(t_h3b.eq.h1b)) THEN
      call sd_t_s1_3(int_mb(k_range+h3b-1),int_mb(k_range+h2b-1),
     &               int_mb(k_range+h1b-1),int_mb(k_range+p6b-1),
     &               int_mb(k_range+p5b-1),int_mb(k_range+p4b-1),
     &               a_t3,dbl_mb(k_a_sort),dbl_mb(k_b_sort))

      END IF
      IF ((t_p4b.eq.p5b).and.(t_p5b.eq.p4b).and.(t_p6b.eq.p6b)
     &.and.(t_h1b.eq.h1b).and.(t_h2b.eq.h2b).and.(t_h3b.eq.h3b)) THEN
      call sd_t_s1_4(int_mb(k_range+h3b-1),int_mb(k_range+h2b-1),
     &               int_mb(k_range+h1b-1),int_mb(k_range+p6b-1),
     &               int_mb(k_range+p5b-1),int_mb(k_range+p4b-1),
     &               a_t3,dbl_mb(k_a_sort),dbl_mb(k_b_sort))

      END IF
      IF ((t_p4b.eq.p5b).and.(t_p5b.eq.p4b).and.(t_p6b.eq.p6b)
     &.and.(t_h1b.eq.h2b).and.(t_h2b.eq.h1b).and.(t_h3b.eq.h3b)) THEN
      call sd_t_s1_5(int_mb(k_range+h3b-1),int_mb(k_range+h2b-1),
     &               int_mb(k_range+h1b-1),int_mb(k_range+p6b-1),
     &               int_mb(k_range+p5b-1),int_mb(k_range+p4b-1),
     &               a_t3,dbl_mb(k_a_sort),dbl_mb(k_b_sort))

      END IF
      IF ((t_p4b.eq.p5b).and.(t_p5b.eq.p4b).and.(t_p6b.eq.p6b)
     &.and.(t_h1b.eq.h2b).and.(t_h2b.eq.h3b).and.(t_h3b.eq.h1b)) THEN
      call sd_t_s1_6(int_mb(k_range+h3b-1),int_mb(k_range+h2b-1),
     &               int_mb(k_range+h1b-1),int_mb(k_range+p6b-1),
     &               int_mb(k_range+p5b-1),int_mb(k_range+p4b-1),
     &               a_t3,dbl_mb(k_a_sort),dbl_mb(k_b_sort))

      END IF
      IF ((t_p4b.eq.p5b).and.(t_p5b.eq.p6b).and.(t_p6b.eq.p4b)
     &.and.(t_h1b.eq.h1b).and.(t_h2b.eq.h2b).and.(t_h3b.eq.h3b)) THEN
      call sd_t_s1_7(int_mb(k_range+h3b-1),int_mb(k_range+h2b-1),
     &               int_mb(k_range+h1b-1),int_mb(k_range+p6b-1),
     &               int_mb(k_range+p5b-1),int_mb(k_range+p4b-1),
     &               a_t3,dbl_mb(k_a_sort),dbl_mb(k_b_sort))

      END IF
      IF ((t_p4b.eq.p5b).and.(t_p5b.eq.p6b).and.(t_p6b.eq.p4b)
     &.and.(t_h1b.eq.h2b).and.(t_h2b.eq.h1b).and.(t_h3b.eq.h3b)) THEN
      call sd_t_s1_8(int_mb(k_range+h3b-1),int_mb(k_range+h2b-1),
     &               int_mb(k_range+h1b-1),int_mb(k_range+p6b-1),
     &               int_mb(k_range+p5b-1),int_mb(k_range+p4b-1),
     &               a_t3,dbl_mb(k_a_sort),dbl_mb(k_b_sort))

      END IF
      IF ((t_p4b.eq.p5b).and.(t_p5b.eq.p6b).and.(t_p6b.eq.p4b)
     &.and.(t_h1b.eq.h2b).and.(t_h2b.eq.h3b).and.(t_h3b.eq.h1b)) THEN
      call sd_t_s1_9(int_mb(k_range+h3b-1),int_mb(k_range+h2b-1),
     &               int_mb(k_range+h1b-1),int_mb(k_range+p6b-1),
     &               int_mb(k_range+p5b-1),int_mb(k_range+p4b-1),
     &               a_t3,dbl_mb(k_a_sort),dbl_mb(k_b_sort))

      END IF
      IF (.not.MA_POP_STACK(l_b_sort)) 
     &       CALL ERRQUIT('ccsd_t_singles_1',7,MA_ERR)
      IF (.not.MA_POP_STACK(l_a_sort)) 
     &       CALL ERRQUIT('ccsd_t_singles_1',8,MA_ERR)
!
      END IF
      END IF
      END IF
!
      END IF
      END IF
      END IF
      END IF
!
      END IF
      END DO
      END


      subroutine ccsd_t_doubles_cutensor(a_i0,d_t2,d_v2,
     &           k_t2_offset,k_v2_offset,
     &           t_h1b,t_h2b,t_h3b,t_p4b,t_p5b,t_p6b)
!     $Id$
!     This is a Fortran77 program generated by Tensor Contraction Engine v.1.0
!     Copyright (c) Battelle & Pacific Northwest National Laboratory (2002)
!     i0 ( p4 p5 p6 h1 h2 h3 )_vt + = -1 * P( 9 ) * Sum ( h7 ) * t ( p4 p5 h1 h7 )_t * v ( h7 p6 h2 h3 )_v
!     i0 ( p4 p5 p6 h1 h2 h3 )_vt + = -1 * P( 9 ) * Sum ( p7 ) * t ( p4 p7 h1 h2 )_t * v ( p5 p6 h3 p7 )_v
      implicit none
#include "global.fh"
#include "mafdecls.fh"
#include "util.fh"
#include "errquit.fh"
#include "tce.fh"
      integer :: t_p4b
      integer :: t_p5b
      integer :: t_p6b
      integer :: t_h1b
      integer :: t_h2b
      integer :: t_h3b
      integer :: d_t2
      integer :: k_t2_offset
      integer :: d_v2
      integer :: k_v2_offset
      integer :: l_v2sub
      integer :: l_t2sub
      integer :: l_scratch
      double precision :: a_i0(*)
      double precision, allocatable :: d_t2sub(:)
      double precision, allocatable :: d_v2sub(:)
      double precision, allocatable :: d_scratch(:)
      integer :: alloc_error
      integer :: range_p4,range_h1,ii
      integer :: t2size,v2size
      call cute_t_v2t2lgth(t2size,v2size)
      allocate(d_t2sub(1:t2size), stat=alloc_error)
      if (alloc_error.ne.0) then
        call errquit('ccsd_t_doubles_t: MA error d_t2sub',0,MA_ERR)
      endif
      allocate(d_scratch(1:max(t2size,v2size)), stat=alloc_error)
      if (alloc_error.ne.0) then
        call errquit('ccsd_t_doubles_t: MA error d_scratch',0,MA_ERR)
      endif
      allocate(d_v2sub(1:v2size), stat=alloc_error)
      if (alloc_error.ne.0) then
        call errquit('ccsd_t_doubles_t: MA error d_v2sub',0,MA_ERR)
      endif
      CALL ccsd_t_doubles_cutensor_12(d_t2,
     &     k_t2_offset,d_v2,k_v2_offset,
     &     a_i0,t_p4b,t_p5b,t_p6b,t_h1b,t_h2b,t_h3b,
     &     d_v2sub,d_t2sub,d_scratch)
      deallocate(d_v2sub)
      deallocate(d_scratch)
      deallocate(d_t2sub)
      END

      subroutine cute_t_v2t2lgth(t2size,v2size)
      implicit none
#include "mafdecls.fh"
#include "tce.fh"
!     compute v2sub and t2sub max length
      integer :: t2size,v2size ! [out]
      integer :: range_p4,range_h1,ii
      range_p4=0
      do ii = noab+1,noab+nvab
         range_p4 = max(range_p4,int_mb(k_range+ii-1))
      enddo
      range_h1=0
      do ii = 1,noab
         range_h1 = max(range_h1,int_mb(k_range+ii-1))
      enddo
      t2size=(range_p4**2)*(range_h1**2)
      v2size=max((range_p4)*(range_h1**3),
     &     (range_p4**3)*(range_h1))
      end

      subroutine ccsd_t_doubles_cutensor_12(d_a,k_a_offset,
     &                                      d_b,k_b_offset,
     &           triplesx,t_p4b,t_p5b,t_p6b,t_h1b,t_h2b,t_h3b,
     &     v2sub,t2sub,scratch)
!     This is a Fortran77 program generated by Tensor Contraction Engine v.1.0
!     Copyright (c) Battelle & Pacific Northwest National Laboratory (2002)
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
#include "ccsd_t_ps.fh"
#include "util.fh"
      integer :: d_a
      integer :: k_a_offset
      integer :: d_b
      integer :: k_b_offset
      integer :: t_p4b
      integer :: t_p5b
      integer :: t_p6b
      integer :: t_h1b
      integer :: t_h2b
      integer :: t_h3b
      integer :: p4b
      integer :: p5b
      integer :: p6b
      integer :: h1b
      integer :: h2b
      integer :: h3b
      integer :: dimc
      integer :: l_c_sort
      integer :: k_c_sort
      integer :: h7b,p7b
      integer :: p5b_2,p7b_1,h2b_1,p7b_2
      integer :: p4b_1
      integer :: p5b_1
      integer :: h1b_1
      integer :: h7b_1
      integer :: p6b_2
      integer :: h7b_2
      integer :: h2b_2
      integer :: h3b_2
      integer :: dim_common
      integer :: dima_sort
      integer :: dima
      integer :: dimb_sort
      integer :: dimb
      integer :: l_a
      integer :: k_a
      integer :: l_b
      integer :: k_b
! -- peta-exa --
      integer :: a3(9,6)
      integer :: ia6,ja6
! --------------
      double precision :: triplesx(*)
! --------------
      integer :: k_range_h3b_1
      integer :: k_range_h2b_1
      integer :: k_range_h1b_1
      integer :: k_range_p6b_1
      integer :: k_range_p5b_1
      integer :: k_range_p4b_1
      integer :: k_range_h7b_1
      integer :: k_range_p7b_1
      integer :: k_a_srt_len
      integer :: k_b_srt_len
      integer :: l_t2sub,l_triplesx,l_v2sub
      double precision :: t2sub(*),v2sub(*),scratch(*)
      logical :: tce_hashnsort,lresult
      external tce_hashnsort
      logical :: tce_hashnsort_2
      external tce_hashnsort_2
      integer :: n_h7b
      integer :: n_p7b
      double precision :: t0
      integer :: h7b_in,p7b_in

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
      if(((p4b.le.p5b).and.(h2b.le.h3b).and.(p4b.ne.0)).and.
     & ((.not.restricted).or.(int_mb(k_spin+p4b-1)+int_mb(k_spin+p5b-1
     &)+int_mb(k_spin+p6b-1)+int_mb(k_spin+h1b-1)+int_mb(k_spin+h2b-1)+i
     &nt_mb(k_spin+h3b-1).ne.12)).and.
     & (int_mb(k_spin+p4b-1)+int_mb(k_spin+p5b-1)+int_mb(k_spin+p6b-1)
     & .eq. int_mb(k_spin+h1b-1)+int_mb(k_spin+h2b-1)+int_mb(k_spin+h3b-
     &1)).and.
     & (ieor(int_mb(k_sym+p4b-1),ieor(int_mb(k_sym+p5b-1),ieor(int_mb(
     &k_sym+p6b-1),ieor(int_mb(k_sym+h1b-1),ieor(int_mb(k_sym+h2b-1),int
     &_mb(k_sym+h3b-1)))))) .eq. ieor(irrep_v,irrep_t))) THEN

      dimc = int_mb(k_range+p4b-1) * int_mb(k_range+p5b-1) * int_mb(k_ra
     &nge+p6b-1) * int_mb(k_range+h1b-1) * int_mb(k_range+h2b-1) * int_m
     &b(k_range+h3b-1)
!
!
!from l_triplesx
      k_range_h3b_1 = int_mb(k_range+h3b-1)
      k_range_h2b_1 = int_mb(k_range+h2b-1)
      k_range_h1b_1 = int_mb(k_range+h1b-1)
      k_range_p6b_1 = int_mb(k_range+p6b-1)
      k_range_p5b_1 = int_mb(k_range+p5b-1)
      k_range_p4b_1 = int_mb(k_range+p4b-1)

!     prep. work
!     figure out  cycles to work on
      n_h7b=0
      DO h7b = 1,noab
         if(tce_hashnsort(.true.,
     &        d_a, d_b,
     &        k_a_offset,k_b_offset,
     &        p4b,p5b,h1b,h7b,
     &        p6b,h2b,h3b,
     &        t2sub,v2sub
     &        )) then
            n_h7b=n_h7b+1
         endif
      enddo

      if(n_h7b.gt.0) then

         do h7b_in=ga_nodeid()+1,ga_nodeid()+noab
            h7b=mod(h7b_in,noab)+1
            if(tce_hashnsort(.false.,
     &           d_a, d_b,
     &           k_a_offset,k_b_offset,
     &           p4b,p5b,h1b,h7b,
     &           p6b,h2b,h3b,
     &           t2sub,v2sub)) then
            k_range_h7b_1 = int_mb(k_range+h7b-1)
      IF ((t_p4b .eq. p4b) .and. (t_p5b .eq. p5b) .and. (t_p6b .eq. p6b)
     & .and. (t_h1b .eq. h1b) .and. (t_h2b .eq. h2b) .and. (t_h3b .eq. h
     &3b)) THEN
         call cutensor_t_d1_1(k_range_h3b_1,k_range_h2b_1,
     &     k_range_h1b_1,k_range_p6b_1,
     &     k_range_p5b_1,k_range_p4b_1,
     &     k_range_h7b_1,
     &     triplesx,t2sub,v2sub)
      END IF
      IF ((t_p4b .eq. p4b) .and. (t_p5b .eq. p5b) .and. (t_p6b .eq. p6b)
     & .and. (t_h1b .eq. h2b) .and. (t_h2b .eq. h1b) .and. (t_h3b .eq. h
     &3b)) THEN
         call cutensor_t_d1_2(k_range_h3b_1,k_range_h2b_1,
     &        k_range_h1b_1,k_range_p6b_1,
     &        k_range_p5b_1,k_range_p4b_1,
     &        k_range_h7b_1,
     &        triplesx,t2sub,v2sub,scratch)
      END IF
      IF ((t_p4b .eq. p4b) .and. (t_p5b .eq. p5b) .and. (t_p6b .eq. p6b)
     & .and. (t_h1b .eq. h2b) .and. (t_h2b .eq. h3b) .and. (t_h3b .eq. h
     &1b)) THEN
      call cutensor_t_d1_3(k_range_h3b_1,k_range_h2b_1,
     &     k_range_h1b_1,k_range_p6b_1,
     &     k_range_p5b_1,k_range_p4b_1,
     &     k_range_h7b_1,
     &     triplesx,t2sub,v2sub,scratch)
       END IF
      IF ((t_p4b .eq. p6b) .and. (t_p5b .eq. p4b) .and. (t_p6b .eq. p5b)
     & .and. (t_h1b .eq. h1b) .and. (t_h2b .eq. h2b) .and. (t_h3b .eq. h
     &3b)) THEN
         call cutensor_t_d1_4(k_range_h3b_1,k_range_h2b_1,
     &        k_range_h1b_1,k_range_p6b_1,
     &        k_range_p5b_1,k_range_p4b_1,
     &        k_range_h7b_1,
     &        triplesx,t2sub,v2sub)
      END IF
      IF ((t_p4b .eq. p6b) .and. (t_p5b .eq. p4b) .and. (t_p6b .eq. p5b)
     & .and. (t_h1b .eq. h2b) .and. (t_h2b .eq. h1b) .and. (t_h3b .eq. h
     &3b)) THEN
      call cutensor_t_d1_5(k_range_h3b_1,k_range_h2b_1,
     &     k_range_h1b_1,k_range_p6b_1,
     &     k_range_p5b_1,k_range_p4b_1,
     &     k_range_h7b_1,
     &     triplesx,t2sub,v2sub,scratch)
      END IF
      IF ((t_p4b .eq. p6b) .and. (t_p5b .eq. p4b) .and. (t_p6b .eq. p5b)
     & .and. (t_h1b .eq. h2b) .and. (t_h2b .eq. h3b) .and. (t_h3b .eq. h
     &1b)) THEN
         call cutensor_t_d1_6(k_range_h3b_1,k_range_h2b_1,
     &        k_range_h1b_1,k_range_p6b_1,
     &        k_range_p5b_1,k_range_p4b_1,
     &        k_range_h7b_1,
     &        triplesx,t2sub,v2sub,scratch)
      END IF
      IF ((t_p4b .eq. p4b) .and. (t_p5b .eq. p6b) .and. (t_p6b .eq. p5b)
     & .and. (t_h1b .eq. h1b) .and. (t_h2b .eq. h2b) .and. (t_h3b .eq. h
     &3b)) THEN
         call cutensor_t_d1_7(k_range_h3b_1,k_range_h2b_1,
     &        k_range_h1b_1,k_range_p6b_1,
     &        k_range_p5b_1,k_range_p4b_1,
     &        k_range_h7b_1,
     &        triplesx,t2sub,v2sub)
      END IF
      IF ((t_p4b .eq. p4b) .and. (t_p5b .eq. p6b) .and. (t_p6b .eq. p5b)
     & .and. (t_h1b .eq. h2b) .and. (t_h2b .eq. h1b) .and. (t_h3b .eq. h
     &3b)) THEN
         call cutensor_t_d1_8(k_range_h3b_1,k_range_h2b_1,
     &        k_range_h1b_1,k_range_p6b_1,
     &        k_range_p5b_1,k_range_p4b_1,
     &        k_range_h7b_1,
     &        triplesx,t2sub,v2sub,scratch)
      END IF
      IF ((t_p4b .eq. p4b) .and. (t_p5b .eq. p6b) .and. (t_p6b .eq. p5b)
     & .and. (t_h1b .eq. h2b) .and. (t_h2b .eq. h3b) .and. (t_h3b .eq. h
     &1b)) THEN
         call cutensor_t_d1_9(k_range_h3b_1,k_range_h2b_1,
     &     k_range_h1b_1,k_range_p6b_1,
     &     k_range_p5b_1,k_range_p4b_1,
     &     k_range_h7b_1,
     &     triplesx,t2sub,v2sub,scratch)
      END IF
      endif
      END DO !h7b loop
      endif ! n_h7b.ne.0
      END IF
      END DO !ia6 loop
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
     & ((.not.restricted).or.(int_mb(k_spin+p4b-1)+int_mb(k_spin+p5b-1
     &)+int_mb(k_spin+p6b-1)+int_mb(k_spin+h1b-1)+int_mb(k_spin+h2b-1)+i
     &nt_mb(k_spin+h3b-1).ne.12)).and.
     & (int_mb(k_spin+p4b-1)+int_mb(k_spin+p5b-1)+int_mb(k_spin+p6b-1)
     & .eq. int_mb(k_spin+h1b-1)+int_mb(k_spin+h2b-1)+int_mb(k_spin+h3b-
     &1)).and.
     & (ieor(int_mb(k_sym+p4b-1),ieor(int_mb(k_sym+p5b-1),ieor(int_mb(
     &k_sym+p6b-1),ieor(int_mb(k_sym+h1b-1),ieor(int_mb(k_sym+h2b-1),int
     &_mb(k_sym+h3b-1)))))) .eq. ieor(irrep_v,irrep_t))) THEN
      dimc = int_mb(k_range+p4b-1) * int_mb(k_range+p5b-1) * int_mb(k_ra
     &nge+p6b-1) * int_mb(k_range+h1b-1) * int_mb(k_range+h2b-1) * int_m
     &b(k_range+h3b-1)
      k_range_h3b_1 = int_mb(k_range+h3b-1)
      k_range_h2b_1 = int_mb(k_range+h2b-1)
      k_range_h1b_1 = int_mb(k_range+h1b-1)
      k_range_p6b_1 = int_mb(k_range+p6b-1)
      k_range_p5b_1 = int_mb(k_range+p5b-1)
      k_range_p4b_1 = int_mb(k_range+p4b-1)
!     figure out  cycles to work on
      n_p7b=0
      DO p7b = noab+1,noab+nvab
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

      do p7b_in= ga_nodeid()+1,nvab+ga_nodeid()
         p7b=mod(p7b_in,nvab)+noab+1
         if(tce_hashnsort_2(.false.,
     &        d_a, d_b,
     &        k_a_offset,k_b_offset,
     &        p4b,p7b,h1b,h2b,
     &        p5b,p6b,h3b,
     &        t2sub,v2sub)) then

            k_range_p7b_1 = int_mb(k_range+p7b-1)
      IF ((t_p4b .eq. p4b) .and. (t_p5b .eq. p5b) .and. (t_p6b .eq. p6b)
     & .and. (t_h1b .eq. h1b) .and. (t_h2b .eq. h2b) .and. (t_h3b .eq. h
     &3b)) THEN
         call cutensor_t_d2_1(k_range_h3b_1,k_range_h2b_1,
     &        k_range_h1b_1,k_range_p6b_1,
     &        k_range_p5b_1,k_range_p4b_1,
     &        k_range_p7b_1,
     &        triplesx,t2sub,v2sub,scratch)
      END IF
      IF ((t_p4b .eq. p4b) .and. (t_p5b .eq. p5b) .and. (t_p6b .eq. p6b)
     & .and. (t_h1b .eq. h3b) .and. (t_h2b .eq. h1b) .and. (t_h3b .eq. h
     &2b)) THEN
         call cutensor_t_d2_2(k_range_h3b_1,k_range_h2b_1,
     &        k_range_h1b_1,k_range_p6b_1,
     &        k_range_p5b_1,k_range_p4b_1,
     &        k_range_p7b_1,
     &        triplesx,t2sub,v2sub,scratch)
      END IF
      IF ((t_p4b .eq. p4b) .and. (t_p5b .eq. p5b) .and. (t_p6b .eq. p6b)
     & .and. (t_h1b .eq. h1b) .and. (t_h2b .eq. h3b) .and. (t_h3b .eq. h
     &2b)) THEN
         call cutensor_t_d2_3(k_range_h3b_1,k_range_h2b_1,
     &        k_range_h1b_1,k_range_p6b_1,
     &        k_range_p5b_1,k_range_p4b_1,
     &        k_range_p7b_1,
     &        triplesx,t2sub,v2sub)
      END IF
      IF ((t_p4b .eq. p5b) .and. (t_p5b .eq. p4b) .and. (t_p6b .eq. p6b)
     & .and. (t_h1b .eq. h1b) .and. (t_h2b .eq. h2b) .and. (t_h3b .eq. h
     &3b)) THEN
         call cutensor_t_d2_4(k_range_h3b_1,k_range_h2b_1,
     &        k_range_h1b_1,k_range_p6b_1,
     &        k_range_p5b_1,k_range_p4b_1,
     &        k_range_p7b_1,
     &        triplesx,t2sub,v2sub,scratch)
      END IF
      IF ((t_p4b .eq. p5b) .and. (t_p5b .eq. p4b) .and. (t_p6b .eq. p6b)
     & .and. (t_h1b .eq. h3b) .and. (t_h2b .eq. h1b) .and. (t_h3b .eq. h
     &2b)) THEN

         call cutensor_t_d2_5(k_range_h3b_1,k_range_h2b_1,
     &        k_range_h1b_1,k_range_p6b_1,
     &        k_range_p5b_1,k_range_p4b_1,
     &        k_range_p7b_1,
     &        triplesx,t2sub,v2sub,scratch)
      END IF
      IF ((t_p4b .eq. p5b) .and. (t_p5b .eq. p4b) .and. (t_p6b .eq. p6b)
     & .and. (t_h1b .eq. h1b) .and. (t_h2b .eq. h3b) .and. (t_h3b .eq. h
     &2b)) THEN
         call cutensor_t_d2_6(k_range_h3b_1,k_range_h2b_1,
     &        k_range_h1b_1,k_range_p6b_1,
     &        k_range_p5b_1,k_range_p4b_1,
     &        k_range_p7b_1,
     &        triplesx,t2sub,v2sub,scratch)
      END IF
      IF ((t_p4b .eq. p5b) .and. (t_p5b .eq. p6b) .and. (t_p6b .eq. p4b)
     & .and. (t_h1b .eq. h1b) .and. (t_h2b .eq. h2b) .and. (t_h3b .eq. h
     &3b)) THEN
         call cutensor_t_d2_7(k_range_h3b_1,k_range_h2b_1,
     &        k_range_h1b_1,k_range_p6b_1,
     &        k_range_p5b_1,k_range_p4b_1,
     &        k_range_p7b_1,
     &        triplesx,t2sub,v2sub)
      END IF
      IF ((t_p4b .eq. p5b) .and. (t_p5b .eq. p6b) .and. (t_p6b .eq. p4b)
     & .and. (t_h1b .eq. h3b) .and. (t_h2b .eq. h1b) .and. (t_h3b .eq. h
     &2b)) THEN
         call cutensor_t_d2_8(k_range_h3b_1,k_range_h2b_1,
     &        k_range_h1b_1,k_range_p6b_1,
     &        k_range_p5b_1,k_range_p4b_1,
     &        k_range_p7b_1,
     &        triplesx,t2sub,v2sub,scratch)
      END IF
      IF ((t_p4b .eq. p5b) .and. (t_p5b .eq. p6b) .and. (t_p6b .eq. p4b)
     & .and. (t_h1b .eq. h1b) .and. (t_h2b .eq. h3b) .and. (t_h3b .eq. h
     &2b)) THEN
         call cutensor_t_d2_9(k_range_h3b_1,k_range_h2b_1,
     &        k_range_h1b_1,k_range_p6b_1,
     &        k_range_p5b_1,k_range_p4b_1,
     &        k_range_p7b_1,
     &        triplesx,t2sub,v2sub,scratch)
      END IF
      endif
      END DO ! p7b loop

      endif ! n_p7b.ne.0
!
!
      END IF
      END DO ! ia6 loop
      END

      subroutine cutensor_t_d1_1(h3d,h2d,h1d,p6d,p5d,p4d,h7d,
     &               triplesx,t2sub,v2sub)
      implicit none
      integer :: h3d,h2d,h1d,p6d,p5d,p4d,h7d
      integer :: h3,h2,h1,p6,p5,p4,h7
      integer :: h2h3
      double precision :: triplesx(h3d*h2d,h1d,p6d,p5d,p4d)
      double precision :: t2sub(h7d,p4d,p5d,h1d)
      double precision :: v2sub(h3d*h2d,p6d,h7d)
      do p4=1,p4d
      do p5=1,p5d
      do p6=1,p6d
      do h1=1,h1d
      do h7=1,h7d
      do h2h3=1,h3d*h2d
       triplesx(h2h3,h1,p6,p5,p4)=triplesx(h2h3,h1,p6,p5,p4)
     &   - t2sub(h7,p4,p5,h1)*v2sub(h2h3,p6,h7)
      enddo
      enddo
      enddo
      enddo
      enddo
      enddo
      end
!
      subroutine cutensor_t_d1_2(h3d,h2d,h1d,p6d,p5d,p4d,h7d,
     &               triplesx,t2sub,v2sub,scratch)
      implicit none
      integer :: h3d,h2d,h1d,p6d,p5d,p4d,h7d
      integer :: h3,h2,h1,p6,p5,p4,h7
      double precision :: triplesx(h3d,h1d,h2d,p6d,p5d,p4d)
      double precision :: t2sub(h7d,p4d,p5d,h1d)
      double precision :: scratch(h7d,h3d,h2d,p6d)
      double precision :: v2sub(*)
      call cute_t_transpt2_7326( v2sub,scratch, h3d,h2d,p6d,h7d)
      do p4=1,p4d
      do p5=1,p5d
      do p6=1,p6d
      do h2=1,h2d
      do h1=1,h1d
      do h3=1,h3d
      do h7=1,h7d
       triplesx(h3,h1,h2,p6,p5,p4)=triplesx(h3,h1,h2,p6,p5,p4)
     &   + t2sub(h7,p4,p5,h1)*scratch(h7,h3,h2,p6)
      enddo
      enddo
      enddo
      enddo
      enddo
      enddo
      enddo
      end
!
      subroutine cutensor_t_d1_3(h3d,h2d,h1d,p6d,p5d,p4d,h7d,
     &               triplesx,t2sub,v2sub,scratch)
      implicit none

      integer :: h3d,h2d,h1d,p6d,p5d,p4d,h7d
      integer :: h3,h2,h1,p6,p5,p4,h7
      integer :: h3h2
      double precision :: triplesx(h1d,h3d*h2d,p6d,p5d,p4d)
      double precision :: scratch(h7d,h3d*h2d,p6d)
!      double precision :: v2sub(h3d*h2d,p6d,h7d)
      double precision :: t2sub(h7d,p4d,p5d,h1d)
      double precision :: v2sub(h3d,h2d,p6d,h7d)
      call cute_t_transpt2_7326( v2sub,scratch, h3d,h2d,p6d,h7d)
      do p4=1,p4d
      do p5=1,p5d
      do p6=1,p6d
      do h3h2=1,h3d*h2d
      do h1=1,h1d
      do h7=1,h7d
       triplesx(h1,h3h2,p6,p5,p4)=triplesx(h1,h3h2,p6,p5,p4)
     &   - t2sub(h7,p4,p5,h1)*scratch(h7,h3h2,p6)
      enddo
      enddo
      enddo
      enddo
      enddo
      enddo
      end
!
      subroutine cutensor_t_d1_4(h3d,h2d,h1d,p6d,p5d,p4d,h7d,
     &               triplesx,t2sub,v2sub)
      implicit none

      integer :: h3d,h2d,h1d,p6d,p5d,p4d,h7d
      integer :: h3,h2,h1,p6,p5,p4,h7
      integer :: h2h3
      double precision :: triplesx(h3d*h2d,h1d,p5d,p4d,p6d)
      double precision :: t2sub(h7d,p4d,p5d,h1d)
      double precision :: v2sub(h3d*h2d,p6d,h7d)
      do p6=1,p6d
      do p4=1,p4d
      do p5=1,p5d
      do h1=1,h1d
      do h7=1,h7d
      do h2h3=1,h2d*h3d
       triplesx(h2h3,h1,p5,p4,p6)=triplesx(h2h3,h1,p5,p4,p6)
     &   - t2sub(h7,p4,p5,h1)*v2sub(h2h3,p6,h7)
      enddo
      enddo
      enddo
      enddo
      enddo
      enddo
      end
!
      subroutine cutensor_t_d1_5(h3d,h2d,h1d,p6d,p5d,p4d,h7d,
     &               triplesx,t2sub,v2sub,scratch)
      implicit none

      integer :: h3d,h2d,h1d,p6d,p5d,p4d,h7d
      integer :: h3,h2,h1,p6,p5,p4,h7
      double precision :: triplesx(h3d,h1d,h2d,p5d,p4d,p6d)
      double precision :: scratch(h7d,h3d,h2d,p6d)
      double precision :: t2sub(h7d,p4d,p5d,h1d)
      double precision :: v2sub(h3d,h2d,p6d,h7d)
      call cute_t_transpt2_7326( v2sub,scratch, h3d,h2d,p6d,h7d)
      do p6=1,p6d
      do p4=1,p4d
      do p5=1,p5d
      do h2=1,h2d
      do h1=1,h1d
      do h3=1,h3d
      do h7=1,h7d
       triplesx(h3,h1,h2,p5,p4,p6)=triplesx(h3,h1,h2,p5,p4,p6)
     &    + t2sub(h7,p4,p5,h1)*scratch(h7,h3,h2,p6)
      enddo
      enddo
      enddo
      enddo
      enddo
      enddo
      enddo
      end
!
      subroutine cutensor_t_d1_6(h3d,h2d,h1d,p6d,p5d,p4d,h7d,
     &               triplesx,t2sub,v2sub,scratch)
      implicit none

      integer :: h3d,h2d,h1d,p6d,p5d,p4d,h7d
      integer :: h3,h2,h1,p6,p5,p4,h7
      integer :: h2h3
      double precision :: triplesx(h1d,h3d*h2d,p5d,p4d,p6d)
      double precision :: scratch(h7d,h3d*h2d,p6d)
      double precision :: t2sub(h7d,p4d,p5d,h1d)
      double precision :: v2sub(*)
      call cute_t_transpt2_7326( v2sub,scratch, h3d,h2d,p6d,h7d)
      do p6=1,p6d
      do p4=1,p4d
      do p5=1,p5d
      do h2h3=1,h2d*h3d
      do h1=1,h1d
      do h7=1,h7d
       triplesx(h1,h2h3,p5,p4,p6)=triplesx(h1,h2h3,p5,p4,p6)
     &   - t2sub(h7,p4,p5,h1)*scratch(h7,h2h3,p6)
      enddo
      enddo
      enddo
      enddo
      enddo
      enddo
      end
!
      subroutine cutensor_t_d1_7(h3d,h2d,h1d,p6d,p5d,p4d,h7d,
     &               triplesx,t2sub,v2sub)
      implicit none
      integer :: h3d,h2d,h1d,p6d,p5d,p4d,h7d
      integer :: h3,h2,h1,p6,p5,p4,h7
      integer :: h2h3
      double precision :: triplesx(h3d*h2d,h1d,p5d,p6d,p4d)
      double precision :: t2sub(h7d,p4d,p5d,h1d)
      double precision :: v2sub(h3d*h2d,p6d,h7d)
      do p4=1,p4d
      do p6=1,p6d
      do p5=1,p5d
      do h1=1,h1d
      do h7=1,h7d
      do h2h3=1,h2d*h3d
       triplesx(h2h3,h1,p5,p6,p4)=triplesx(h2h3,h1,p5,p6,p4)
     &   + t2sub(h7,p4,p5,h1)*v2sub(h2h3,p6,h7)
      enddo
      enddo
      enddo
      enddo
      enddo
      enddo
      end
!
      subroutine cutensor_t_d1_8(h3d,h2d,h1d,p6d,p5d,p4d,h7d,
     &               triplesx,t2sub,v2sub,scratch)
      implicit none
      integer :: h3d,h2d,h1d,p6d,p5d,p4d,h7d
      integer :: h3,h2,h1,p6,p5,p4,h7
      double precision :: triplesx(h3d,h1d,h2d,p5d,p6d,p4d)
      double precision :: t2sub(h7d,p4d,p5d,h1d)
      double precision :: v2sub(*)
      double precision :: scratch(h7d,h3d,h2d,p6d)
      call cute_t_transpt2_7326( v2sub,scratch, h3d,h2d,p6d,h7d)
      do p4=1,p4d
      do p6=1,p6d
      do p5=1,p5d
      do h2=1,h2d
      do h1=1,h1d
      do h3=1,h3d
      do h7=1,h7d
       triplesx(h3,h1,h2,p5,p6,p4)=triplesx(h3,h1,h2,p5,p6,p4)
     &   - t2sub(h7,p4,p5,h1)*scratch(h7,h3,h2,p6)
      enddo
      enddo
      enddo
      enddo
      enddo
      enddo
      enddo
      end
!
      subroutine cutensor_t_d1_9(h3d,h2d,h1d,p6d,p5d,p4d,h7d,
     &               triplesx,t2sub,v2sub,scratch)
      implicit none
      integer :: h3d,h2d,h1d,p6d,p5d,p4d,h7d
      integer :: h3,h2,h1,p6,p5,p4,h7
      integer :: h3h2
      double precision :: triplesx(h1d,h3d*h2d,p5d,p6d,p4d)
      double precision :: scratch(h7d,h3d*h2d,p6d)
      double precision :: t2sub(h7d,p4d,p5d,h1d)
      double precision :: v2sub(*)
      call cute_t_transpt2_7326( v2sub,scratch, h3d,h2d,p6d,h7d)
      do p4=1,p4d
      do p6=1,p6d
      do p5=1,p5d
      do h3h2=1,h3d*h2d
      do h1=1,h1d
      do h7=1,h7d
       triplesx(h1,h3h2,p5,p6,p4)=triplesx(h1,h3h2,p5,p6,p4)
     &   + t2sub(h7,p4,p5,h1)*scratch(h7,h3h2,p6)
      enddo
      enddo
      enddo
      enddo
      enddo
      enddo
      end
!
      subroutine cutensor_t_d2_1(h3d,h2d,h1d,p6d,p5d,p4d,p7d,
     &               triplesx,t2sub,v2sub,scratch)
      implicit none
      integer :: h3d,h2d,h1d,p6d,p5d,p4d,p7d
      integer :: h3,h2,h1,p6,p5,p4,p7
      double precision :: triplesx(h3d,h2d,h1d,p6d,p5d,p4d)
      double precision :: scratch(p7d,h2d,h1d,p4d),t2sub(*)
      double precision :: v2sub(p7d,h3d,p6d,p5d)
      call cute_t_transpt2_7124( t2sub,scratch, p4d, h1d, h2d, p7d)
      do p4=1,p4d
      do p5=1,p5d
      do p6=1,p6d
      do h3=1,h3d
      do h1=1,h1d
      do h2=1,h2d
      do p7=1,p7d
       triplesx(h3,h2,h1,p6,p5,p4)=triplesx(h3,h2,h1,p6,p5,p4)
     &   - scratch(p7,h2,h1,p4)*v2sub(p7,h3,p6,p5)
      enddo
      enddo
      enddo
      enddo
      enddo
      enddo
      enddo
      end
!
      subroutine cutensor_t_d2_2(h3d,h2d,h1d,p6d,p5d,p4d,p7d,
     &               triplesx,t2sub,v2sub,scratch)
      implicit none
      integer :: h3d,h2d,h1d,p6d,p5d,p4d,p7d
      integer :: h3,h2,h1,p6,p5,p4,p7
      integer :: h2h1
      double precision :: triplesx(h2d*h1d,h3d,p6d,p5d,p4d)
      double precision :: scratch(h2d*h1d,p4d,p7d),t2sub(*)
      double precision :: v2sub(p7d,h3d,p6d,p5d)
      call cute_t_transpt2( t2sub,scratch, h2d,h1d,p4d,p7d)
      do p4=1,p4d
      do p5=1,p5d
      do p6=1,p6d
      do h3=1,h3d
      do p7=1,p7d
      do h2h1=1,h2d*h1d
       triplesx(h2h1,h3,p6,p5,p4)=triplesx(h2h1,h3,p6,p5,p4)
     &   - scratch(h2h1,p4,p7)*v2sub(p7,h3,p6,p5)
      enddo
      enddo
      enddo
      enddo
      enddo
      enddo
      end
!
      subroutine cutensor_t_d2_3(h3d,h2d,h1d,p6d,p5d,p4d,p7d,
     &               triplesx,t2sub,v2sub)
      implicit none
      integer :: h3d,h2d,h1d,p6d,p5d,p4d,p7d
      integer :: h3,h2,h1,p6,p5,p4,p7
      double precision :: triplesx(h2d,h3d,h1d,p6d,p5d,p4d)
      double precision :: t2sub(p7d,p4d,h1d,h2d)
      double precision :: v2sub(p7d,h3d,p6d,p5d)
      double precision :: scratch(p7d,h2d,h1d,p4d)
      call cute_t_transpt2_7124( t2sub,scratch, p4d,h1d,h2d,p7d)
      do p4=1,p4d
      do p5=1,p5d
      do p6=1,p6d
      do h1=1,h1d
      do h3=1,h3d
      do h2=1,h2d
      do p7=1,p7d
       triplesx(h2,h3,h1,p6,p5,p4)=triplesx(h2,h3,h1,p6,p5,p4) +
     &        scratch(p7,h2,h1,p4)*v2sub(p7,h3,p6,p5)
      enddo
      enddo
      enddo
      enddo
      enddo
      enddo
      enddo
      end
!
      subroutine cutensor_t_d2_4(h3d,h2d,h1d,p6d,p5d,p4d,p7d,
     &               triplesx,t2sub,v2sub,scratch)
      implicit none
      integer :: h3d,h2d,h1d,p6d,p5d,p4d,p7d
      integer :: h3,h2,h1,p6,p5,p4,p7
      double precision :: triplesx(h3d,h2d,h1d,p6d,p4d,p5d)
      double precision :: scratch(p7d,h2d,h1d,p4d),t2sub(*)
      double precision :: v2sub(p7d,h3d,p6d,p5d)
      call cute_t_transpt2_7124( t2sub,scratch, p4d, h1d, h2d, p7d)
      do p5=1,p5d
      do p4=1,p4d
      do p6=1,p6d
      do h1=1,h1d
      do h2=1,h2d
      do h3=1,h3d
      do p7=1,p7d
       triplesx(h3,h2,h1,p6,p4,p5)=triplesx(h3,h2,h1,p6,p4,p5)
     &   + scratch(p7,h2,h1,p4)*v2sub(p7,h3,p6,p5)
      enddo
      enddo
      enddo
      enddo
      enddo
      enddo
      enddo
      end
!
      subroutine cutensor_t_d2_5(h3d,h2d,h1d,p6d,p5d,p4d,p7d,
     &               triplesx,t2sub,v2sub,scratch)
      implicit none
      integer :: h3d,h2d,h1d,p6d,p5d,p4d,p7d
      integer :: h3,h2,h1,p6,p5,p4,p7
      integer :: h2h1
      double precision :: triplesx(h2d*h1d,h3d,p6d,p4d,p5d)
      double precision :: scratch(h2d*h1d,p4d,p7d),t2sub(*)
      double precision :: v2sub(p7d,h3d,p6d,p5d)
      call cute_t_transpt2( t2sub,scratch, h2d,h1d,p4d,p7d)
      do p5=1,p5d
      do p4=1,p4d
      do p6=1,p6d
      do h3=1,h3d
      do p7=1,p7d
      do h2h1=1,h2d*h1d
       triplesx(h2h1,h3,p6,p4,p5)=triplesx(h2h1,h3,p6,p4,p5)
     &   + scratch(h2h1,p4,p7)*v2sub(p7,h3,p6,p5)
      enddo
      enddo
      enddo
      enddo
      enddo
      enddo
      end
!
      subroutine cutensor_t_d2_6(h3d,h2d,h1d,p6d,p5d,p4d,p7d,
     &               triplesx,t2sub,v2sub,scratch)
      implicit none
      integer :: h3d,h2d,h1d,p6d,p5d,p4d,p7d
      integer :: h3,h2,h1,p6,p5,p4,p7
      double precision :: triplesx(h2d,h3d,h1d,p6d,p4d,p5d)
      double precision :: t2sub(p7d,p4d,h1d,h2d)
      double precision :: scratch(p7d,h2d,h1d,p4d)
      double precision :: v2sub(p7d,h3d,p6d,p5d)
      call cute_t_transpt2_7124( t2sub,scratch, p4d,h1d,h2d,p7d)
      do p5=1,p5d
      do p4=1,p4d
      do p6=1,p6d
      do h1=1,h1d
      do h3=1,h3d
      do h2=1,h2d
      do p7=1,p7d
       triplesx(h2,h3,h1,p6,p4,p5)=triplesx(h2,h3,h1,p6,p4,p5)
     &   - scratch(p7,h2,h1,p4)*v2sub(p7,h3,p6,p5)
      enddo
      enddo
      enddo
      enddo
      enddo
      enddo
      enddo
      end
!
      subroutine cutensor_t_d2_7(h3d,h2d,h1d,p6d,p5d,p4d,p7d,
     &               triplesx,t2sub,v2sub)
      implicit none
      integer :: h3d,h2d,h1d,p6d,p5d,p4d,p7d
      integer :: h3,h2,h1,p6,p5,p4,p7
      double precision :: triplesx(h3d,h2d,h1d,p4d,p6d,p5d)
      double precision :: t2sub(p7d,p4d,h1d,h2d)
      double precision :: scratch(p7d,h2d,h1d,p4d)
      double precision :: v2sub(p7d,h3d,p6d,p5d)
      call cute_t_transpt2_7124( t2sub,scratch, p4d,h1d,h2d,p7d)
      do p5=1,p5d
      do p6=1,p6d
      do p4=1,p4d
      do h1=1,h1d
      do h2=1,h2d
      do h3=1,h3d
      do p7=1,p7d
       triplesx(h3,h2,h1,p4,p6,p5)=triplesx(h3,h2,h1,p4,p6,p5)
     &   - scratch(p7,h2,h1,p4)*v2sub(p7,h3,p6,p5)
      enddo
      enddo
      enddo
      enddo
      enddo
      enddo
      enddo
      end
!
      subroutine cutensor_t_d2_8(h3d,h2d,h1d,p6d,p5d,p4d,p7d,
     &               triplesx,t2sub,v2sub,scratch)
      implicit none
      integer :: h3d,h2d,h1d,p6d,p5d,p4d,p7d
      integer :: h3,h2,h1,p6,p5,p4,p7
      integer :: h2h1
      double precision :: triplesx(h2d*h1d,h3d,p4d,p6d,p5d)
      double precision :: scratch(h2d*h1d,p4d,p7d),t2sub(*)
      double precision :: v2sub(p7d,h3d,p6d,p5d)
      call cute_t_transpt2( t2sub,scratch, h2d,h1d,p4d,p7d)
      do p5=1,p5d
      do p6=1,p6d
      do p4=1,p4d
      do h3=1,h3d
      do p7=1,p7d
      do h2h1=1,h2d*h1d
       triplesx(h2h1,h3,p4,p6,p5)=triplesx(h2h1,h3,p4,p6,p5)
     &   - scratch(h2h1,p4,p7)*v2sub(p7,h3,p6,p5)
      enddo
      enddo
      enddo
      enddo
      enddo
      enddo
      end
!
      subroutine cutensor_t_d2_9(h3d,h2d,h1d,p6d,p5d,p4d,p7d,
     &               triplesx,t2sub,v2sub,scratch)
      implicit none
      integer :: h3d,h2d,h1d,p6d,p5d,p4d,p7d
      integer :: h3,h2,h1,p6,p5,p4,p7
      double precision :: triplesx(h2d,h3d,h1d,p4d,p6d,p5d)
      double precision :: t2sub(p7d,p4d,h1d,h2d)
      double precision :: scratch(p7d,h2d,h1d,p4d)
      double precision :: v2sub(p7d,h3d,p6d,p5d)
      call cute_t_transpt2_7124( t2sub,scratch, p4d,h1d,h2d,p7d)
      do p5=1,p5d
      do p6=1,p6d
      do p4=1,p4d
      do h1=1,h1d
      do h3=1,h3d
      do h2=1,h2d
      do p7=1,p7d
       triplesx(h2,h3,h1,p4,p6,p5)=triplesx(h2,h3,h1,p4,p6,p5)
     &    + scratch(p7,h2,h1,p4)*v2sub(p7,h3,p6,p5)
      enddo
      enddo
      enddo
      enddo
      enddo
      enddo
      enddo
      end

      subroutine cute_t_transpt2(t2sub,scratch,h1d,p5d,p4d,h7d)
      implicit none
      integer :: h1d,p5d,p4d,h7d
      double precision :: t2sub(h7d,p4d,p5d,h1d)
      double precision :: scratch(h1d,p5d,p4d,h7d)
      integer :: h1,p5,p4,h7
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


      subroutine cute_t_transpt2_7124(t2sub,scratch,p4d,h1d,h2d,p7d)
      implicit none
      integer :: p4d,h1d,h2d,p7d
      double precision :: t2sub(p7d,p4d,h1d,h2d)
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


      subroutine cute_t_transpt2_7326(v2sub,scratch,h3d,h2d,p6d,h7d)
      implicit none
      integer :: h3d,h2d,p6d,h7d
      double precision :: v2sub(h3d,h2d,p6d,h7d)
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

      end module tce_ccsd_t_cutensor_nodts
