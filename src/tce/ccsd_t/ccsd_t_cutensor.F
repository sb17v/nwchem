
      module tce_ccsd_t_cutensor_nodts

      contains

      SUBROUTINE ccsd_t_cutensor(d_t1,k_t1_offset,d_t2,k_t2_offset,
     &            d_v2,k_v2_offset,energy1,energy2,size_t1)
      IMPLICIT NONE
#include "global.fh"
#include "mafdecls.fh"
#include "util.fh"
#include "errquit.fh"
#include "tce.fh"
#include "tce_main.fh"
#include "ccsd_t_ps.fh"
      integer d_t1
      integer k_t1_offset
      integer d_t2
      integer k_t2_offset
      integer d_v2
      integer k_v2_offset
      integer t_h1b, t_h1
      integer t_h2b, t_h2
      integer t_h3b, t_h3
      integer t_p4b, t_p4
      integer t_p5b, t_p5
      integer t_p6b, t_p6
      double precision, allocatable :: f_singles(:),f_doubles(:)
      integer alloc_error
      integer size,i
      integer nxtask0
      integer next
      integer nprocs
      integer count
      integer offset_p4,offset_p5,offset_p6
      integer offset_h1,offset_h2,offset_h3
      integer range_p4,range_p5,range_p6
      integer range_h1,range_h2,range_h3
! - T1/X1 LOCALIZATION -------------------
      integer l_t1_local,k_t1_local
      integer size_t1
! ---------------------------------------
      double precision energy(2)
      double precision energy1,energy2
      integer k_aux,k_list,l_aux,l_list
      integer ccsd_t_6tasks,tot_task,w_thresh,task_thresh
      integer k,iptr,k_grain
      double precision wall
      integer tasks_skip
      integer n_progr,pct_progr
      parameter(n_progr=20)
      logical i_progr(n_progr)
      external nxtask0,ccsd_t_6tasks
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
      size=(range_p4**3)*(range_h1**3)

      allocate( f_singles(1:size), stat=alloc_error)
      if (alloc_error.ne.0) then
        call errquit('ccsd_t: MA error sgl',0,MA_ERR)
      endif
      allocate( f_doubles(1:size), stat=alloc_error)
      if (alloc_error.ne.0) then
        call errquit('ccsd_t: MA error dbl',0,MA_ERR)
      endif
      wall=-util_wallsec()
      tot_task= ccsd_t_6tasks(restricted,noab,nvab,
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
      call ccsd_t_neword(tot_task, w_thresh,task_thresh,
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
            call ccsd_t_loop(k,energy1,energy2,
     &              int_mb(k_list),int_mb(k_range),int_mb(k_offset),
     &              f_singles,f_doubles,
     &              k_t1_local,k_t1_offset,
     &              d_t2,d_v2,k_t2_offset,k_v2_offset,
     &              restricted,k_evl_sorted,size)
!
            if(ga_nodeid().eq.2) then
               pct_progr=(k*n_progr)/tot_task
               if(i_progr(pct_progr)) then
                  i_progr(pct_progr)=.false.
                  write(6,'(a,i5,a,i4,a,f15.1,a,f9.1)')
     &                  '0task ',k,'  done ',
     &                  int((k*100d0)/tot_task),'%  at',
     &                  wall+util_wallsec(),' sec, (size)^1/6= ',
     &                  (size)**(1d0/6d0)
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
               call ccsd_t_loop(k,energy1,energy2,
     &                 int_mb(k_list),int_mb(k_range),int_mb(k_offset),
     &                 f_singles,f_doubles,
     &                 k_t1_local,k_t1_offset,
     &                 d_t2,d_v2,k_t2_offset,k_v2_offset,
     &                 restricted,k_evl_sorted,size)
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
     &                     (size)**(1d0/6d0)
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
      return
      end

      subroutine ccsd_t_esum(a_singles, a_doubles, restricted,
     &                       h1b,h2b,h3b,p4b,p5b,p6b,
     &                       o_h1,o_h2,o_h3,
     &                       o_p4,o_p5,o_p6,
     &                       r_h1,r_h2,r_h3,
     &                       r_p4,r_p5,r_p6,
     &                       energy1,energy2)
      implicit none
      integer h1b, h2b, h3b, p4b, p5b, p6b
      double precision o_h1(*),o_h2(*),o_h3(*)
      double precision o_p4(*),o_p5(*),o_p6(*)
      integer r_h1,r_h2,r_h3
      integer r_p4,r_p5,r_p6
      double precision a_singles(*)
      double precision a_doubles(*)
      logical restricted
      double precision energy1,energy2
      call ccsd_t_dot(a_singles,a_doubles, restricted,
     &  h1b,h2b,h3b,p4b,p5b,p6b,
     &  o_h1,o_h2,o_h3,o_p4,o_p5,o_p6,
     &  r_h1,r_h2,r_h3,r_p4,r_p5,r_p6,
     &  energy1,energy2)
      end



      subroutine ccsd_t_loop(k,energy1,energy2,
     &     k_list,k_range,k_offset,a_singles,a_doubles,
     &     k_t1_local,k_t1_offset,d_t2,d_v2,k_t2_offset,k_v2_offset,
     &     restricted,k_evl_sorted,size)
      implicit none
#include "mafdecls.fh"
#include "ccsd_t_ps.fh"
      integer k
      double precision energy1,energy2
      integer k_list(7,*)
      integer k_range(*),k_offset(*)
      double precision a_singles(:),a_doubles(:)
      logical restricted
      integer k_evl_sorted
      integer k_t1_local,k_t1_offset
      integer d_t2,d_v2
      integer k_t2_offset,k_v2_offset
      integer size
!
      integer t_p4b,t_p5b,t_p6b,t_h1b,t_h2b,t_h3b
      integer range_p4,range_p5,range_p6
      integer range_h1,range_h2,range_h3
      integer offset_p4,offset_p5,offset_p6
      integer offset_h1,offset_h2,offset_h3
!
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
!
      size = range_p4 * range_p5 * range_p6
     &     * range_h1 * range_h2 * range_h3
      a_singles = 0.0d0
      a_doubles = 0.0d0
      if (otceps) call pstat_on(ps_cctsng)
      call ccsd_t_singles_l(a_singles,
     &                 k_t1_local,d_v2,k_t1_offset,k_v2_offset,
     &                 t_h1b,t_h2b,t_h3b,t_p4b,t_p5b,t_p6b,2)
      if (otceps) call pstat_off(ps_cctsng)


      if (otceps) call pstat_on(ps_cctdbl)      
      call ccsd_t_doubles_l(a_doubles,
     &                 d_t2,d_v2,k_t2_offset,k_v2_offset,
     &                 t_h1b,t_h2b,t_h3b,t_p4b,t_p5b,t_p6b,2)
      if (otceps) call pstat_off(ps_cctdbl)
      call ccsd_t_esum(a_singles,a_doubles,restricted,
     &                 t_h1b,t_h2b,t_h3b,t_p4b,t_p5b,t_p6b,
     &                 dbl_mb(offset_h1+1),dbl_mb(offset_h2+1),
     &                 dbl_mb(offset_h3+1),dbl_mb(offset_p4+1),
     &                 dbl_mb(offset_p5+1),dbl_mb(offset_p6+1),
     &                 range_h1,range_h2,range_h3,
     &                 range_p4,range_p5,range_p6,
     &                 energy1,energy2)
      return
      end

      integer function ccsd_t_6tasks(
     &     restricted,noab,nvab,kspin,ksym)
!
!     $Id$
!
      implicit none
      logical restricted
      integer noab,nvab
      integer kspin(*),ksym(*)
!
      integer t_p4b,t_p5b,t_p6b
      integer t_h1b,t_h2b,t_h3b
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
      return
      end
!
      subroutine ccsd_t_neword(tot_task, 
     &     w_thresh,task_thresh,
     &     restricted,noab,nvab,kspin,ksym,
     &     krange,kaux,klist)
      implicit none
#include "errquit.fh"
#include "mafdecls.fh"
#include "global.fh"
      logical restricted
      integer w_thresh,task_thresh
      integer tot_task
      integer noab,nvab
      integer kspin(*),ksym(*)
      integer krange(*)
      integer kaux(7,*),klist(7,*)
!
      integer t_p4b,t_p5b,t_p6b
      integer t_h1b,t_h2b,t_h3b
      integer i,j,m
      integer wl_max,wl_min
      integer range_p4,range_p5,range_p6,
     &     range_h1,range_h2,range_h3
      integer nsplits,ii,w_in
      integer highindx
      parameter (highindx=7)
      double precision rr456,rr4561,rr45612
!
      task_thresh=tot_task
      m=0
      do t_p4b = noab+1,noab+nvab
       range_p4 = krange(t_p4b)
       do t_p5b = t_p4b,noab+nvab
        range_p5 = krange(t_p5b)
        do t_p6b = t_p5b,noab+nvab
         range_p6 = krange(t_p6b)
         rr456=range_p4*range_p5*range_p6
         do t_h1b = 1,noab
          range_h1 = krange(t_h1b)
            rr4561=rr456*range_h1
          do t_h2b = t_h1b,noab
             range_h2 = krange(t_h2b)
             rr45612 = rr4561*range_h2
           do t_h3b = t_h2b,noab
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
            range_h3 = krange(t_h3b)
            kaux(highindx,m)=rr45612*range_h3
            endif
            endif
            endif
           enddo
          enddo
         enddo
        enddo
       enddo
      enddo
#ifdef DEBUG
      do i=1,tot_task
         write(6,12) i,(kaux(j,i),j=1,highindx)
 12      format(I7,6I8,I20)
      enddo
#endif
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
#if 1
      if(((wl_max-wl_min)*100d0)/wl_max.lt.1d0) then 
      do m=1,tot_task
         do j=1,highindx
            klist(j,m)=kaux(j,m)
         enddo
      enddo
      return
      endif
#endif

!
! simple sorting
!
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

#ifdef DEBUG
      write(6,*) ' sort1 '
      do i=1,tot_task
         write(6,11) i,(klist(j,i),j=1,6)
      enddo
#endif
      nsplits=16
      do ii=nsplits,1,-1
         w_in=wl_min+((wl_max-wl_min)*(ii-1))/nsplits
          call ccsd_sillysort(w_in,
     &     kaux,klist, tot_task,m,highindx)
       enddo
      call ccsd_sillysort(0,
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
#ifdef DEBUG
      write(6,*) ' sorting ok '
      do m=1,tot_task
         write(6,11) m,(klist(j,m),j=1,6)
 11      format(I7,6I8)
      enddo
#endif
      return
      end
      subroutine ccsd_sillysort(value,
     &     kaux,klist,n,found,highindx )
      implicit none
!
! very simple sorting
!
      integer value
      integer n
      integer found
      integer highindx
      integer kaux(highindx,*)
      integer klist(highindx,*)
!
      integer i,j,m
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
      return
      end

      end module tce_ccsd_t_cutensor_nodts
