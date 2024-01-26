      subroutine tce_sort_4_y(u,s,a,b,c,d,i,j,k,l,f)
      use cudafor
      use cutensor_v2
      implicit none
      integer, intent(in) :: a,b,c,d
      integer, intent(in) :: i,j,k,l
      double precision, intent(out) :: s(a*b*c*d)
      double precision, intent(in) :: u(a*b*c*d)
      double precision, intent(in) :: f
!#include "errquit.fh"
      type(cutensorHandle) :: handle
      type(cutensorTensorDescriptor) :: descIn,descOut
      integer(4), parameter :: ndim = 4
      integer(4), dimension(ndim) :: modesIn, modesOut
      integer(8), dimension(ndim) :: extentsIn, extentsOut
      integer(8), dimension(ndim) :: stridesIn, stridesOut
      integer(4), parameter :: ialign = 128
      type(cutensorStatus) :: cstat
      type(cutensorOperationDescriptor) :: pdesc
      type(cutensorComputeDescriptor) :: descCompute
      type(cutensorPlan) :: plan
      type(cutensorPlanPreference) :: pref
      integer :: ii

      cstat = cutensorCreate(handle)
      if (cstat.ne.CUTENSOR_STATUS_SUCCESS) then
        write(6,*) cutensorGetErrorString(cstat)
        !CALL ERRQUIT('tce_sort_4_cutensor',cstat,UNKNOWN_ERR)
      end if

      extentsIn(1) = a
      extentsIn(2) = b
      extentsIn(3) = c
      extentsIn(4) = d
      modesIn(1) = 1
      modesIn(2) = 2
      modesIn(3) = 3
      modesIn(4) = 4
      modesOut(1) = i
      modesOut(2) = j
      modesOut(3) = k
      modesOut(4) = l
      extentsOut(1) = extentsIn(modesOut(1))
      extentsOut(2) = extentsIn(modesOut(2))
      extentsOut(3) = extentsIn(modesOut(3))
      extentsOut(4) = extentsIn(modesOut(4))
      stridesIn(ndim) = 1
      do ii = ndim-1, 1, -1
        stridesIn(ii) = stridesIn(ii+1) * extentsIn(ii+1)
      end do
      stridesOut(ndim) = 1
      do ii = ndim-1, 1, -1
        stridesOut(ii) = stridesOut(ii+1) * extentsOut(ii+1)
      end do

      cstat = cutensorCreateTensorDescriptor(handle, descIn,
     &                           ndim, extentsIn, stridesIn,
     &                            cuTensorDataType(CUDA_R_64F), ialign)
      if (cstat.ne.CUTENSOR_STATUS_SUCCESS) then
        write(6,*) cutensorGetErrorString(cstat)
        !CALL ERRQUIT('tce_sort_4_cutensor',cstat,UNKNOWN_ERR)
        STOP
      end if

      cstat = cutensorCreateTensorDescriptor(handle, descOut,
     &                           ndim, extentsOut, stridesOut,
     &                           cuTensorDataType(CUDA_R_64F), ialign)
      if (cstat.ne.CUTENSOR_STATUS_SUCCESS) then
        write(6,*) cutensorGetErrorString(cstat)
        !CALL ERRQUIT('tce_sort_4_cutensor',cstat,UNKNOWN_ERR)
        STOP
      end if

      descCompute = CUTENSOR_COMPUTE_DESC_64F
      cstat = cutensorCreatePermutation(handle, pdesc,
     &        descIn, modesIn, CUTENSOR_OP_IDENTITY,
     &        descOut, modesOut, descCompute) 
      if (cstat.ne.CUTENSOR_STATUS_SUCCESS) then
          print *,cutensorGetErrorString(cstat)
          stop
      endif
      cstat = cutensorCreatePlanPreference(handle, pref,
     &              CUTENSOR_ALGO_DEFAULT, CUTENSOR_JIT_MODE_NONE)
      if (cstat.ne.CUTENSOR_STATUS_SUCCESS) then
          print *,cutensorGetErrorString(cstat)
          stop
      endif
      cstat = cutensorCreatePlan(handle, plan, pdesc, pref, 0)
      if (cstat.ne.CUTENSOR_STATUS_SUCCESS) then
          print *,cutensorGetErrorString(cstat)
          print *,'args:',a,b,c,d,i,j,k,l
          stop
      endif
      cstat = cutensorPermute(handle, plan, f, u, s, 0)
      if (cstat.ne.CUTENSOR_STATUS_SUCCESS) then
          print *,cutensorGetErrorString(cstat)
          stop
      endif

      cstat = cutensorDestroyPlan(plan)
      if (cstat.ne.CUTENSOR_STATUS_SUCCESS) then
          print *,cutensorGetErrorString(cstat)
          stop
      endif
      cstat = cutensorDestroyPlanPreference(pref)
      if (cstat.ne.CUTENSOR_STATUS_SUCCESS) then
          print *,cutensorGetErrorString(cstat)
          stop
      endif
      cstat = cutensorDestroyOperationDescriptor(pdesc)
      if (cstat.ne.CUTENSOR_STATUS_SUCCESS) then
          print *,cutensorGetErrorString(cstat)
          stop
      endif
      cstat = cutensorDestroyTensorDescriptor(descOut)
      if (cstat.ne.CUTENSOR_STATUS_SUCCESS) then
          print *,cutensorGetErrorString(cstat)
          stop
      endif
      cstat = cutensorDestroyTensorDescriptor(descIn)
      if (cstat.ne.CUTENSOR_STATUS_SUCCESS) then
          print *,cutensorGetErrorString(cstat)
          stop
      endif
      cstat = cutensorDestroy(handle)
      if (cstat.ne.CUTENSOR_STATUS_SUCCESS) then
          print *,cutensorGetErrorString(cstat)
          stop
      endif

      end
