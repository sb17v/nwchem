      subroutine tce_sort_4_y(u,s,a,b,c,d,i,j,k,l,f)
      use cudafor
      use cutensor_v2
      implicit none
      integer, intent(in) :: a,b,c,d
      integer, intent(in) :: i,j,k,l
      double precision, intent(out) :: s(a*b*c*d)
      double precision, intent(in) :: u(a*b*c*d)
      double precision, intent(in) :: f
#include "stdio.fh"
#include "errquit.fh"
      type(cutensorHandle) :: handle
      type(cutensorTensorDescriptor) :: descIn,descOut
      integer(4), parameter :: ndim = 4
      integer(4), dimension(ndim) :: modesIn, modesOut
      integer(8), dimension(ndim) :: extents
      integer(8), dimension(ndim) :: stridesIn, stridesOut
      integer(4), parameter :: ialign = 128
      type(cutensorStatus) :: cstat
      type(cutensorOperationDescriptor) :: pdesc
      type(cutensorComputeDescriptor) :: descCompute
      type(cutensorPlan) :: plan
      type(cutensorPlanPreference) :: pref
      integer(4), dimension(ndim) :: modeIn, modeOut
      integer :: ii

      cstat = cutensorCreate(handle)
      if (cstat.ne.CUTENSOR_STATUS_SUCCESS) then
        write(LuOut,*) cutensorGetErrorString(cstat)
        CALL ERRQUIT('tce_sort_4_cutensor',cstat,UNKNOWN_ERR)
      end if

      extents(1) = a
      extents(2) = b
      extents(3) = c
      extents(4) = d
      modesIn(1) = 1
      modesIn(2) = 2
      modesIn(3) = 3
      modesIn(4) = 4
      modesOut(1) = i
      modesOut(2) = j
      modesOut(3) = k
      modesOut(4) = l
      stridesIn(ndim) = 1
      do ii = ndim-1, 1, -1
        stridesIn(ii) = stridesIn(ii+1) * extents(ii+1)
      end do
      stridesOut(ndim) = 1
      do ii = ndim-1, 1, -1
        stridesOut(ii) = stridesOut(ii+1) * extents(modesOut(ii+1))
      end do

      cstat = cutensorCreateTensorDescriptor(handle, descIn,
     &                           ndim, extents, stridesIn,
     &                            cuTensorDataType(CUDA_R_64F), ialign)
      if (cstat.ne.CUTENSOR_STATUS_SUCCESS) then
        write(LuOut,*) cutensorGetErrorString(cstat)
        CALL ERRQUIT('tce_sort_4_cutensor',cstat,UNKNOWN_ERR)
      end if

      cstat = cutensorCreateTensorDescriptor(handle, descOut,
     &                           ndim, extents, stridesOut,
     &                           cuTensorDataType(CUDA_R_64F), ialign)
      if (cstat.ne.CUTENSOR_STATUS_SUCCESS) then
        write(LuOut,*) cutensorGetErrorString(cstat)
        CALL ERRQUIT('tce_sort_4_cutensor',cstat,UNKNOWN_ERR)
      end if

      descCompute = CUTENSOR_COMPUTE_DESC_64F
      cstat = cutensorCreatePermutation(handle, pdesc,
     &        descIn, modeIn, CUTENSOR_OP_IDENTITY,
     &        descOut, modeOut, descCompute) 
      if (cstat.ne.CUTENSOR_STATUS_SUCCESS) print *,cutensorGetErrorString(cstat)
      cstat = cutensorCreatePlanPreference(handle, pref,
     &              CUTENSOR_ALGO_DEFAULT, CUTENSOR_JIT_MODE_NONE)
      if (cstat.ne.CUTENSOR_STATUS_SUCCESS) print *,cutensorGetErrorString(cstat)
      cstat = cutensorCreatePlan(handle, plan, pdesc, pref, 0)
      if (cstat.ne.CUTENSOR_STATUS_SUCCESS) print *,cutensorGetErrorString(cstat)
      cstat = cutensorPermute(handle, plan, f, u, s, 0)
      if (cstat.ne.CUTENSOR_STATUS_SUCCESS) print *,cutensorGetErrorString(cstat)

      cstat = cutensorDestroyPlan(plan)
      if (cstat.ne.CUTENSOR_STATUS_SUCCESS) print *,cutensorGetErrorString(cstat)
      cstat = cutensorDestroyPlanPreference(pref)
      if (cstat.ne.CUTENSOR_STATUS_SUCCESS) print *,cutensorGetErrorString(cstat)
      cstat = cutensorDestroyOperationDescriptor(pdesc)
      if (cstat.ne.CUTENSOR_STATUS_SUCCESS) print *,cutensorGetErrorString(cstat)
      cstat = cutensorDestroyTensorDescriptor(descOut)
      if (cstat.ne.CUTENSOR_STATUS_SUCCESS) print *,cutensorGetErrorString(cstat)
      cstat = cutensorDestroyTensorDescriptor(descIn)
      if (cstat.ne.CUTENSOR_STATUS_SUCCESS) print *,cutensorGetErrorString(cstat)
      cstat = cutensorDestroy(handle)
      if (cstat.ne.CUTENSOR_STATUS_SUCCESS) print *,cutensorGetErrorString(cstat)

      end
