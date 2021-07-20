#define CHUNK 16
      subroutine ccsd_t_transpt2_stdpar(t2sub,scratch,h1d,p5d,p4d,h7d)
      implicit none
      integer, intent(in) :: h1d,p5d,p4d,h7d
      double precision, intent(in) :: t2sub(h7d,p4d,p5d,h1d)
      double precision, intent(out) :: scratch(h1d,p5d,p4d,h7d)
      integer :: h1,p5,p4,h7
      integer :: pp5,hh1
      do concurrent (h7=1:h7d, p4=1:p4d)
            do pp5=1,p5d,CHUNK
               do hh1=1,h1d,CHUNK
                  do p5=pp5,min(pp5+CHUNK-1,p5d)
                     do h1=hh1,min(hh1+CHUNK-1,h1d)
                        scratch(h1,p5,p4,h7)=t2sub(h7,p4,p5,h1)
                     enddo
                  enddo
               enddo
            enddo
      enddo
      end

      subroutine ccsd_t_transpt2_7124_stdpar(t2sub,scratch,
     &                                       p4d,h1d,h2d,p7d)
      implicit none
      integer, intent(in) :: p4d,h1d,h2d,p7d
      double precision, intent(in) :: t2sub(p7d,p4d,h1d,h2d)
      double precision, intent(out) :: scratch(p7d,h2d,h1d,p4d)
      integer :: p4,h1,h2,p7
      integer :: hh2,pp7
      do concurrent (p4=1:p4d, h1=1:h1d)
            do hh2=1,h2d,CHUNK
               do pp7=1,p7d,CHUNK
                  do h2=hh2,min(hh2+CHUNK-1,h2d)
                     do p7=pp7,min(pp7+CHUNK-1,p7d)
                        scratch(p7,h2,h1,p4)=t2sub(p7,p4,h1,h2)
                     enddo
                  enddo
               enddo
            enddo
      enddo
      end

      subroutine ccsd_t_transpt2_7326_stdpar(v2sub,scratch,
     &                                       h3d,h2d,p6d,h7d)
      implicit none
      integer, intent(in) :: h3d,h2d,p6d,h7d
      double precision, intent(in) :: v2sub(h3d,h2d,p6d,h7d)
      double precision, intent(out) :: scratch(h7d,h3d,h2d,p6d)
      integer :: h3,h2,p6,h7
      integer :: hh3,hh7
      do (p6=1:p6d, h2=1:h2d)
            do hh3=1,h3d,CHUNK
               do hh7=1,h7d,CHUNK
                  do h3=hh3,min(hh3+CHUNK-1,h3d)
                     do h7=hh7,min(hh7+CHUNK-1,h7d)
                        scratch(h7,h3,h2,p6)=v2sub(h3,h2,p6,h7)
                     enddo
                  enddo
               enddo
            enddo
      enddo
      end
