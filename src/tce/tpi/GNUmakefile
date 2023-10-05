#$Id$

OBJ_OPTIMIZE = add_block.o get_block.o get_block_ind.o put_block.o \
	tce_profile_agemm.o tce_profile_nxtask.o \
	tce_profile_sort.o tce_profile_sortacc.o
               
OBJ = tce_profile.o

LIB_INCLUDES = -I../include

LIBRARY = libtce.a

USES_BLAS = tce_profile_agemm.F

LIB_DEFINES = -DDEBUG_PRINT

HEADERS = 

include ../../config/makefile.h
include ../../config/makelib.h

