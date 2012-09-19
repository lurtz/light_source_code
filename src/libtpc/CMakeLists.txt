
SET(TPC_SRC
  lss_lib.c
  memory_handler.c
  nnls.c
)

add_library(libtpc ${TPC_SRC})
