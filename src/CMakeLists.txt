##Linking with libcutil (expected to be in ${CUDA_SDK_ROOT_DIR}/C/lib)
##LINK_LIBRARIES("-L/${CUDA_SDK_ROOT_DIR}/C/lib -lcutil")

##Include the FindCUDA script
#INCLUDE(FindCUDA)

##The rule to build the executable program 
#CUDA_ADD_EXECUTABLE(pba pba/pba3DHost.cu main.cpp)
#TARGET_LINK_LIBRARIES(pba ${OpenCV_LIBS})
