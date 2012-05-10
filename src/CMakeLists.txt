SET(Exercise10_SRC
  MeshObj.cpp
  ObjLoader.cpp
)

#The rule to build the executable program 
ADD_EXECUTABLE(light_sources main.cpp ${Exercise10_SRC})
TARGET_LINK_LIBRARIES(light_sources ${OpenCV_LIBS})
