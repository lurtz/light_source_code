SET(Exercise10_SRC
  MeshObj.cpp
  ObjLoader.cpp
)

#The rule to build the executable program 
ADD_EXECUTABLE(light_sources main.cpp opengl.cpp ${Exercise10_SRC})
TARGET_LINK_LIBRARIES(light_sources
  ${OpenCV_LIBS}
  ${OpenGL_LIBRARIES}
  ${GLUT_LIBRARIES}
  ${GLEW_LIBRARIES}
  ${GL_LIBRARY}
  ${OPENGL_LIBRARIES}
)
