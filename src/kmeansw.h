#ifndef __CV_KMEANS_W__
#define __CV_KMEANS_W__

#ifdef OPENCV_OLD_INCLUDES
  #include <cv.h>
#else
  #include <opencv2/core/core.hpp>
#endif

// taken and modified from opencv2.3 from debian sid (2012.08.26)
// each point has besides its coordinates a weight

namespace cv {
double kmeansw( InputArray _data, int K,
                   InputOutputArray _bestLabels,
                   TermCriteria criteria, int attempts,
                   int flags, OutputArray _centers,
                   const vector<double>& weights);
}

void testkmeansw();

template<typename T, int dim>
std::ostream& operator<<(std::ostream& out, cv::Vec<T, dim> vec) {
  out << "cv::Vec<" << typeid(T).name() << ", " << dim << ">(";
  for (unsigned int i = 0; i < dim-1; i++)
    out << vec[i] << ", ";
  out << vec[dim-1] << ")";
  return out;
}

#endif /* __CV_KMEANS_W__ */