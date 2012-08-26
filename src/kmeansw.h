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
                   int flags, OutputArray _centers );
}

#endif /* __CV_KMEANS_W__ */