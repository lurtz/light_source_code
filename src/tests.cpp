#include "tests.h"
#include "kmeansw.h"
#include "utils.h"

bool test_gsl_vector_iterator() {
  const unsigned int size = 10;
  gsl::vector<3,3> v(size);
  for (size_t i = 0; i < size; i++)
    v.get(i) = static_cast<double>(i);
  double x = sum(v);
  bool ret_val = x == size*(size-1)/2;
  double current_val = 0;
  for (auto val : v)
    ret_val &= current_val++ == val;
  
  for (auto& val : v)
    val = size;
  ret_val &= sum(v) == size*size;
  
  auto iteranfang = v.begin();
  auto iterende = v.end();
  
  iterende--;
  --iterende;
  
  auto iterende2 = v.end()-2;
  auto iterende3 = std::end(v)-3;
  
  ret_val &= iteranfang.v == iterende.v;
  ret_val &= iterende.v == iterende2.v;
  ret_val &= iterende.v == iterende3.v;
  
  std::cout << v << std::endl;
  
  return ret_val;
}

void testkmeansw() {
  cv::Mat_<cv::Vec3f> points(8,1);
  points << cv::Vec3f(0,0,0), cv::Vec3f(0,0,1), cv::Vec3f(0,1,1), cv::Vec3f(0,1,0),
            cv::Vec3f(1,1,0), cv::Vec3f(1,0,0), cv::Vec3f(1,0,1), cv::Vec3f(1,1,1);
  const int k = 4;
  cv::Mat_<int> labels(points.rows, points.cols);
  for (int i = 0; i < points.rows; i++)
    labels << i%k;
  cv::TermCriteria termcrit(cv::TermCriteria::EPS, 1000, 0.01);
  cv::Mat centers;
  std::vector<double> weights(points.rows);
  for (unsigned int i = 0; i < weights.size(); i++)
    weights.at(i) = i+1;
  weights.back() = 10000;

  cv::kmeansw(points, k, labels, termcrit, 1, cv::KMEANS_RANDOM_CENTERS, centers, weights);

  std::cout << "got " << centers.rows << " centers" << std::endl;
  for (int i = 0; i < centers.rows; i++)
    std::cout << centers.at<cv::Vec3f>(i) << std::endl;
}

void testkmeansw2() {
  const unsigned int cols = 500, rows = 500;
  const int k = 30;
  cv::TermCriteria termcrit(cv::TermCriteria::EPS, 1000, 0.01);
  cv::Mat_<cv::Vec2f> centers;
  cv::Mat_<cv::Vec2f> points(rows * cols, 1);
  cv::Mat_<int> labels;
  cv::Mat_<double> weights_image(rows, cols);
  std::vector<double> weights(points.rows);
  cv::Vec2f light(100, 100);
  cv::Vec2f size(rows, cols);
  const double max_dist = cv::norm(size - light);
  for (unsigned int row = 0; row < rows; row++)
    for (unsigned int col = 0; col < cols; col++) {
      cv::Vec2f tmp(row, col);
      points(row*cols + col) = tmp;
      double weight = max_dist - cv::norm(tmp - light);
      weights_image(row, col) = weight;
//      weights.at(row*cols + col) = std::pow(weight, 10);
      weights.at(row*cols + col) = std::pow(20, 2*weight/max_dist);
//      weights.at(row*cols + col) = 1;
    }
  weights_image /= max_dist;

//  cv::kmeans(points, k, labels, termcrit, 1, cv::KMEANS_RANDOM_CENTERS, centers);
  cv::kmeansw(points, k, labels, termcrit, 1, cv::KMEANS_RANDOM_CENTERS, centers, weights);

  std::cout << "got " << centers.rows << " centers" << std::endl;
  for (int i = 0; i < centers.rows; i++) {
    std::cout << centers(i) << std::endl;
    cv::Vec2f pos(centers(i));
    mark(weights_image, pos);
  }

  cv::imshow("weights_image", weights_image);
  cv::waitKey(0);
}

void testkmeans() {
  cv::Mat_<cv::Vec2f> samples(8,1);
  samples << cv::Vec2f(2,2), cv::Vec2f(2,2.1), cv::Vec2f(1.9,2), cv::Vec2f(1.9,1.9), cv::Vec2f(4,4), cv::Vec2f(4.1,4), cv::Vec2f(4,4.1), cv::Vec2f(10,10);
  cv::Mat_<int> labels;
  cv::Mat_<cv::Vec2f> centers;

  // double kmeans(const Mat& samples, int clusterCount, Mat& labels, cv::TermCriteria termcrit, int attempts, int flags, cv::Mat* centers
  double compactness = cv::kmeans(samples, 3, labels, cv::TermCriteria(), 2, cv::KMEANS_PP_CENTERS, centers);

  std::cout << "labels:" << std::endl;
  for(int i = 0; i < labels.rows; ++i) {
    std::cout << labels.at<int>(0, i) << ' ';
  }
  std::cout << std::endl;

  std::cout << "\ncenters:" << std::endl;
  for(int i = 0; i < centers.rows; ++i) {
    std::cout << centers(i) << std::endl;
  }

  std::cout << "\ncompactness: " << compactness << std::endl;
}

void testkmeansall() {
  testkmeans();
//  testkmeansw();
  testkmeansw2();

  std::exit(0);
}

bool test_all() {
  bool ret_val = true;
  ret_val &= test_gsl_vector_iterator();
  
  //  testkmeansall();
  return ret_val;
}