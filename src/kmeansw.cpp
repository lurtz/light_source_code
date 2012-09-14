#include "kmeansw.h"
#include <opencv2/core/operations.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>

#define CV_KMEANS_USE_INITIAL_LABELS    1

namespace cv
{

float normL2Sqr_(const float* a, const float* b, int n)
{
    int j = 0; float d = 0.f;
#if CV_SSE
    if( USE_SSE2 )
    {
        float CV_DECL_ALIGNED(16) buf[4];
        __m128 d0 = _mm_setzero_ps(), d1 = _mm_setzero_ps();

        for( ; j <= n - 8; j += 8 )
        {
            __m128 t0 = _mm_sub_ps(_mm_loadu_ps(a + j), _mm_loadu_ps(b + j));
            __m128 t1 = _mm_sub_ps(_mm_loadu_ps(a + j + 4), _mm_loadu_ps(b + j + 4));
            d0 = _mm_add_ps(d0, _mm_mul_ps(t0, t0));
            d1 = _mm_add_ps(d1, _mm_mul_ps(t1, t1));
        }
        _mm_store_ps(buf, _mm_add_ps(d0, d1));
        d = buf[0] + buf[1] + buf[2] + buf[3];
    }
    else
#endif
    {
        for( ; j <= n - 4; j += 4 )
        {
            float t0 = a[j] - b[j], t1 = a[j+1] - b[j+1], t2 = a[j+2] - b[j+2], t3 = a[j+3] - b[j+3];
            d += t0*t0 + t1*t1 + t2*t2 + t3*t3;
        }
    }

    for( ; j < n; j++ )
    {
        float t = a[j] - b[j];
        d += t*t;
    }
    return d;
}

}

namespace cv
{

static void generateRandomCenter(const vector<Vec2f>& box, float* center, RNG& rng)
{
    size_t j, dims = box.size();
    float margin = 1.f/dims;
    for( j = 0; j < dims; j++ )
        center[j] = ((float)rng*(1.f+margin*2.f)-margin)*(box[j][1] - box[j][0]) + box[j][0];
}


/*
k-means center initialization using the following algorithm:
Arthur & Vassilvitskii (2007) k-means++: The Advantages of Careful Seeding
*/
static void generateCentersPP(const Mat& _data, Mat& _out_centers,
                              int K, RNG& rng, int trials)
{
    int i, j, k, dims = _data.cols, N = _data.rows;
    const float* data = _data.ptr<float>(0);
    size_t step = _data.step/sizeof(data[0]);
    vector<int> _centers(K);
    int* centers = &_centers[0];
    vector<float> _dist(N*3);
    float* dist = &_dist[0], *tdist = dist + N, *tdist2 = tdist + N;
    double sum0 = 0;

    centers[0] = (unsigned)rng % N;

    for( i = 0; i < N; i++ )
    {
        dist[i] = normL2Sqr_(data + step*i, data + step*centers[0], dims);
        sum0 += dist[i];
    }

    for( k = 1; k < K; k++ )
    {
        double bestSum = DBL_MAX;
        int bestCenter = -1;

        for( j = 0; j < trials; j++ )
        {
            double p = (double)rng*sum0, s = 0;
            for( i = 0; i < N-1; i++ )
                if( (p -= dist[i]) <= 0 )
                    break;
            int ci = i;
            for( i = 0; i < N; i++ )
            {
                tdist2[i] = std::min(normL2Sqr_(data + step*i, data + step*ci, dims), dist[i]);
                s += tdist2[i];
            }

            if( s < bestSum )
            {
                bestSum = s;
                bestCenter = ci;
                std::swap(tdist, tdist2);
            }
        }
        centers[k] = bestCenter;
        sum0 = bestSum;
        std::swap(dist, tdist);
    }

    for( k = 0; k < K; k++ )
    {
        const float* src = data + step*centers[k];
        float* dst = _out_centers.ptr<float>(k);
        for( j = 0; j < dims; j++ )
            dst[j] = src[j];
    }
}

}

namespace cv {
double kmeansw( InputArray _data, int K,
                   InputOutputArray _bestLabels,
                   TermCriteria criteria, int attempts,
                   int flags, OutputArray _centers,
                   const vector<double>& weights)
{
    const int SPP_TRIALS = 3;
    Mat data = _data.getMat();
    bool isrow = data.rows == 1 && data.channels() > 1;
    int N = !isrow ? data.rows : data.cols;
    int dims = (!isrow ? data.cols : 1)*data.channels();
    int type = data.depth();

    attempts = std::max(attempts, 1);
    CV_Assert( data.dims <= 2 && type == CV_32F && K > 0 );
    CV_Assert( N >= K );

    _bestLabels.create(N, 1, CV_32S, -1, true);

    Mat _labels, best_labels = _bestLabels.getMat();
    if( flags & CV_KMEANS_USE_INITIAL_LABELS )
    {
        CV_Assert( (best_labels.cols == 1 || best_labels.rows == 1) &&
                  best_labels.cols*best_labels.rows == N &&
                  best_labels.type() == CV_32S &&
                  best_labels.isContinuous());
        best_labels.copyTo(_labels);
    }
    else
    {
        if( !((best_labels.cols == 1 || best_labels.rows == 1) &&
             best_labels.cols*best_labels.rows == N &&
            best_labels.type() == CV_32S &&
            best_labels.isContinuous()))
            best_labels.create(N, 1, CV_32S);
        _labels.create(best_labels.size(), best_labels.type());
    }
    int* labels = _labels.ptr<int>();

    Mat centers(K, dims, type), old_centers(K, dims, type), temp(1, dims, type);
    vector<double> sum_of_weight_per_cluster;
    vector<int> counters(K);
    vector<Vec2f> _box(dims);
    Vec2f* box = &_box[0];

    double best_compactness = DBL_MAX, compactness = 0;
    RNG& rng = theRNG();
    int a, iter, i, j, k;

    if( criteria.type & TermCriteria::EPS )
        criteria.epsilon = std::max(criteria.epsilon, 0.);
    else
        criteria.epsilon = FLT_EPSILON;
    criteria.epsilon *= criteria.epsilon;

    if( criteria.type & TermCriteria::COUNT )
        criteria.maxCount = std::min(std::max(criteria.maxCount, 2), 100);
    else
        criteria.maxCount = 100;

    if( K == 1 )
    {
        attempts = 1;
        criteria.maxCount = 2;
    }

    const float* sample = data.ptr<float>(0);
    for( j = 0; j < dims; j++ )
        box[j] = Vec2f(sample[j], sample[j]);

    for( i = 1; i < N; i++ )
    {
        sample = data.ptr<float>(i);
        for( j = 0; j < dims; j++ )
        {
            float v = sample[j];
            box[j][0] = std::min(box[j][0], v);
            box[j][1] = std::max(box[j][1], v);
        }
    }

    for( a = 0; a < attempts; a++ )
    {
        double max_center_shift = DBL_MAX;
        for( iter = 0;; )
        {
            // reset weights
            sum_of_weight_per_cluster = vector<double>(K, 0.0);
            swap(centers, old_centers);

            if( iter == 0 && (a > 0 || !(flags & KMEANS_USE_INITIAL_LABELS)) )
            {
                if( flags & KMEANS_PP_CENTERS )
                    generateCentersPP(data, centers, K, rng, SPP_TRIALS);
                else
                {
                    for( k = 0; k < K; k++ )
                        generateRandomCenter(_box, centers.ptr<float>(k), rng);
                }
            }
            else
            {
                if( iter == 0 && a == 0 && (flags & KMEANS_USE_INITIAL_LABELS) )
                {
                    for( i = 0; i < N; i++ )
                        CV_Assert( (unsigned)labels[i] < (unsigned)K );
                }

                // compute centers
                centers = Scalar(0);
                for( k = 0; k < K; k++ )
                    counters[k] = 0;

                for( i = 0; i < N; i++ )
                {
                    sample = data.ptr<float>(i);
                    k = labels[i];
                    float* center = centers.ptr<float>(k);
                    j=0;
                    #if CV_ENABLE_UNROLLED
                    for(; j <= dims - 4; j += 4 )
                    {
                        float t0 = center[j] + sample[j];
                        float t1 = center[j+1] + sample[j+1];

                        center[j] = t0;
                        center[j+1] = t1;

                        t0 = center[j+2] + sample[j+2];
                        t1 = center[j+3] + sample[j+3];

                        center[j+2] = t0;
                        center[j+3] = t1;
                    }
                    #endif
                    for( ; j < dims; j++ )
                        center[j] += sample[j]*weights.at(i);
                    counters[k]++;
                    sum_of_weight_per_cluster.at(k) += weights.at(i);
                }

                if( iter > 0 )
                    max_center_shift = 0;

                for( k = 0; k < K; k++ )
                {
                    if( counters[k] != 0 )
                        continue;

                    // if some cluster appeared to be empty then:
                    //   1. find the biggest cluster
                    //   2. find the farthest from the center point in the biggest cluster
                    //   3. exclude the farthest point from the biggest cluster and form a new 1-point cluster.
                    int max_k = 0;
                    for( int k1 = 1; k1 < K; k1++ )
                    {
                        if( counters[max_k] < counters[k1] )
                            max_k = k1;
                    }

                    double max_dist = 0;
                    int farthest_i = -1;
                    float* new_center = centers.ptr<float>(k);
                    float* old_center = centers.ptr<float>(max_k);
                    float* _old_center = temp.ptr<float>(); // normalized
                    float scale = 1.f/counters[max_k];
                    for( j = 0; j < dims; j++ )
                        _old_center[j] = old_center[j]*scale;

                    for( i = 0; i < N; i++ )
                    {
                        if( labels[i] != max_k )
                            continue;
                        sample = data.ptr<float>(i);
                        double dist = normL2Sqr_(sample, _old_center, dims);

                        if( max_dist <= dist )
                        {
                            max_dist = dist;
                            farthest_i = i;
                        }
                    }

                    counters[max_k]--;
                    counters[k]++;
                    sum_of_weight_per_cluster.at(max_k) -= weights.at(farthest_i);
                    sum_of_weight_per_cluster.at(k) += weights.at(farthest_i);
                    labels[farthest_i] = k;
                    sample = data.ptr<float>(farthest_i);

                    for( j = 0; j < dims; j++ )
                    {
                        old_center[j] -= sample[j]*weights.at(farthest_i);
                        new_center[j] += sample[j]*weights.at(farthest_i);
                    }
                }

                for( k = 0; k < K; k++ )
                {
                    float* center = centers.ptr<float>(k);
                    CV_Assert( counters[k] != 0 );

                    float scale = 1.f/sum_of_weight_per_cluster.at(k);
                    for( j = 0; j < dims; j++ )
                        center[j] *= scale;

                    if( iter > 0 )
                    {
                        double dist = 0;
                        const float* old_center = old_centers.ptr<float>(k);
                        for( j = 0; j < dims; j++ )
                        {
                            double t = center[j] - old_center[j];
                            dist += t*t;
                        }
                        max_center_shift = std::max(max_center_shift, dist);
                    }
                }
            }

            if( ++iter == MAX(criteria.maxCount, 2) || max_center_shift <= criteria.epsilon )
                break;

            // assign labels
            compactness = 0;
            for( i = 0; i < N; i++ )
            {
                sample = data.ptr<float>(i);
                int k_best = 0;
                double min_dist = DBL_MAX;

                for( k = 0; k < K; k++ )
                {
                    const float* center = centers.ptr<float>(k);
                    double dist = normL2Sqr_(sample, center, dims);

                    if( min_dist > dist )
                    {
                        min_dist = dist;
                        k_best = k;
                    }
                }

                compactness += min_dist;
                labels[i] = k_best;
            }
        } // monsterschleifen ende

        if( compactness < best_compactness )
        {
            best_compactness = compactness;
            if( _centers.needed() )
                centers.copyTo(_centers);
            _labels.copyTo(best_labels);
        }
    }

    return best_compactness;
}
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

template<typename T, typename T1>
void mark_if(cv::Mat_<T>& mat, const T1 row, const T1 col, const T val) {
  if (row >= 0 && row < mat.rows && col >= 0 && col < mat.cols)
    mat(row, col) = val;
}

template<typename T, typename T1>
void mark(cv::Mat_<T>& mat, cv::Vec<T1, 2> pos) {
  for (int i = -1; i < 2; i++)
    for (int j = -1; j < 2; j++) {
      mark_if(mat, pos[0]+i, pos[1]+j, 0.0);
    }
  mark_if(mat, pos[0], pos[1], 1.0);
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
//      weight = std::pow(weight, 10); // TODO see if points get more attracted to light
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