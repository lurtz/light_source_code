#include "tests.h"

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