#include "lights.h"

halton_sequence::halton_sequence(const unsigned int base, const unsigned int number) : m_base(base), m_number(number) {
}

/*
 FUNCTION (index, base)
   BEGIN
       result = 0;
       f = 1 / base;
       i = index;
       WHILE (i > 0) 
       BEGIN
           result = result + f * (i % base);
           i = FLOOR(i / base);
           f = f / base;
       END
       RETURN result;
   END
 */

float halton_sequence::operator()() {
  // http://orion.math.iastate.edu/reu/2001/voronoi/halton_sequence.html
  unsigned int i = m_number;
  float number(0.0f);
  float divid(1.0f/static_cast<float>(m_base));
  while ( i != 0 ) {
    unsigned int digit = i % m_base;
    number += digit*divid;
    i = (i-digit)/2;
    divid /= m_base;
  }
  m_number++;
  return number;
}

void halton_sequence::discard(unsigned long long z) {
  m_number += z;
}

void halton_sequence::seed(const unsigned int i) {
  m_number = static_cast<float>(i);
}

float halton_sequence::min() {
  return 0.0f;
}

float halton_sequence::max() {
  return 1.0f;
}
