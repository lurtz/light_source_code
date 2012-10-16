#include "utils.h"

std::vector<std::string> &split(const std::string &s, char delim, std::vector<std::string> &elems) {
  std::stringstream ss(s);
  std::string item;
  while(std::getline(ss, item, delim)) {
    elems.push_back(item);
  }
  return elems;
}

std::vector<std::string> split(const std::string &s, char delim) {
  std::vector<std::string> elems;
  return split(s, delim, elems);
}

halton_sequence::halton_sequence(const unsigned int base, const unsigned int number) : m_base(base), m_number(number) {
}

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