#ifndef LIGHTS_H_
#define LIGHTS_H_

template<typename T>
struct Light {
  typedef std::map<std::string, std::vector<T> > properties;
};

template<typename T>
typename Light<T>::properties create_light(const unsigned int number, const std::vector<T>& position, const std::vector<T>& ambient, const std::vector<T>& diffuse, const std::vector<T>& specular) {
  typename Light<T>::properties tmp;
  std::stringstream name;
  name << "lights[" << number << "]." << "position";
  tmp[name.str()] = position;
  name.str("");
  name << "lights[" << number << "]." << "ambient";
  tmp[name.str()] =  ambient;
  name.str("");
  name << "lights[" << number << "]." << "diffuse";
  tmp[name.str()] = diffuse;
  name.str("");
  name << "lights[" << number << "]." << "specular";
  tmp[name.str()] = specular;
  return tmp;
}

template<typename T, int dim>
std::vector<T> create_vector_from_array(const T (&array)[dim]) {
  std::vector<T> tmp(array, array+dim);
  return tmp;
}

template<typename T, int dim>
std::vector<typename Light<T>::properties> create_lights(const T light_props[][dim], const unsigned int count) {
  std::vector<typename Light<T>::properties> tmp;
  for (unsigned int i = 0; i < 4*count; i+=4) {
	  std::vector<T> position = create_vector_from_array(light_props[i+0]);
	  std::vector<T> ambient = create_vector_from_array(light_props[i+1]);
	  std::vector<T> diffuse = create_vector_from_array(light_props[i+2]);
	  std::vector<T> specular = create_vector_from_array(light_props[i+3]);

	  typename Light<T>::properties props = create_light(i/4, position, ambient, diffuse, specular);
	  tmp.push_back(props);
  }
  return tmp;
}

template<typename T>
void prints_lights(const std::vector<typename Light<T>::properties> &lights) {
	unsigned int size = lights.size();
	  std::cout << size << std::endl;
	  for (auto iter = lights.begin(); iter != lights.end(); iter++) {
		  typename Light<T>::properties prop = *iter;
		  for (auto iter_prop = prop.begin(); iter_prop != prop.end(); iter_prop++) {
			  std::string name = iter_prop->first;
			  std::vector<T> value = iter_prop->second;
			  std::cout
			    << "light property name: "
			    << name
			    << ", value: ";
			  for (auto iter_val = value.begin(); iter_val != value.end(); iter_val++)
			    std::cout << *iter_val << ", ";
			  std::cout << std::endl;
		  }
	  }
}

#endif /* LIGHTS_H_ */
