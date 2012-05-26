#include "ObjLoader.h"
#include <getopt.h>
#include <string>
#include <iostream>
#include <stdlib.h>
#include <cv.hpp>
#include <highgui.h>
#include <sstream>
#include <fstream>
#include <vector>

// This is my plan:
// I have to synthesize a new image using the mesh obj and the light source
// for better error accumulation I could crop everything except the mesh from the original image data
// for image synthesis I could use a ray tracer
// then for image synthesation it is better to build a kd-Tree around the mesh obj, which demands, that rotation, translation and scalation is done before kd-Tree construction

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

typedef struct arguments {
  std::string mesh_filename;
  std::string image_filename;
  float scale;
  float rotation[3];
  float translation[3];
  arguments() : mesh_filename(""), image_filename(""), scale(1), rotation{0}, translation{0} {}
} arguments;

const char * opt_string = "m:i:s:r:t:hc:";

const struct option long_opts[] = {
  { "mesh-name", required_argument, NULL, 'm'},
  { "image-name", required_argument, NULL, 'i'},
  { "scale", required_argument, NULL, 's'},
  { "rotation", required_argument, NULL, 'r'},
  { "translation", required_argument, NULL, 't'},
  { "help", no_argument, NULL, 'h'},
  { "config", required_argument, NULL, 'c'},
  { NULL, no_argument, NULL, 0}
};

template<class T, int N>
std::string print_coords(const T (&coords)[N]) {
  std::stringstream ret_val;
  ret_val << coords[0];
  for (unsigned int i = 1; i < N; i++)
    ret_val << '/' << coords[i];
  return ret_val.str();
}

void print_arguments(const arguments& args) {
  std::cout << "mesh filename: " << args.mesh_filename << '\n'
    << "image filename: " << args.image_filename << '\n'
    << "scale: " << args.scale << '\n'
    << "rotation: " << print_coords(args.rotation) << '\n'
    << "translation: " << print_coords(args.translation)
    << std::endl;
}

void print_help() {
  std::cout << "-m|--mesh-name as filename of the 3d mesh inside the image\n"
    << "-i|--image-name as filename of the image for which lighting shall be estimated\n"
    << "-h|--help print this message and exit\n"
    << "-s|--scale scale factor to be applied to the mesh. format: s\n"
    << "-r|--rotation rotation to be applied to the mesh. format: x/y/z\n"
    << "-t|--translation position of the mesh in the image. format: x/y/z\n"
    << "-c|--config configuration file\n"
    << std::endl;
}

template<class T, int N>
void read_coords(const std::string raw, T (&coords)[N], const char split_arg = '/') {
   std::vector<std::string> tokens = split(raw, split_arg);
   for (unsigned int i = 0; i < N; i++) {
     std::stringstream ss;
     ss << tokens.at(i);
     ss >> coords[i];
   }
}

void read_config_file(const std::string filename, arguments &args) {
  std::ifstream myfile(filename.c_str());
  size_t prefix_pos = filename.find_last_of("/\\");
  std::string prefix("");
  if (prefix_pos != std::string::npos) {
    prefix = filename.substr(0, prefix_pos+1);
  }
  while (myfile) {
    std::string token;
    myfile >> token;
    if (token == "mesh-name") {
      std::string tmp_filename;
      myfile >> tmp_filename;
      args.mesh_filename = prefix + tmp_filename;
    } else if (token == "image-name") {
      std::string tmp_filename;
      myfile >> tmp_filename;
      args.image_filename = prefix + tmp_filename;
    } else if (token == "scale") {
      myfile >> args.scale;
    } else if (token == "rotation") {
      std::string coords;
      myfile >> coords;
      read_coords(coords, args.rotation);
    } else if (token == "translation") {
      std::string coords;
      myfile >> coords;
      read_coords(coords, args.translation);
    }
  }
}

arguments parse_options(const int& argc, char * const argv[]) {
  arguments args;
  int opt = 0;
  int long_index = 0;
  while ((opt = getopt_long(argc, argv, opt_string, long_opts, &long_index)) != -1) {
    switch (opt) {
      case 'm':
        args.mesh_filename = std::string(optarg);
        break;
      case 'i':
        args.image_filename = std::string(optarg);
        break;
      case 's': {
        std::stringstream ss;
        ss << optarg;
        ss >> args.scale;
        break;
      }
      case 'r':
        read_coords(optarg, args.rotation);
        break;
      case 't':
        read_coords(optarg, args.translation);
        break;
      case 'c':
        read_config_file(optarg, args);
        break;
      case 'h':
      default:
        print_help();
        exit(0);
    }
  }
  print_arguments(args);
  return args;
}

int main(int argc, char * argv[]) {
  arguments args = parse_options(argc, argv);
  cv::Mat image;
  if (args.image_filename != "") {
    image = cv::imread(args.image_filename);
    cv::imshow("test", image);
    cv::waitKey(0);
  }
  if (args.mesh_filename != "") {
    ObjLoader objl;
    MeshObj * mesh = objl.loadObjFile(args.mesh_filename, args.mesh_filename);
  }

  std::cout << "OK" << std::endl;
  return 0;
}
