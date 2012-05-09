#include "ObjLoader.h"
#include <getopt.h>
#include <string>
#include <iostream>
#include <stdlib.h>

typedef struct {
  std::string mesh_filename;
  std::string image_filename;
} arguments;

const char * opt_string = "m:i:h";

const struct option long_opts[] = {
  { "mesh-name", required_argument, NULL, 'm'},
  { "image-name", required_argument, NULL, 'i'},
  { "help", no_argument, NULL, 'h'},
  { NULL, no_argument, NULL, 0}
};

void print_arguments(const arguments& args) {
  std::cout << "mesh filename: " << args.mesh_filename << "\n"
    << "image filename: " << args.image_filename
    << std::endl;
}

void print_help() {
  std::cout << "-m|--mesh-name as filename of the 3d mesh inside the image\n"
    << "-i|--image-name as filename of the image for which lighting shall be estimated"
    << "-h|--help print this message and exit"
    << std::endl;
}

arguments parse_options(const int& argc, char * const argv[]) {
  arguments args; // = {0};
  print_arguments(args);
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
  return 0;
}
