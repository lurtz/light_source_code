/*
* Header file for memory_handelr
*/

#ifndef MEMORY_HANDLER
#define MEMORY_HANDLER

#define DEBUG_BUFFER_BORDER 256
/*#define DEBUG_MEMORY*/
#define DEBUG_BUFFER_VALUE 69

int init_memory_handler();
void* allocate_memory( int size );

void debug_print_memory();

void free_memory( void* memory );
void true_free_memory( void* memory );
void free_all_memory();



#endif
