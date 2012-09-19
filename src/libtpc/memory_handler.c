/*****************************************************************************

  memory_handler.c (c) 2006,2011 Turku PET Centre

  This file contains simple memory handler, and it routines. 
  Memory handler allocates memory on need, and keeps list of
  memory in use, and memory used, but not currently in use.
  
  This makes memory allocation and freeing easy and fast,
  since programmers dont need to care about memory allocation.
  This is nice with complex programgs, with lot's of places
  where there might be need for allocating memory. Also, in 
  this way one don't need long argument lists for functions
  to give workspace addresses.
  
  This version uses one-way linked list to keep track of used memory,
  and when allocating memory it first checks if there are
  free memory (so once allocated, then freed) memory pieces and
  chooses the best (less amount of wasted memory) of them.
  If such (used) memory piece is not available, it allocates such
  from free memory.
  
  Note that because how this system works, it is recommandable to
  use little bigger memory than needed (or say standard size memory blocks,
  that are not fully used), so that they can be reused.
  Also, it is not recommendable to alloc and de-alloc 
  memory in tight loop, since the algorithm goes through the 
  whole list of free memory. Also, it might be a good idea
  to make clean-up once in a while to avoid memory being to fragmented.
  
  If one is sure that memory allocated to workspace WILL NOT
  be used again, then use true_free_memory.

  ****************************************************************************

  This library is free software; you can redistribute it and/or modify it under
  the terms of the GNU Lesser General Public License as published by the Free
  Software Foundation; either version 2.1 of the License, or (at your option)
  any later version.

  This library is distributed in the hope that it will be useful, but WITHOUT
  ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
  FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more
  details.

  You should have received a copy of the GNU Lesser General Public License along
  with this library; if not, write to the Free Software Foundation, Inc.,
  59 Temple Place, Suite 330, Boston, MA 02111-1307 USA

  Turku PET Centre hereby disclaims all copyright interest in the library
  written by Pauli Sundberg.

  Juhani Knuuti
  Director, Professor
  Turku PET Centre, Turku, Finland, http://www.turkupetcentre.fi/

  Modification history:
  2006-07    Pauli Sundberg (paemsu@utu.fi) / Turku PET Centre
             First created.
  2011-09-09 VO
             Tiny changes to prevent compiler warnings on 64-bit Windows.
 

*****************************************************************************/


#include <stdlib.h>
#include <stdio.h>

#include "include/memory_handler.h"

extern int DEBUG;


/** Structure for one-way list to keep track of used memory */
struct Memory_handler {
   /** Pointer to next structure */
   void* next;
   /** Pointer to memory allocated */
   void* mem;
   /** size of this allocated memory (in bytes) */
   unsigned int size;
   /** Is this memory block used or free. 1 means 'in use' and 0 mean not in use, but allocated.
    -1 means not allocated */
   char used;
   
   #ifdef DEBUG_MEMORY
      unsigned int true_size;
   #endif
};
         
/** Global handle for memory handler */         
struct Memory_handler* MEM_HANDLE = NULL;

/** For internal use only */
void _free_memory( void* memory, int true_f );


/*!
 * Initializes this whole handler thingie. MUST BE CALLED
 *
 * @return 0 - on success, 1 - on failure (out of memory )
 */
int init_memory_handler() {
   MEM_HANDLE = (struct Memory_handler *)malloc (sizeof( struct Memory_handler ) );
   
   if (MEM_HANDLE == NULL)
   {
      return 1;
   }
   
   MEM_HANDLE->next = NULL;
   MEM_HANDLE->mem = NULL;
   MEM_HANDLE->size = 0;
   MEM_HANDLE->used = -1;

   if (DEBUG > 1)
   {
      printf("(II) MemH: memory handler init ok!\n");
   }
   return 0;   
}

/*!
 * Fills memory with DEBUG_BUFFER_VALUE for
 * DEBUG_BUFFER_BORDER bytes
 *
 * @param mem pointer to memory
 */
void fill_memory(void* mem) {
   char* help = (char*) mem;
   int loop = 0;
   for ( loop = 0; loop < DEBUG_BUFFER_BORDER ; loop ++ )
   {
      help[loop] = DEBUG_BUFFER_VALUE;
   }
}

/*!
 * Checks that memory usage of mem has not borders on both
 * sides
 *
 * @param mem pointer to memory
 */
int check_memory(void* mem) {
   char* help = (char*) mem;
   int loop = 0;
   if (DEBUG > 1)
   {
       printf("(II) Checking memory for overflow @ %p \n", mem );
   }
   for( loop = 0; loop < DEBUG_BUFFER_BORDER ; loop ++ )
   {
      if ( help[loop] != DEBUG_BUFFER_VALUE)
      {
         printf("(WW) Memory usage over on %p at %d!\n", mem, loop );
      }
   }
   return 0;
}

/*!
 * Allocate memory for atleast size bytes.
 *
 * @param size size of memory in bytes
 * @return pointer to memory if ok NULL if out of memory
 */
void* allocate_memory( int size ) {
   struct Memory_handler* loop = NULL;
   struct Memory_handler* best = NULL;
   int waste;
   
   loop = MEM_HANDLE;
   waste = loop->size - size;

   while ( 1 )
   {
      /* do we have free memory? */
      if ( loop->used == 0)
      {
         /* we are wasting memory here, but why not ,) */
         if (loop->size >= size + 2*DEBUG_BUFFER_BORDER)
         {
            if (best == NULL || loop->size - size -2* DEBUG_BUFFER_BORDER< waste)
            {
               best = loop;
               waste = loop->size - size;
            }
         }
      }
      if (loop->next == NULL)
         break;

      loop = (struct Memory_handler *)loop->next;
      
   }
   /* and loop is pointer to last node */

   /* ok check did we find one */
   if (best == NULL)
   {
      if (DEBUG > 1)
      {
         printf("(II) MemH: did not found match, allocating new (%d) \n", size);
      }

      /* noup, we need to alloc memory */
      
      /* if this node is in use, create new node */
      if (loop->used != -1)
      {
         loop->next = malloc (sizeof( struct Memory_handler ) );
         if (loop->next == NULL)
            return NULL;
         loop = (struct Memory_handler *)loop->next;
      }
      
      loop->mem  = malloc( size + 2*DEBUG_BUFFER_BORDER);
      loop->size = size + 2 * DEBUG_BUFFER_BORDER;
      loop->used = 1;
      loop->next = NULL;
      
      #ifdef DEBUG_MEMORY
         fill_memory(loop->mem );
         fill_memory(loop->mem + size + DEBUG_BUFFER_BORDER);
         loop->true_size = size;
      #endif
                 
      return (void *)(DEBUG_BUFFER_BORDER + (size_t)loop->mem);
   }
   else
   {
      if (DEBUG > 1)
      {
         printf("(II) MemH: Found mathc memory from %p waste %d!\n", best, waste);
      }
      best->used = 1;

      #ifdef DEBUG_MEMORY
         fill_memory(best->mem );
         fill_memory(best->mem + size + DEBUG_BUFFER_BORDER);
         best->true_size = size;
      #endif

      return (void *)((size_t)best->mem + DEBUG_BUFFER_BORDER);
   }
}

/*!
 * Sets memory free for re-use, but does not
 * actually free any memory
 *
 * @param memory
 */
void free_memory( void* memory ) {
   _free_memory( memory, 0);
}

/*!
 * Does the really freeing memory, deallocates the memory.
 *
 * @param memory pointer to memory
 */
void true_free_memory( void* memory ) {
   _free_memory( memory, 1);
}

/*!
 * Marks memory as freed, optionally actually frees the memory
 *
 * @param memory pointer to memory
 * @param true_f 1 for actual freeing
 */
void _free_memory( void* memory, int true_f ) {
   struct Memory_handler* loop   = NULL;
   struct Memory_handler* loopm1 = NULL;
   loop = MEM_HANDLE;

   #ifdef DEBUG_MEMORY
      memory = memory - DEBUG_BUFFER_BORDER;
   #endif
   
   while(1)
   {
      if ( loop->mem == memory ) 
      {
         if (DEBUG > 1)
         {
            printf("(II) MemH: freeing memory from %p for %d, truefree: %d\n",
                   loop, loop->size, true_f  );
         }
         
         #ifdef DEBUG_MEMORY
            check_memory( loop->mem );
            check_memory( loop->mem + DEBUG_BUFFER_BORDER + loop->true_size );
         #endif
         /* do we do the actual freeing */
         if ( true_f )
         {
            free ( loop->mem );
            /* and we got to free also the memory handler
             * memory, if this is not the root node.
             */
            loop->mem = NULL;
            loop->used = -1;
            loop->size = 0;
            
            if (loopm1 != NULL)
            {
               loopm1->next = loop->next;
               free(loop);
               return;
            }
         }
         else
         {
            loop->used = 0;
            return;
         }
      }
      
      if (loop->next == NULL)
         break;

      loopm1 = loop;
      loop = (struct Memory_handler *)loop->next;
   }
   printf("(WW) Memory handler: did not find memory to be freed\n");
}

/*!
 * Free all memory, permanently. Used as clean_up() function.
 */
void free_all_memory() {
   struct Memory_handler* loop = (struct Memory_handler *)MEM_HANDLE->next;
   struct Memory_handler* next = NULL;

   if (MEM_HANDLE->used)
   {
      free( MEM_HANDLE->mem );
   }

   MEM_HANDLE->mem = NULL;
   MEM_HANDLE->used = -1 ;
   MEM_HANDLE->size = 0;
   MEM_HANDLE->next = NULL;
  
   while( loop != NULL )
   {
      next = (struct Memory_handler *)loop->next;
      
      
      free ( loop->mem );
      free ( loop ) ;
      
      loop->mem = NULL;
      loop->next = NULL;
      loop->used = -1;
      loop->size = 0;
      loop = next;
   }
}

/*!
 * Debug printing.
 * Prints usage of memory and some info.
 */
void debug_print_memory() {
   struct Memory_handler* loop = MEM_HANDLE;
   int mem_alloc = 0;
   int mem_used  = 0;
   int mem_free  = 0;
   int nodes     = 0;
   while( loop != NULL )
   {
      if (DEBUG > 1)
      {
        printf(" node: %p memory %p size %d next: %p used:%d\n",
                 loop, loop->mem, loop->size, 
                 loop->next, loop->used);
      }
      nodes++;
      mem_alloc += loop->size;
      if (loop->used == 1)
      {
         mem_used += loop->size;
      }
      else
      {
         mem_free += loop->size;
      }

      loop = (struct Memory_handler *)loop->next;
   }
   printf("Total nodes: %d memory used: %d  currently in use: %d currently free: %d\n",
           nodes, mem_alloc, mem_used, mem_free );
}
