#include <gpu_malloc.h>
#include <queue>
#include <mutex>
#include <condition_variable>
#define gpu_malloc_info

// Performance improvement for compression using reusable buffer.
void * reusable_buffer_space;
void * initial_buffer_space; 
int reusable_buffer_size = 0;
int max_buf_size = 0;
void * decompress_reusable_buffer_space;
int decompress_reusable_buffer_size = 0;
bool reusable_buffer_allocated = false; 
std::queue<reusable_gpu_space*> * reusable_space_queue;
bool reusable_space_allocated = false; 
size_t reusable_space_size = 0;
std::mutex q_lock;
std::condition_variable tensor_available; 

blasx_gpu_singleton* blasx_gpu_singleton::instance = NULL;

// Initialize a slab on a GPU given its id.
blasx_gpu_malloc_t *blasx_gpu_malloc_init(int GPU_id)
{
    cudaError_t cuda_err = cudaSetDevice(GPU_id);
    assert(cuda_err == cudaSuccess);
    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);
#ifdef gpu_malloc_info
    fprintf(stderr, "--------init@%d--------\n", GPU_id);
#endif
    if (free_mem > BLASX_GPU_INIT_MEM) {
#ifdef gpu_malloc_info
        printf("initilizing gpu mem");
        printf("free mem:%lu, total_mem:%lu\n", free_mem/1000000, total_mem/1000000);
#endif
    }else{
        fprintf(stderr,"the entire backend memory is less than %luMB FreeMEM:%lu M\n", (size_t)BLASX_GPU_INIT_MEM/1000000, (size_t)free_mem/1000000);
        return NULL;
    }
//    size_t BLASX_GPU_MEM_SIZE = free_mem*0.6;       // TODO : give a fix size?
//    size_t BLASX_GPU_MEM_SIZE = 1024L*1024L*10000L;
#ifdef gpu_malloc_info
    fprintf(stderr,"blasx_gpu_malloc is using %lu MB\n", (size_t) BLASX_GPU_MEM_SIZE/1000000);
#endif
    // initalize the gpu malloc block.
    blasx_gpu_malloc_t *gdata = (blasx_gpu_malloc_t*)malloc( sizeof(blasx_gpu_malloc_t) );
    void *ptr = NULL;
    blasx_gpu_segment_t *s;
    cudaError_t rc;
    int i;
    gdata->base               = NULL;
    gdata->allocated_segments = NULL;
    gdata->free_segments      = NULL;
    gdata->max_segment        = BLASX_GPU_MEM_MAX_SEGMENT + 2;
    gdata->total_size         = BLASX_GPU_MEM_SIZE;
    //cudamalloc
    rc = (cudaError_t) cudaMalloc ( &ptr, BLASX_GPU_MEM_SIZE );

//    fprintf(stderr,"blasx pool start ptr %p\n", ptr);

    gdata->base = ptr;
    if( (cudaSuccess != rc) || (NULL == gdata->base) ) {
        fprintf(stderr, "unable to allocate backend memory\n");
        free(gdata);
        exit(1);
    }
    // Allocate gpu segments. 
    for(i = 0 ; i < BLASX_GPU_MEM_MAX_SEGMENT; i++) {
        s = (blasx_gpu_segment_t*)malloc(sizeof(blasx_gpu_segment_t));
        s->next = gdata->free_segments;
        s->prev = NULL;
        if (gdata->free_segments != NULL) {
            gdata->free_segments->prev = s;
        }
        gdata->free_segments = s;
    }

    /* First and last segments are persistent. Simplifies the algorithm */
    gdata->allocated_segments = (blasx_gpu_segment_t*)malloc(sizeof(blasx_gpu_segment_t));
    gdata->allocated_segments->addr        = ptr;
    gdata->allocated_segments->mem_size    = 0;
    gdata->allocated_segments->mem_free    = BLASX_GPU_MEM_SIZE;
    gdata->free_size                       = BLASX_GPU_MEM_SIZE;

    gdata->allocated_segments->next = (blasx_gpu_segment_t*)malloc(sizeof(blasx_gpu_segment_t));
    gdata->allocated_segments->next->addr        = NULL;
    gdata->allocated_segments->next->mem_size    = 0;
    gdata->allocated_segments->next->mem_free    = 0;
    gdata->allocated_segments->next->next        = NULL;
#ifdef gpu_malloc_info
    printf("trying to allocate start@%p\n",gdata->base);
    printf("--------------------\n");
#endif
    return gdata;
}

void blasx_gpu_malloc_fini(blasx_gpu_malloc_t* gdata, int GPU_id)
{
    cudaSetDevice(GPU_id);
#ifdef gpu_malloc_info
    printf("--------dest@%d--------\n", GPU_id);
    printf("Cublas finalization is called.\n");
#endif
    blasx_gpu_segment_t *s;
    cudaError_t rc;
    while( NULL != gdata->allocated_segments ) {
        s = gdata->allocated_segments->next;
        free(gdata->allocated_segments);
        gdata->allocated_segments = s;
    }

    while( NULL != gdata->free_segments ) {
        s = gdata->free_segments->next;
        free(gdata->free_segments);
        gdata->free_segments = s;
    }
    rc = (cudaError_t)cudaFree(gdata->base);
#ifdef gpu_malloc_info
    printf("trying to free %p and cudaError is %d\n",gdata->base, rc);
#endif
    if( cudaSuccess != rc ) {
        fprintf(stderr, "Failed to free the GPU backend memory.\n");
    }
    gdata->max_segment = 0;
    gdata->total_size = 0;
    gdata->base = NULL;
    gdata->free_size = 0;
    free(gdata);
#ifdef gpu_malloc_info
    printf("--------------------\n");
#endif
}
// Allocate to an existing data. Return the pointer to the memory.
void *blasx_gpu_malloc(blasx_gpu_malloc_t *gdata, size_t nbytes)
{
    //cuDNN needs an alignment of 16
    if(nbytes % 16 != 0) {
        nbytes = (nbytes / 16 + 1) * 16;
    }

    blasx_gpu_segment_t *s, *n;
    for(s = gdata->allocated_segments; s->next != NULL; s = s->next) {
        if ( s->mem_free >= nbytes ) {
            assert(nbytes >= 0);
            n = gdata->free_segments;
            gdata->free_segments = gdata->free_segments->next;
            n->addr = s->addr + s->mem_size;
            n->mem_size = nbytes;
            n->mem_free = s->mem_free - n->mem_size;
            n->next = s->next;
            n->prev = s;
            if (s->next != NULL) {
                s->next->prev = n;
            }
            s->mem_free = 0;
            s->next = n;

//            printf(" blasx malloc size %zu, ptr %p\n", nbytes, n->addr);

            gdata->free_size -= nbytes;

            return (void*)(n->addr);
        }
    }
    return NULL;
}

void blasx_gpu_free(blasx_gpu_malloc_t *gdata, void *addr)
{
    if (gdata == NULL || gdata->allocated_segments == NULL) {
        return;
    }
    blasx_gpu_segment_t *s, *p;
    p   = gdata->allocated_segments;

    for(s = gdata->allocated_segments->next; s->next != NULL; s = s->next) {
        if ( s->addr == addr ) {
            p->next = s->next;
            if (s->next != NULL) {
                s->next->prev = p;
            }

            gdata->free_size += s->mem_size + s->mem_free;

            p->mem_free += s->mem_size + s->mem_free;
            s->next = gdata->free_segments;
            s->prev = NULL;
            gdata->free_segments->prev = s;
            gdata->free_segments = s;
            return;
        }
        p = s;
    }
    //fprintf(stderr,"address to free not allocated\n");
}

// For all the decompressed spaces.
void update_reusable_buffer_size(int tensor_size) {
	reusable_buffer_size += tensor_size;
	return;
}

void max_buffer_size(int buf_size) {
	printf("Asking for buf size %d\n", buf_size);
	if(max_buf_size < buf_size) {
		max_buf_size = buf_size;
	}
}

void delete_compressed_tensor(int delete_size) {
	reusable_buffer_space -= delete_size;
	if(reusable_buffer_space < initial_buffer_space) 
		printf("Wrong memory being accessed\n");
}

void * acquire_reusable_buffer(int buf_size) {
	/*if(reusable_buffer_size == 0) {
		cudaMalloc(&reusable_buffer_space, buf_size);
		reusable_buffer_size = buf_size; 
		return reusable_buffer_space;
	} else {
		if(reusable_buffer_size >= buf_size) {
			return reusable_buffer_space; 
		} else {
			cudaFree(reusable_buffer_space);
			cudaMalloc(&reusable_buffer_space, buf_size);
                	reusable_buffer_size = buf_size;
                	return reusable_buffer_space; 

	} */
	if(!reusable_buffer_allocated) {
		printf("Acquiring reusable buf space compressed tensors size %d max buf size %d\n", reusable_buffer_size, max_buf_size);
		cudaMalloc(&reusable_buffer_space, reusable_buffer_size + max_buf_size);
		reusable_buffer_allocated = true;
		printf("Initial buffer value is %d", reusable_buffer_space);
		initial_buffer_space = reusable_buffer_space; 
		return reusable_buffer_space;
	}

	return reusable_buffer_space;

}

// Keep compressed tensor inside the buffer space.
void update_reusable_pointer(int zfp_size) {
	reusable_buffer_space += zfp_size;
}

void * acquire_decompress_reusable_buffer(int compressed_size) {
	/*if(decompress_reusable_buffer_size == 0) {
		cudaMalloc(&decompress_reusable_buffer_space, buf_size);
		decompress_reusable_buffer_size = buf_size; 
		return reusable_buffer_space;
	} else {
		if(decompress_reusable_buffer_size >= buf_size) {
			return decompress_reusable_buffer_space; 
		} else {
			cudaFree(decompress_reusable_buffer_space);
			cudaMalloc(&decompress_reusable_buffer_space, buf_size);
                	decompress_reusable_buffer_size = buf_size;
                	return decompress_reusable_buffer_space; 
		}
	}*/
	return reusable_buffer_space + compressed_size; 
}

void register_reusable_space(size_t size) {
	if(size > reusable_space_size) 
		reusable_space_size = size; 
}

void register_test_tensors() {
	std::lock_guard<std::mutex> lock(q_lock);
	for(int i = 0; i < 200; i++) {
		auto x = new reusable_gpu_space();
		cudaMalloc(&(x->gpu_ptr), reusable_space_size);
		x->tensor_counter = 0; 
		reusable_space_queue->push(x);
	}
}

void * acquire_reusable_gpu_space() {
	std::unique_lock<std::mutex> lock(q_lock);

	if(!reusable_space_allocated) {
		// Initialize reusable spaces. 
		reusable_space_queue = new std::queue<reusable_gpu_space *>();
		for(int i = 0; i < 5; i++) {
			auto x = new reusable_gpu_space();	
			cudaMalloc(&(x->gpu_ptr), reusable_space_size);	
			x->tensor_counter = 0; 
			reusable_space_queue->push(x);	
		
		}
		printf("Allocated reusable spaces of size %zu and queue size is %d\n", reusable_space_size, reusable_space_queue->size());
		reusable_space_allocated = true;
	}
	// printf("Trying to acquire tensor, queue size is %d\n",  reusable_space_queue->size());
	while(reusable_space_queue->size() == 0) {
		tensor_available.wait(lock);	
	}
	
	auto x = reusable_space_queue->front(); reusable_space_queue->pop();
	return x;
}

void free_test_tensors() {
	std::lock_guard<std::mutex> lock(q_lock);
	for(int i = 0; i < 200; i++) {
		auto x = reusable_space_queue->front(); reusable_space_queue->pop();
		cudaFree(x->gpu_ptr);
	}
}

void free_reusable_gpu_space(reusable_gpu_space * reusable_space) {
	std::lock_guard<std::mutex> lock(q_lock);
	// printf("Freeing memory current queue size is %d\n",  reusable_space_queue->size());
	reusable_space->tensor_counter = 0;
	reusable_space_queue->push(reusable_space);
	tensor_available.notify_all();
}



/*----leave for future testing purpose----*/
//gpu_malloc testing
//gcc gpu_malloc.c -o out -I./ -I/usr/local/cuda/include -L/usr/local/cuda/lib -lcudart
//int main()
//{
//    int i;
//    blasx_gpu_malloc_t *mymem = blasx_gpu_malloc_init(0);
//
//    float *A = (float *)blasx_gpu_malloc(mymem, sizeof(float)*100);
//    float *B = (float *)blasx_gpu_malloc(mymem, sizeof(float)*100);
//    float *C = (float *)blasx_gpu_malloc(mymem, sizeof(float)*100);
//    blasx_gpu_free(mymem, (void *)A);
//    blasx_gpu_free(mymem, (void *)B);
//    blasx_gpu_free(mymem, (void *)C);
//    float *A_prior;
//    float *B_prior;
//    float *C_prior;
//
//    for (i = 0; i < 20; i++) {
//        int SIZE = sizeof(float)*1024*1024;
//        printf("==>i:%d malloc SIZE:%d\n", i, SIZE);
//        A = (float *)blasx_gpu_malloc(mymem, sizeof(float)*SIZE);
//        B = (float *)blasx_gpu_malloc(mymem, sizeof(float)*SIZE);
//        C = (float *)blasx_gpu_malloc(mymem, sizeof(float)*SIZE);
//        if (A == NULL || B == NULL || C == NULL) {
//            blasx_gpu_free(mymem, (void*) A_prior);
//            blasx_gpu_free(mymem, (void*) B_prior);
//            blasx_gpu_free(mymem, (void*) C_prior);
//            printf("not enought mem: free A B C\n");
//        } else {
//            printf("A:%p B:%p C:%p\n", A, B, C);
//            printf("\n");
//        }
//        A_prior = A;
//        B_prior = B;
//        C_prior = C;
//    }
//
//    blasx_gpu_malloc_fini(mymem, 0);
//    return 0;
//}
