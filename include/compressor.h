#if !defined(COMPRESSOR_H)
#define COMPRESSOR_H
#include <mutex>
#include <queue>
#include <thread>
#include <chrono>
#include <stack>
#include <tensor.h>
#include <condition_variable> 
#include <chrono>
#include <unordered_set>

typedef std::chrono::steady_clock Clock;

namespace SuperNeurons {

// Maintains all logic for compression optimization.
template <class value_type>
class Compressor { 

private:
	std::mutex queue_lock, d_queue_lock;
	std::condition_variable c, d, c_empty;
	std::queue<tensor_t<value_type>* > compression_queue;
	std::queue<tensor_t<value_type>* > decompression_queue;
	std::stack<tensor_t<value_type>* > decompression_stack;
	std::unordered_set<tensor_t<value_type> *> pending_tensors;
	int compress_counter = 0, decompress_counter = 3; 
	std::thread t1, t2, t3;
        bool decompress = false; 

public:
	Compressor() {
		t1 = std::thread(&SuperNeurons::Compressor<value_type>::compress_tensor, this);	
		t2 = std::thread(&SuperNeurons::Compressor<value_type>::decompress_tensor, this);
		// t2 = std::thread(&SuperNeurons::Compressor<value_type>::decompress_tensor, this);
		// t2 = std::thread(&SuperNeurons::Compressor<value_type>::compress_tensor, this);
		// t3 = std::thread(&SuperNeurons::Compressor<value_type>::compress_tensor, this);
		/*cpu_set_t cpuset;
    		CPU_ZERO(&cpuset);
    		CPU_SET(2, &cpuset);
    		int rc = pthread_setaffinity_np(t1.native_handle(), sizeof(cpu_set_t), &cpuset);*/
	        // int rc2 = pthread_setaffinity_np(t2.native_handle(), sizeof(cpu_set_t), &cpuset);	
	        // int rc2 = pthread_setaffinity_np(t2.native_handle(), sizeof(cpu_set_t), &cpuset);	
    		//int rc2 = pthread_setaffinity_np(t2.native_handle(), sizeof(cpu_set_t), &cpuset);
    		//int rc3 = pthread_setaffinity_np(t3.native_handle(), sizeof(cpu_set_t), &cpuset);
    		// int rc4 = pthread_setaffinity_np(std::this_thread.native_handle(), sizeof(cpu_set_t), &cpuset);
	}
	
	~Compressor() {
		printf("Destroying compressor\n");
		t1.join();
		t2.join();
		//t2.join();
		//t3.join();
	}
	
	// Start the compress phase and stop the decompress phase.
	void start_compress() {
		std::lock_guard<std::mutex> lock(queue_lock);
		decompress = false;	
		c.notify_all();
	}
	
	void trigger_compress() {
		std::lock_guard<std::mutex> lock(queue_lock);
		if(!compression_queue.empty())
			compress_counter++;
		c.notify_all();
	}
	
	void trigger_decompress() {
		std::lock_guard<std::mutex> lock(queue_lock);
                if(!decompression_stack.empty())
                        decompress_counter++;
                
		c.notify_all();	
	}
	
	// Start the decompress phase and stop the compress phase.
	void start_decompress() {
		std::unique_lock<std::mutex> lock(queue_lock);
		compress_counter += compression_queue.size(); 
		c.notify_all();
		if(!compression_queue.empty()) {
			c_empty.wait(lock);
		}
		decompress = true;
		decompress_counter = 3; 
		c.notify_all();	
		lock.unlock();
	}
	
	void add_tensor_to_queue(tensor_t<value_type>* t) {
		std::lock_guard<std::mutex> lock(queue_lock);
		compression_queue.push(t);
		c.notify_all();
	}

	void compress_tensor() {
		while(true) {
			tensor_t<value_type> * t = nullptr;
			std::unique_lock<std::mutex> lock(queue_lock);
                        auto t3 = Clock::now();
			while(compression_queue.empty() || decompress || !compress_counter) {
				//printf("Sleeping compression queue size is %d and decompress is %d\n", compression_queue.size(), decompress);
				// printf("Compress counter is %d\n");
				if(compression_queue.empty())
					c_empty.notify_all();
				c.wait(lock);
			}
	
			t = compression_queue.front(); compression_queue.pop();
			compress_counter--;
			// printf("Size of compression queue %d\n", compression_queue.size());
			decompression_stack.push(t);
			lock.unlock();
                        auto t4 = Clock::now();
                        // printf("Time compression thread slept %d ms\n", std::chrono::duration_cast<std::chrono::milliseconds>(t4 - t3).count());
			if(t != nullptr) { 
                                auto t1 = Clock::now();
				// decompression_stack.push(t);
				// pending_tensors.insert(t);
				t->compress();
				auto t2 = Clock::now();
				// printf("Time taken in compression %d micros\n", std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count());
			}
		}
	
	}

	void decompress_tensor() {
		int size = decompression_stack.size();
		while(true) {
			tensor_t<value_type> * t = nullptr;
			std::unique_lock<std::mutex> lock(queue_lock);
                        auto t3 = Clock::now();
			while(decompression_stack.empty() || !decompress || !decompress_counter) {
				c.wait(lock);
			}
			t = decompression_stack.top(); decompression_stack.pop();
			decompress_counter--;
			// printf("Size of decompression stack: %d\n", decompression_stack.size());
			lock.unlock();
                        auto t4 = Clock::now();
                        // printf("Time compression thread slept %d ms\n", std::chrono::duration_cast<std::chrono::milliseconds>(t4 - t3).count());
			if(t != nullptr) { 
                                auto t1 = Clock::now();
				t->decompress();
				auto t2 = Clock::now();
				// printf("Time taken in decompression %d micros\n", std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count());
			}
		}	
	}

};

} // namespace Superneurons
#endif
