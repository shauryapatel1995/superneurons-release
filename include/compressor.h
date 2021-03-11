#if !defined(_COMPRESSOR_H_)
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
	std::condition_variable c, d;
	std::queue<tensor_t<value_type>* > compression_queue;
	std::queue<tensor_t<value_type>* > decompression_queue;
	std::stack<tensor_t<value_type>* > decompression_stack;
	std::unordered_set<tensor_t<value_type> *> pending_tensors;
	int counter; 
	std::thread t1, t2, t3; 

public:
	Compressor() {
                counter = 4; 
		t1 = std::thread(&SuperNeurons::Compressor<value_type>::compress_tensor, this);	
		// t2 = std::thread(&SuperNeurons::Compressor<value_type>::decompress_tensor, this);
		// t2 = std::thread(&SuperNeurons::Compressor<value_type>::decompress_tensor, this);
		// t2 = std::thread(&SuperNeurons::Compressor<value_type>::compress_tensor, this);
		// t3 = std::thread(&SuperNeurons::Compressor<value_type>::compress_tensor, this);
		cpu_set_t cpuset;
    		CPU_ZERO(&cpuset);
    		CPU_SET(1, &cpuset);
    		int rc = pthread_setaffinity_np(t1.native_handle(), sizeof(cpu_set_t), &cpuset);
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
	
	void start_decompress(tensor_t<value_type> * _t) {
		while(compression_queue.size() != 0) {
			printf("Compression queue size %d\n", compression_queue.size());
		}
		/*if(counter < 3) {
			++counter; 
			printf("Called %d counter\n", counter);
			return;
		} */
                
		std::lock_guard<std::mutex> lock(d_queue_lock);
		/*tensor_t<value_type> * t = nullptr;
		// printf("%d is the answer\n", _t == decompression_stack.top());
		if(pending_tensors.count(_t) > 0) {
			// printf("We tried to look in pending tensors\n");
			while(_t != t) {
				 if(decompression_stack.empty()) {
                                	break;
                        	}
                        	t = decompression_stack.top(); decompression_stack.pop();
                        	decompression_queue.push(t);
				pending_tensors.erase(t);
                	}
                	d.notify_one();
	
		} else {
		for(int i = 0; i < 3; i++) {
			if(decompression_stack.empty()) {
				break;
			}
			t = decompression_stack.top(); decompression_stack.pop();
			decompression_queue.push(t);
			pending_tensors.erase(t);
		} */
		d.notify_one();
                // printf("Decompression stack size %d, Decompression queue size %d\n", decompression_stack.size() , decompression_queue.size());	
		//}
	}
	
	void add_tensor_to_queue(tensor_t<value_type>* t) {
		std::lock_guard<std::mutex> lock(queue_lock);
		// printf("Adding to queue");
		compression_queue.push(t);
		// printf("Size of queue is %d", compression_queue.size());
		c.notify_one();
	}

	void compress_tensor() {
		while(true) {
			tensor_t<value_type> * t = nullptr;
			std::unique_lock<std::mutex> lock(queue_lock);
                        auto t3 = Clock::now();
			while(compression_queue.empty()) {
				c.wait(lock);
			}
			t = compression_queue.front(); compression_queue.pop();
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
		
		while(true) {
			printf("I'm awake!\n");
			tensor_t<value_type> * t = nullptr;
			std::unique_lock<std::mutex> lock(d_queue_lock);
                        auto t3 = Clock::now();
			while(decompression_stack.empty()) {
				d.wait(lock);
			}
			t = decompression_stack.top(); decompression_stack.pop();
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
