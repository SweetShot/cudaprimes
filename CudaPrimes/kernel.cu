
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <xutility>
#include <algorithm>
#include <vector>
#include <thrust/device_ptr.h>
#include <thrust/scan.h>
#include <chrono>

#define number_type unsigned long long

const int block_size = 1024; // 2**10 threads 
const int thread_size = 32768 * 2 * 2; // 2**15 max elements per thread always keep even number
const number_type max_chunk_size = pow(2, 31) + pow(2, 30); // 2**31 items cause reduce ram use else failed allocations, always keep even number

cudaError_t find_primes_cuda(number_type n, number_type r);

void set_one(char* dev_arr, unsigned int size);
template <typename T>
void reset(T* dev_arr, size_t count);

template <typename T>
T* device(size_t count);
template <typename T>
T* host(size_t count);
void confirmCudaNoError();
void cudaWait();
template <typename T>
T* to_host(const T* dev_ptr, size_t count, T* host_ptr = nullptr);
template <typename T>
T* to_device(const T* host_ptr, size_t count, T* dev_ptr = nullptr);



//__global__ void markNonPrimeKernel(char* dev_chunk, number_type* min_primes, number_type currentValue, number_type currentValueSqr,
//	const number_type startValue, const number_type endValue, const int thread_size)
//{
//	const auto myThreadId = blockIdx.x * block_size + threadIdx.x;
//	const auto myStartValue = startValue + myThreadId * thread_size;
//	auto myEndValue = myStartValue + thread_size;
//	if (myEndValue > endValue)
//	{
//		myEndValue = endValue;
//	}
//	bool is_first_set = (min_primes != nullptr)?false:true;
//
//	number_type offset = 1;
//	// if current min first is set then we can offset by currentValue but if 
//	// the number i is odd (which we can make sure of) then we can increment by 
//	// currentValue * 2 as then we skip all even numbers in between which we dont need anyway
//	// as they will be already marked in case of 2
//	int offsetMultiplier = (currentValue == 2) ? 1 : 2; // 
//
//	for (auto i = myStartValue; i < myEndValue; i += offset)
//	{
//		if (i <= currentValue)
//		{
//			continue;
//		}
//		if (offset != 1)
//		{
//			dev_chunk[i - startValue] = 0; // cancel that number, min is already marked, offset is current number
//		}
//		else if (i % currentValue == 0) {
//			dev_chunk[i - startValue] = 0; // cancel that number
//		}
//		else // set in min primes if not null
//		{
//			if (!is_first_set && dev_chunk[i - startValue] == 1)
//			{
//				min_primes[myThreadId] = i;
//				is_first_set = true;
//			}
//			if (is_first_set) // done just now or dont need to 
//			{
//				// as we have already found first min prime,
//				// we update i to next which will be divisible and incrementation to complete faster 
//				i -= i % currentValue;
//				if (i % 2 == 0) // if even make it odd as only odd numbers can be marked off (even are done in case of 2)
//				{
//					i -= currentValue;
//				}
//
//				if (i < currentValueSqr)
//					i = currentValueSqr - currentValue * offsetMultiplier;
//
//				offset = currentValue * offsetMultiplier;
//			}
//		}
//	}
//}

__global__ void markNonPrimeKernel(char* dev_chunk, number_type currentValue, number_type currentValueSqr,
	const number_type startValue, const number_type endValue, const int thread_size)
{
	const auto myThreadId = blockIdx.x * block_size + threadIdx.x;
	const auto myStartValue = startValue + myThreadId * thread_size;
	auto myEndValue = myStartValue + thread_size;
	if (myEndValue > endValue)
	{
		myEndValue = endValue;
	}

	number_type offset = 1;
	// if current min first is set then we can offset by currentValue but if 
	// the number i is odd (which we can make sure of) then we can increment by 
	// currentValue * 2 as then we skip all even numbers in between which we dont need anyway
	// as they will be already marked in case of 2
	const int offsetMultiplier = (currentValue == 2) ? 1 : 2; // 

	auto updated_start = myStartValue;
	if (updated_start != 0) // in case of zero first statement will underflow and will lead to max value
	{
		updated_start = myStartValue - myStartValue % currentValue;
		if (updated_start % 2 == 0) // if even make it odd as only odd numbers can be marked off 
		//(even are done in case of 2, in which case subtracting 2 will still make it even)
		{
			updated_start -= currentValue;
		}
	}

	if (updated_start < currentValueSqr)
		updated_start = currentValueSqr;
	offset = currentValue * offsetMultiplier;

	for (auto i = updated_start; i < myEndValue; i += offset)
	{
		dev_chunk[i - startValue] = 0; // cancel that number, min is already marked, offset is current number
	}
}

__global__ void getNextPrime(number_type* dev_temp_min_primes, int size, number_type* d_ans)
{
	auto threadId = threadIdx.x;
	if (threadId == 0)
	{
		for (auto i = 0; i < size; i++)
		{
			auto number = dev_temp_min_primes[i];
			if (number != 0)
			{
				*d_ans = number;
				return;
			}
		}
	}
}

// only needed for first chunk id = 0
__global__ void getNextPrimeFast(char* dev_chunk, number_type currentValue, const number_type startValue, const number_type endValue, number_type* d_ans)
{
	auto threadId = threadIdx.x;
	if (threadId == 0)
	{
		for (auto i = currentValue + 1; i < endValue; i++)
		{
			auto number = dev_chunk[i];
			if (number == 1)
			{
				*d_ans = i;
				return;
			}
		}
	}
}

__global__ void countPrimes(char* dev_chunk, number_type* count_accumulation_chunk,
	const number_type startValue, const number_type endValue, const int thread_size)
{
	const auto my_thread_id = blockIdx.x * block_size + threadIdx.x;
	auto my_start_value = startValue + my_thread_id * thread_size;
	auto my_end_value = my_start_value + thread_size;
	if (my_end_value > endValue)
	{
		my_end_value = endValue;
	}
	unsigned long count = 0;
	if (my_start_value == 0)
	{
		count += 1; // add first prime 2, cause all others are odd
	}
	if (my_start_value % 2 == 0) // make odd
	{
		my_start_value += 1;
	}
	for (auto i = my_start_value; i < my_end_value; i+=2)
	{
		const auto current_status = dev_chunk[i - startValue];
		if (current_status == 1)
		{
			count += 1;
		}
	}
	count_accumulation_chunk[my_thread_id] = count;
}

__global__ void copyPrimes(char* dev_chunk, number_type* base_index_arr, number_type* primes_arr,
	const number_type startValue, const number_type endValue, const int thread_size)
{
	const auto my_thread_id = blockIdx.x * block_size + threadIdx.x;
	auto my_start_value = startValue + my_thread_id * thread_size;
	auto my_end_value = my_start_value + thread_size;
	if (my_end_value > endValue)
	{
		my_end_value = endValue;
	}
	unsigned long index = base_index_arr[my_thread_id];

	if (my_start_value == 0)
	{
		primes_arr[index] = 2; // add first prime 2, cause all others are odd
		index++;
	}
	if (my_start_value % 2 == 0) // make odd, cause prime can only be in odd place
	{
		my_start_value += 1;
	}

	for (auto i = my_start_value; i < my_end_value; i+=2)
	{
		const auto current_status = dev_chunk[i - startValue];
		if (current_status == 1)
		{
			primes_arr[index] = i;
			index += 1;
		}
	}
}

int main(int argc, char* argv[])
{
	number_type n = 1000000000;
	number_type r = 60000000;
	if (argc == 1)
	{
		fprintf(stderr, "usage: CudaPrimes.exe N R\n");
		fprintf(stderr, "N = primes until N\n");
		printf("Using default : %llu\n", n);
		fprintf(stderr, "R = reserve primes space\n");
		printf("Using default : %llu\n", r);
	}
	else if (argc == 3)
	{
		n = static_cast<number_type>(strtoull(argv[1], nullptr, 10));
		if (argc == 3)
		{
			r = static_cast<number_type>(strtoull(argv[2], nullptr, 10));
		}
	}
	auto start = std::chrono::system_clock::now();
	cudaError_t cudaStatus = find_primes_cuda(n, r);
	auto end = std::chrono::system_clock::now();
	std::chrono::duration<double> elapsed_seconds = end - start;
	std::cout << "Total time taken: " << elapsed_seconds.count() << "s\n";
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "findPrimesCuda failed!");
		return 1;
	}

	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		return 1;
	}
	//int a = 0;
	//scanf("%d", &a);

	return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t find_primes_cuda(const number_type n, const number_type r)
{
	char* dev_chunk = nullptr;
	number_type* dev_count_accumulation_chunk = nullptr;
	number_type* dev_next_prime = nullptr;
	const unsigned long long chunk_size = static_cast<unsigned int>(std::min(max_chunk_size, n));
	number_type* primes = new number_type[r];
	number_type n_primes = 0;

	// Choose which GPU to run on, change this on a multi-GPU system.
	const cudaError_t cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	// Allocate GPU buffer for temp chunk to store markings.
	dev_chunk = device<char>(chunk_size);
	const int total_chunks = (n <= max_chunk_size) ? 1 : (n / chunk_size + 1);
	for (auto chunk_id = 0; chunk_id * chunk_size < n; chunk_id++) {
		const unsigned int numbers_for_this_chunk = static_cast<unsigned int>(std::min(n - chunk_id * chunk_size,
			static_cast<number_type>(chunk_size)));

		// we have fixed the following for GTX1060 with 10 SM (so assuming max 128 blocks for 4 billion chunk, 
		// which is good enough)
		// we have fixed the threads per block and amount of work per thread, so lets calculate required number of blocks
		const int total_num_threads = numbers_for_this_chunk / thread_size + 2; // +1 cause its integer division + 1 cause need spare later
		const int blocks = total_num_threads / block_size + 1; // +1 cause its integer division


		const number_type currentMin = chunk_id * chunk_size;
		const number_type currentMax = currentMin + numbers_for_this_chunk;

		printf("Chunk Id : %d / %d, Start: %llu End: %llu\n", chunk_id + 1, total_chunks, currentMin, currentMax);
		printf("Total threads : %d, Total Blocks: %d \n", total_num_threads, blocks);

		// Set all to 1 in dev_chunk (yet to be decided state, 0 means nope)
		set_one(dev_chunk, chunk_size);
		std::cout << "Reset\n";

		if (chunk_id == 0)
		{
			cudaMemset(dev_chunk, 0, 2); // only for first chunk

			// Allocate an array to store next found prime number [0 means not used] 
			// size should be total number of threads
			dev_count_accumulation_chunk = device<number_type>(total_num_threads);
			dev_next_prime = device<number_type>(1);
		}

		number_type currentPrime = 2;
		const number_type stoppingCondition = static_cast<number_type>(sqrtl(currentMax));

		double printVal = 0.2;
		unsigned long long primesConsidered = 0;
		number_type primesIterator = 0;
		while (currentPrime <= stoppingCondition) {
			const auto percentDone = currentPrime / static_cast<double>(stoppingCondition);
			primesConsidered++;
			if (percentDone > printVal)
			{
				printf("At %lf and currentPrime %llu this time %llu\n", percentDone, currentPrime, primesConsidered);
				while (printVal < percentDone)
				{
					printVal += 0.19;
					primesConsidered = 0;
				}
			}

			// Step 1: Start kernel with currentPrime
			markNonPrimeKernel << <dim3(blocks), dim3(block_size) >> > (dev_chunk,
				currentPrime, currentPrime * currentPrime, currentMin, currentMax, thread_size);

			// Check for any errors launching the kernel
			confirmCudaNoError();
			cudaWait();

			// Step 2: Get Next prime from kernel if chunk_id == 0 else get from primes vector 
			//         and set as current prime
			if (chunk_id == 0)
			{
				//getNextPrime << <dim3(1), dim3(1) >> > (dev_temp_min_primes, total_num_threads, dev_next_prime);
				getNextPrimeFast << <dim3(1), dim3(1) >> > (dev_chunk, currentPrime, currentMin,
					currentMax, dev_next_prime);
				confirmCudaNoError();
				cudaWait();
				to_host<number_type>(dev_next_prime, 1, &currentPrime);
			}
			else
			{
				primesIterator++;
				currentPrime = primes[primesIterator];
			}
		}

		// Step 3: Count all primes in dev_chunk (as same number of threads will be used we use same dev_temp_min_primes array)

		countPrimes << <dim3(blocks), dim3(block_size) >> > (dev_chunk, dev_count_accumulation_chunk, currentMin,
			currentMax, thread_size);
		confirmCudaNoError();
		cudaWait();

		// exclusive scan on it to get primes before me for every thread
		thrust::device_ptr<number_type> dev_ptr = thrust::device_pointer_cast(dev_count_accumulation_chunk);
		thrust::exclusive_scan(dev_ptr, dev_ptr + total_num_threads, dev_ptr); // in place

		const auto n_primes_curr_temp = to_host<number_type>(dev_count_accumulation_chunk + total_num_threads - 1, 1);
		const auto n_primes_curr_chunk = *n_primes_curr_temp;
		free(n_primes_curr_temp);
		printf("Total %llu primes found in this chunk\n", n_primes_curr_chunk);

		// Step 4: Copy back all primes to prime vector

		// 4.1 Gather primes
		const auto dev_primes = device<number_type>(n_primes_curr_chunk);
		copyPrimes << <dim3(blocks), dim3(block_size) >> > (dev_chunk, dev_count_accumulation_chunk, dev_primes,
			currentMin, currentMax, thread_size);
		confirmCudaNoError();
		cudaWait();

		// 4.2 gather back on host array directly
		to_host<number_type>(dev_primes, n_primes_curr_chunk, primes+n_primes);
		cudaFree(dev_primes);
		// 4.3 update start index for next
		n_primes += n_primes_curr_chunk;

		// clean first chunk extra vars
		if (chunk_id == 0)
		{
			cudaFree(dev_next_prime);
		}
	}

	printf("Total Primes found : %llu \n", n_primes);

Error:
	cudaFree(dev_chunk);
	delete[] primes;
	return cudaStatus;
}

void set_one(char* dev_arr, unsigned int size)
{
	const auto error = cudaMemset(dev_arr, 1, size);
	if (error != cudaSuccess)
	{
		throw error;
	}
}

template <typename T>
void reset(T* dev_arr, size_t count)
{
	const auto error = cudaMemset(dev_arr, 0, count * sizeof(T));
	if (error != cudaSuccess)
	{
		throw error;
	}
}

template <typename T>
T* device(size_t count)
{
	T* dev_ptr = nullptr;
	const auto error = cudaMalloc(reinterpret_cast<void**>(&dev_ptr), count * sizeof(T));
	if (error != cudaSuccess)
	{
		auto x = cudaGetErrorString(cudaGetLastError());
		throw error;
	}
	return dev_ptr;
}

template <typename T>
T* host(size_t count)
{
	T* host_ptr = reinterpret_cast<T*>(malloc(count * sizeof(T)));
	if (host_ptr == nullptr)
	{
		throw "Could not allocate on host!";
	}
	return host_ptr;
}

template <typename T>
T* to_host(const T* dev_ptr, size_t count, T* host_ptr)
{
	if (host_ptr == nullptr)
		host_ptr = host<T>(count);
	auto error = cudaMemcpy(host_ptr, dev_ptr, count * sizeof(T), cudaMemcpyDeviceToHost);
	if (error != cudaSuccess)
	{
		throw error;
	}
	return host_ptr;
}

template <typename T>
T* to_device(const T* host_ptr, size_t count, T* dev_ptr)
{
	if (dev_ptr == nullptr)
		dev_ptr = device<T>(count);
	auto error = cudaMemcpy(dev_ptr, host_ptr, count * sizeof(T), cudaMemcpyHostToDevice);
	if (error != cudaSuccess)
	{
		throw error;
	}
	return dev_ptr;
}


void confirmCudaNoError()
{
	const auto error = cudaGetLastError();
	if (error != cudaSuccess)
	{
		throw error;
	}
}

void cudaWait()
{
	const auto error = cudaDeviceSynchronize();
	if (error != cudaSuccess)
	{
		auto x = cudaGetErrorString(cudaGetLastError());
		throw error;
	}
}