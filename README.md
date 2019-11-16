# cudaprimes
Cuda program to find all the primes upto n  

It supports N in long long int range (2^64-1)

It takes 2 arguments: N as number until which to find primes, R as expected number of primes (should be greater than or equal to actual number)

We use R to preallocate array to store primes.

It is just a prototyped attempt to implement sieve of eratosthenes using cuda. It uses chunking to allocate memeory in 3Gb 
chunks on GPU at a time to find primes in big n where entire array cannot be allocated on GPU at a time.

Stats: (i7-8750h, 16 GB ram, GTX1060 Max-Q 6GB, 1TB WD Black SN750 M.2 SSD (for paging performance on host ram in case of 100 bil test))
1) n = 1000000000   (1   bil) r = 51000000   (51  mil) Time: 2.5  sec (1.2   sec GPU time) 
2) n = 10000000000  (10  bil) r = 460000000  (460 mil) Time: 30.5 sec (24    sec GPU time)
3) n = 100000000000 (100 bil) r = 4200000000 (4.2 bil) Time: 585  sec (~380  sec GPU time) 

