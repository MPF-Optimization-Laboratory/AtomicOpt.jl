using Printf
n = Int(1e9)
A = zeros(n)

@show Threads.nthreads()
startTime = time();
Threads.@threads for i = 1:n
    A[i] = A[i] + i
end
endTime = time();
@printf("elapsed time: %e\n", endTime - startTime)
