from collections import defaultdict,deque # Hits:1.0, Total execution time: 1.791 ms, Average time per hit: 1.791 ms 
import sys,heapq,bisect,math,itertools,string,queue,datetime # Hits:1.0, Total execution time: 2.0 ms, Average time per hit: 2.0 ms 
sys.setrecursionlimit(10**8) # Hits:1.0, Total execution time: 0.25 ms, Average time per hit: 0.25 ms 
INF = float('inf') # Hits:1.0, Total execution time: 0.417 ms, Average time per hit: 0.417 ms 
mod = 10**9+7 # Hits:1.0, Total execution time: 0.042 ms, Average time per hit: 0.042 ms 
eps = 10**-7 # Hits:1.0, Total execution time: 0.084 ms, Average time per hit: 0.084 ms 
def inpl(): return list(map(int, input().split())) # Hits:1.0, Total execution time: 0.125 ms, Average time per hit: 0.125 ms 
def inpls(): return list(input().split()) # Hits:1.0, Total execution time: 0.084 ms, Average time per hit: 0.084 ms 
A,B = inpl() # Hits:1.0, Total execution time: 39.667 ms, Average time per hit: 39.667 ms 
if (A*B)%2 == 0: # Hits:1.0, Total execution time: 0.375 ms, Average time per hit: 0.375 ms 
	print('No') # Hits:1.0, Total execution time: 6.708 ms, Average time per hit: 6.708 ms 
else:
	print('Yes') # Hits:1.0, Total execution time: 6.792 ms, Average time per hit: 6.792 ms 
