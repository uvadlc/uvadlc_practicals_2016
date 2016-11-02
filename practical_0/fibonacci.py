import sys
import numpy as np

def fibonacci(n):
	fibonacci = np.zeros(10, dtype=np.int32)
	fibonacci_pow = np.zeros(10, dtype=np.int32)
	fibonacci[0] = 0
	fibonacci[1] = 1
	for i in np.arange(2, 10):
		fibonacci[i] = fibonacci[i-1] + fibonacci[i-2]
		fibonacci[i] = int(fibonacci[i])

	print(fibonacci)

	for i in np.arange(10):
		fibonacci_pow[i] = np.power(int(fibonacci[i]), int(n))


	print(fibonacci_pow)
	print(np.vstack((fibonacci, fibonacci_pow)))
	np.savetxt('myfibonaccis.txt', np.hstack((fibonacci, fibonacci_pow)), fmt='%u')

def main(n):
	fibonacci(n)

if __name__ == '__main__':
	INPUT = sys.argv[1]
	print(INPUT)
	main(INPUT)
