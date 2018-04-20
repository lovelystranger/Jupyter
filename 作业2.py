from functools import reduce
def prod(L):
	def mul(x,y):
		return x*y
	return reduce(mul,L)

l1 = [11,2,5,3]
a = prod(l1)
print(a)