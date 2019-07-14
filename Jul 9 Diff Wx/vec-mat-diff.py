from sympy import init_session
# init_session() 

from sympy.matrices import Matrix, eye, zeros, ones, diag, GramSchmidt
from sympy import Symbol,symarray,diff


W = Matrix(symarray('w', (2, 2)))
print("W=\n"
pprint(W,use_unicode=False)


x = Matrix(symarray('x', (2, 1)))
x

W@x

diff(W@x,W)