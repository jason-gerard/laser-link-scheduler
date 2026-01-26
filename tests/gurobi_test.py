import pulp as pl


solver_list = pl.listSolvers(onlyAvailable=True)
print(solver_list)

model = pl.LpProblem("Example", pl.LpMinimize)
solver = pl.GUROBI_CMD()
_var = pl.LpVariable("a")
_var2 = pl.LpVariable("a2")
model += _var + _var2 == 1
result = model.solve(solver)
