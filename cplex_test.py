from docplex.mp.model import Model

# 创建模型
model = Model(name='test_lp')

# 决策变量
x = model.continuous_var(name='x')
y = model.continuous_var(name='y')

# 目标函数
model.maximize(20 * x + 30 * y)

# 约束条件
model.add_constraint(x + 2 * y <= 40, 'c1')
model.add_constraint(3 * x + y <= 60, 'c2')

# 求解
solution = model.solve()

# 输出结果
if solution:
    print("✅ 求解成功！")
    print(f"x = {x.solution_value}")
    print(f"y = {y.solution_value}")
    print(f"最大利润 = {model.objective_value}")
else:
    print("求解失败，请检查是否正确安装了 CPLEX Solver")
