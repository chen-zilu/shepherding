import sympy as sp

# 定义符号变量（全部为实数）
xh, yh, x, y = sp.symbols('xh yh x y', real=True)

# 参数
xi = 6
gamma = 10
delta = 5

# 构造 integrand（对应于 MATLAB g1x_smp）
g1x_smp = (1 + gamma * (sp.sqrt(x**2 + y**2) - sp.sqrt(xh**2 + yh**2))) * \
          (x * (1 + delta/sp.sqrt(x**2 + y**2)) - xh)

# 执行双重积分：先对 x，再对 y
G1x_smp = sp.integrate(
    sp.integrate(g1x_smp, (x, xh - xi, xh + xi)),
    (y, yh - xi, yh + xi)
)

print(G1x_smp)