import pulp
import pandas as pd
import numpy as np
import sys
idx = [i for i in range(70)]
df = pd.DataFrame({
    "b(i)": pd.Series([36, 54, 47, 68, 32, 48, 70, 75, 80,
              64, 38, 59, 62, 40, 65, 74, 44, 46,
              82, 75, 68, 46, 42, 57, 82, 64, 33,
              73, 43, 77, 35, 82, 74, 68, 62, 36,
              42, 30, 60, 78, 46, 84, 69, 50, 56,
              45, 55, 68, 32, 31, 78, 63, 58, 35,
              79, 64, 56, 39, 40, 47, 57, 76, 58,
              80, 57, 46, 62, 38, 63, 57], index=idx)
})
params = list(map(int, sys.argv[1].split(",")))
df = pd.DataFrame({
    "b(i)": pd.Series(params, index=idx)
})

# Create variables and model
teachers = [pulp.LpVariable.dicts(f"Teacher {i + 1}", df.index, lowBound=0, cat='Binary') for i in range(5)]
mod = pulp.LpProblem("Results", pulp.LpMinimize)

# Objective function
obj_func = []

for i in range(5):
    arr = []
    for idx in df.index:
        arr.append(teachers[i][idx] * df["b(i)"][idx])
    obj_func.append(800 - sum(arr))

mod += sum(obj_func)

mod += sum([(teachers[0][idx] * df["b(i)"][idx]) for idx in df.index]) <= 800
mod += sum([(teachers[1][idx] * df["b(i)"][idx]) for idx in df.index]) <= 800
mod += sum([(teachers[2][idx] * df["b(i)"][idx]) for idx in df.index]) <= 800
mod += sum([(teachers[3][idx] * df["b(i)"][idx]) for idx in df.index]) <= 800
mod += sum([(teachers[4][idx] * df["b(i)"][idx]) for idx in df.index]) <= 800
mod += sum([(teachers[0][idx] * df["b(i)"][idx]) for idx in df.index]) >= 720
mod += sum([(teachers[1][idx] * df["b(i)"][idx]) for idx in df.index]) >= 720
mod += sum([(teachers[2][idx] * df["b(i)"][idx]) for idx in df.index]) >= 720
mod += sum([(teachers[3][idx] * df["b(i)"][idx]) for idx in df.index]) >= 720
mod += sum([(teachers[4][idx] * df["b(i)"][idx]) for idx in df.index]) >= 720

mod += sum([(teachers[0][idx]) for idx in df.index]) >= 1
mod += sum([(teachers[1][idx]) for idx in df.index]) >= 1
mod += sum([(teachers[2][idx]) for idx in df.index]) >= 1
mod += sum([(teachers[3][idx]) for idx in df.index]) >= 1
mod += sum([(teachers[4][idx]) for idx in df.index]) >= 1

for i in range(5):
    arr = []
    for idx in df.index:
        arr.append(teachers[i][idx] * df["b(i)"][idx])
    m = sum(arr)
    v = 800 - m
    mod += v - m + 800 >= 0
    mod += v + m - 800 >= 0

# Lower and upper bounds:
for idx in df.index:
    mod += sum(teachers[i][idx] for i in range(5)) == 1
print(mod)

# Solve model
mod.solve()

# Output solution
d_sol = []
for idx in df.index:
    d_sol.append([teachers[i][idx].value() for i in range(5)])

df = np.array(d_sol)

print("Parameters:")

for dd in df:
    print(*[int(g) for g in dd])