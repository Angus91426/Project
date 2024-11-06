__author__ = "Angus Liao 2023/4/27"

from gurobipy import *
from gurobipy import quicksum
from gurobipy import Model
from gurobipy import GRB

import pandas as pd
import numpy as np
import random as rd

#讀入用戶數據資料

User_data = pd.read_csv('data\整點資料CSV.csv')
index_length = len(User_data)

Price, Load, Theta = [], [], []
for index in range(index_length):
    Price.append(User_data['Price'][index])
    Load.append(User_data['Load(kW)'][index])
    Theta.append(User_data['Theta'][index])
    
Total_load = 0
for index in range(index_length):
    Total_load += Load[index]
    
#宣告係數

C_holding = 0.5         #儲能內部持有成本

C_power_generation = 5  #光伏發電成本

C_charge = 10.6         #儲能充電成本

beta = 0.90             #放電效率

CO2_pv = 35             #光伏發電碳排放量

CO2_purchase = 509      #購電碳排放量

CO2_bat = 50            #儲能充放電碳排放量

def Lower_optimization_model(Subsidy):  #下層最佳化求解
    S_pv = Subsidy[0]
    S_bat = Subsidy[1]
    S_purchase = Subsidy[2]
    
    #儲存下層求解結果
    Total, Charge_result, Discharge_result, Purchase_result, Inventory_result, PVoutput_result = [], [], [], [], [], []
    
    #建立模型
    opt_mod = Model(name = 'Electric_optimization')
    
    #加入決策變數
    #安裝容量決策變數
    E_pv = opt_mod.addVar(name = "E_pv", vtype = GRB.CONTINUOUS, lb = 0)                       #光伏容量
    E_bat = opt_mod.addVar(name = "E_bat", vtype= GRB.CONTINUOUS, lb = 0)                      #儲能容量

    #年度總量決策變數
    Total_PVoutput = opt_mod.addVar(name = "Total_PVoutput", vtype= GRB.CONTINUOUS, lb = 0)    #太陽能總發電量
    Total_Purchase = opt_mod.addVar(name = "Total_Purchase", vtype= GRB.CONTINUOUS, lb = 0)    #總購電量
    Total_Charge = opt_mod.addVar(name = "Total_Charge", vtype= GRB.CONTINUOUS, lb = 0)        #總充電量
    Total_Discharge = opt_mod.addVar(name = "Total_Discharge", vtype= GRB.CONTINUOUS, lb = 0)  #總放電量
    Total_Inventory = opt_mod.addVar(name = "Total_Inventory", vtype= GRB.CONTINUOUS, lb = 0)  #總放電量

    #時段性決策變數
    Charge, Discharge, Purchase, Inventory, PVoutput = [], [], [], [], []
    for index in range(index_length):
        Charge.append(opt_mod.addVar(name = 'Charge[' + str(index) +']', vtype = GRB.CONTINUOUS, lb = 0))          #時段儲能充電量
        Discharge.append(opt_mod.addVar(name = 'Discharge[' + str(index) +']', vtype = GRB.CONTINUOUS, lb = 0))    #時段儲能放電量
        Purchase.append(opt_mod.addVar(name = 'Purchase[' + str(index) +']', vtype = GRB.CONTINUOUS, lb = 0))      #時段用戶購電量
        Inventory.append(opt_mod.addVar(name = 'Inventory[' + str(index) +']', vtype = GRB.CONTINUOUS, lb = 0))    #時段期末存貨量
        PVoutput.append(opt_mod.addVar(name = 'PVoutput[' + str(index) +']', vtype = GRB.CONTINUOUS, lb = 0))      #時段光伏輸出量
        
    #加入限制式
    #1.計算時段光伏輸出
    opt_mod.addConstrs((PVoutput[index] == Theta[index] * E_pv for index in range(index_length)))
        
    #2.計算各年度總量
    opt_mod.addConstr(Total_PVoutput == quicksum(PVoutput[index] for index in range(index_length)))
    opt_mod.addConstr(Total_Purchase == quicksum(Purchase[index] for index in range(index_length)))
    opt_mod.addConstr(Total_Charge == quicksum(Charge[index] for index in range(index_length)))
    opt_mod.addConstr(Total_Discharge == quicksum(Discharge[index] for index in range(index_length)))
    opt_mod.addConstr(Total_Inventory == quicksum(Inventory[index] for index in range(index_length)))

    #3.光伏年度輸出須滿足20%年度用戶負載量
    opt_mod.addConstr(Total_PVoutput >= 0.2 * Total_load)

    #4.充電量不超過剩餘容量空間
    for index in range(index_length):
        if index == 0:  #儲能期初存貨為滿的
            opt_mod.addConstr(Charge[index] <= Discharge[index])
        else:           #剩餘容量空間 => 儲能總容量 - 前期期末存貨 + 當期放電量
            opt_mod.addConstr(Charge[index] <= E_bat - Inventory[index - 1] + Discharge[index])
            
    #5.放電量不超過剩餘電量，需考慮放電效率
    for index in range(index_length):
        if index == 0:  #儲能期初存貨為滿的
            opt_mod.addConstr(Discharge[index] <= beta * E_bat)
        else:           #剩餘電量 => 前期期末存貨
            opt_mod.addConstr(Discharge[index] <= beta * Inventory[index - 1])
            
    #6.期末存貨具有連續性 => 前期期末存貨 - 當期放電 + 當期充電
    for index in range(index_length):
        if index == 0:  #儲能騎出存貨為滿的
            opt_mod.addConstr(Inventory[index] == E_bat + Charge[index] - (Discharge[index]/beta))
        else:
            opt_mod.addConstr(Inventory[index] == Inventory[index - 1] + Charge[index] - (Discharge[index]/beta))
            
    #7.時段綠色能源需滿足20%用戶時段負載量
    for index in range(index_length):
        if Theta[index] == 0:
            opt_mod.addConstr(Discharge[index] >= 0.2 * Load[index])
        else:
            opt_mod.addConstr((Discharge[index] + PVoutput[index]) >= 0.2 * Load[index])

    #8.全時段供給必滿足需求
    opt_mod.addConstrs(PVoutput[index] + Discharge[index] + Purchase[index] - Charge[index] - Load[index] >= 0 for index in range(index_length))

    #9.期末存貨量不超過安裝容量
    opt_mod.addGenConstrMax(E_bat, [Inventory[index] for index in range(index_length)])

    #10.年度電力浪費不超過100000
    opt_mod.addConstr(Total_PVoutput + Total_Discharge + Total_Purchase - Total_Charge - Total_load <= 100000)

    #11.碳排放量不超過7000000000
    opt_mod.addConstr((CO2_pv * Total_PVoutput) + (CO2_purchase * Total_Purchase) + (CO2_bat * Total_Charge) + (CO2_bat * Total_Discharge) <= 7000000000)

    #12.綠色能源佔總供給比例40%
    opt_mod.addConstr(0.4 * (Total_PVoutput + Total_Discharge + Total_Purchase) <= (Total_PVoutput + Total_Discharge))

    #13.總補助金額不超過150,000,000
    #opt_mod.addConstr(((S_pv * Total_PVoutput) + (S_bat * E_bat) + (S_purchase * Total_Purchase)) <= 150000000)
    
    #定義目標函式
    obj_function = sum((Charge[index]*C_charge) + (PVoutput[index]*C_power_generation) + (Inventory[index]*C_holding) + (Purchase[index]*Price[index]) for index in range(index_length)) - ((S_pv * Total_PVoutput) + (S_bat * E_bat) + (S_purchase * Total_Purchase))
    opt_mod.setObjective(obj_function, GRB.MINIMIZE)
    
    opt_mod.update()
    opt_mod.optimize()
    
    objval = opt_mod.objval
    
    for v in opt_mod.getVars():
        if "Total" in v.VarName:
            Total.append(v.x)
        if "E_pv" in v.VarName:
            E_pv_sol = v.x
        if "E_bat" in v.VarName:
            E_bat_sol = v.x
        if "Charge" in v.VarName:
            Charge_result.append([v.varName, v.x])  #充電量
        if "Discharge" in v.VarName:
            Discharge_result.append([v.varName, v.x])  #放電量
        if "Purchase" in v.VarName:
            Purchase_result.append([v.varName, v.x])  #購電量
        if "Inventory" in v.VarName:
            Inventory_result.append([v.varName, v.x])  #期末存貨量
        if "PVoutput" in v.VarName:
            PVoutput_result.append([v.varName, v.x])  #光伏輸出量
    
    return objval, E_pv_sol, E_bat_sol, Total, Charge_result, Discharge_result, Purchase_result, Inventory_result, PVoutput_result

def Upper_obj_function(upper_sol, E_bat, Total):  #計算上層目標函式值
    S_pv = upper_sol[0]         #光伏發電補助金額
    S_bat = upper_sol[1]        #儲能安裝補助金額
    S_purchase = upper_sol[2]   #購電補助金額
    
    Total_pv = Total[0]
    Total_purchase = Total[1]
    
    obj_val = (S_pv * Total_pv) + (S_bat * E_bat) + (S_purchase * Total_purchase)
    return obj_val

def Upper_constraint(upper_sol, E_bat, Total):  #檢查上層解搭配下層解是否滿足上層限制
    violated = False
    
    S_pv = upper_sol[0]
    S_bat = upper_sol[1]
    S_purchase = upper_sol[2]
    
    Total_pv = Total[0]
    Total_purchase = Total[1]
    
    obj_val = (S_pv * Total_pv) + (S_bat * E_bat) + (S_purchase * Total_purchase)
    if S_pv < 0:
        violated = True
    
    if S_bat < 0:
        violated = True
        
    if S_purchase < 0:
        violated = True
    
    if obj_val > 150000000:
        violated = True
        
    return violated

def Get_upper_solution_population(Constraint, sol_num, lower_model):  #取得隨機上層解
    upper_level_dimension = 3
    upper_sol = np.zeros([sol_num, upper_level_dimension])
    lower_obj_val, lower_E_pv, lower_E_bat, lower_Total, lower_Charge_result, lower_Discharge_result, lower_Purchase_result, lower_Inventory_result, lower_PVoutput_result = [], [], [], [], [], [], [], [], []
    E_bat_sol, Total_sol = [], []
    
    for sol_index in range(sol_num):
        print("Searching for solution ", sol_index+1, "...")
        for index in range(upper_level_dimension):
            upper_sol[sol_index][index] = rd.uniform(0, 15)
        Obj_val, E_pv, E_bat, Total, Charge_result, Discharge_result, Purchase_result, Inventory_result, PVoutput_result = lower_model(upper_sol[sol_index])
        is_violated = Constraint(upper_sol[sol_index], E_bat, Total)
        
        while is_violated is True:
            print("Constraint violated, search again for solution ", sol_index+1, "...")
            is_violated = False
            for index in range(upper_level_dimension):
                upper_sol[sol_index][index] = rd.uniform(0, 15)
            Obj_val, E_pv, E_bat, Total, Charge_result, Discharge_result, Purchase_result, Inventory_result, PVoutput_result = lower_model(upper_sol[sol_index])
            is_violated = Constraint(upper_sol[sol_index], E_bat, Total)
        print("Solution ", sol_index+1, "found!")
        print("\n")
        lower_obj_val.append(Obj_val)
        lower_E_pv.append(E_pv)
        lower_E_bat.append(E_bat)
        lower_Total.append(Total)
        lower_Charge_result.append(Charge_result)
        lower_Discharge_result.append(Discharge_result)
        lower_Purchase_result.append(Purchase_result)
        lower_Inventory_result.append(Inventory_result)
        lower_PVoutput_result.append(PVoutput_result)
        
        E_bat_sol.append(E_bat)
        Total_sol.append(Total)
    return upper_sol, E_bat_sol, Total_sol, lower_obj_val, lower_E_pv, lower_E_bat, lower_Total, lower_Charge_result, lower_Discharge_result, lower_Purchase_result, lower_Inventory_result, lower_PVoutput_result

def Update_solution(K, eta_max, subset_solutions, current_population_solution, Upper_constraint):  #更新上層解
    #Calculating 'center of mass' (vector c)
    # => sum[m(vector u) dot vector u] / M
    # M : sum of masses of vectors  => SUM_mass
    # m(vector u) : mass of vector u, which is objective value of vector u  => mass_of_vector
    SUM_mass = 0  # M in the equation (5)
    mass_of_vector = []
    mass_dot_vector = []
    
    for subset_index in range(K):
        # m(vector u) => objective value of vector u
        mass_of_vector.append(Upper_obj_function(subset_solutions[subset_index][0], subset_solutions[subset_index][1], subset_solutions[subset_index][2]))
        
        # m(vector u) dot vector u
        mass_dot_vector.append([])
        for i in range(3):
            mass_dot_vector[subset_index].append(mass_of_vector[subset_index]*subset_solutions[subset_index][0][i])
    
    # M
    SUM_mass = sum(mass_of_vector[subset_index] for subset_index in range(K))
    
    # sum[m(vector u) dot vector u]
    SUM_mass_dot_vector = []
    for i in range(3):
        SUM_mass_dot_vector.append(sum(mass_dot_vector[subset_index][i] for subset_index in range(K)))
        
    # vector c
    center_of_mass = []
    for i in range(3):
        center_of_mass.append(SUM_mass_dot_vector[i] / SUM_mass)
        
    # the worst solution vector u
    worst_index = mass_of_vector.index(min(mass_of_vector))
    worst_sol = []
    for i in range(3):
        worst_sol.append(subset_solutions[worst_index][i])
    
    new_sol = []
    for i in range(3):
        eta = rd.uniform(0, eta_max)
        new_sol.append(current_population_solution[0][i] + eta*(center_of_mass[i] - worst_sol[0][i]))
    is_violated = Upper_constraint(new_sol, current_population_solution[1], current_population_solution[2])
    
    while is_violated is True:
        print("New solution violated, searching for new solution...")
        is_violated = False
        new_sol = []
        for i in range(3):
            eta = rd.uniform(0, eta_max)
            new_sol.append(current_population_solution[0][i] + eta*(center_of_mass[i] - worst_sol[0][i]))
        is_violated = Upper_constraint(new_sol, current_population_solution[1], current_population_solution[2])
    
    return new_sol

def PIABO(K, eta_max, max_iteration, tolerance, Upper_constraint, Upper_obj_function, Get_upper_solution_population, Update_solution, Lower_optimization_model):  #演算法function
    
    gap_records = []
    objective_value_gap = 100000
    iteration_time = 0
    upper_level_dimension = 3
    population_size = K * upper_level_dimension  # N in the equation (7)
    
    get_solution_population = Get_upper_solution_population(Upper_constraint, population_size, Lower_optimization_model)  #Generate initial solution population
    solution_population = []  #Put the same index solution together in the list
    for size in range(population_size):
        solution_population.append([])
        solution_population[size].append(get_solution_population[0][size])    #Subsidy
        solution_population[size].append(get_solution_population[1][size])    #E_bat
        solution_population[size].append(get_solution_population[2][size])    #Total list
        
        solution_population[size].append(get_solution_population[3][size])      #lower_obj_val
        solution_population[size].append(get_solution_population[4][size])      #lower_E_pv
        solution_population[size].append(get_solution_population[5][size])      #lower_E_bat
        solution_population[size].append(get_solution_population[6][size])      #lower_Total
        solution_population[size].append(get_solution_population[7][size])      #lower_Charge_result
        solution_population[size].append(get_solution_population[8][size])      #lower_Discharge_result
        solution_population[size].append(get_solution_population[9][size])      #lower_Purchase_result
        solution_population[size].append(get_solution_population[10][size])     #lower_Inventory_result
        solution_population[size].append(get_solution_population[11][size])     #lower_PVoutput_result

    while (objective_value_gap >= tolerance):
        if iteration_time > max_iteration:
            print("Reach max iteration time.")
            break
        print("Iteration ", iteration_time, "...")
        
        #Searching for the best solution and objective value for the original solution population
        original_obj_values = []
        for sol_index in range(population_size):
            original_obj_values.append(Upper_obj_function(solution_population[sol_index][0], solution_population[sol_index][1], solution_population[sol_index][2]))
        best_original_value = max(original_obj_values)
        
        #Dealing with the solution population
        for sol_index in range(population_size):
            subset_solutions = rd.sample(solution_population, K)  #Generate a subset of the solution population
            
            #Get an update solution
            print("Get a new solution")
            new_sol = Update_solution(K, eta_max, subset_solutions, solution_population[sol_index], Upper_constraint)
            
            #Check if the new solution can replace the current solution
            new_obj_val = Upper_obj_function(new_sol, solution_population[sol_index][1], solution_population[sol_index][2])
            current_obj_val = Upper_obj_function(solution_population[sol_index][0], solution_population[sol_index][1], solution_population[sol_index][2])
            if current_obj_val < new_obj_val:
                print("Replace the worst solution with the new solution.")
                solution_population[sol_index][0] = new_sol
        
        #Searching for the best solution and objective value for the new solution population
        new_obj_values = []
        for sol_index in range(population_size):
            new_obj_values.append(Upper_obj_function(solution_population[sol_index][0], solution_population[sol_index][1], solution_population[sol_index][2]))
        best_new_value = max(new_obj_values)
        
        #Calculating the gap between the original objective value and the new objective value
        objective_value_gap = abs(best_new_value - best_original_value)
        print(objective_value_gap)
        print("\n")
        gap_records.append(objective_value_gap)
        
        print("Resizing population_size...")
        population_size = round(K * (upper_level_dimension - (((upper_level_dimension - 2)*iteration_time)/max_iteration)))
        iteration_time += 1
    print("The objective value has converged to a minimum error.")
    
    obj_val = []
    for sol_index in range(population_size):
        obj_val.append(Upper_obj_function(solution_population[sol_index][0], solution_population[sol_index][1], solution_population[sol_index][2]))
    best_index = obj_val.index(max(obj_val))
    best_obj_val = max(obj_val)
    
    best_sol = solution_population[best_index][0]
    best_lower_obj_val = solution_population[best_index][3]
    best_lower_E_pv = solution_population[best_index][4]
    best_lower_E_bat = solution_population[best_index][5]
    best_lower_Total = solution_population[best_index][6]
    best_lower_Charge_result = solution_population[best_index][7]
    best_lower_Discharge_result = solution_population[best_index][8]
    best_lower_Purchase_result = solution_population[best_index][9]
    best_lower_Inventory_result = solution_population[best_index][10]
    best_lower_PVoutput_result = solution_population[best_index][11]
    return best_obj_val, best_sol, best_lower_E_pv, best_lower_E_bat, best_lower_Total, best_lower_Charge_result, best_lower_Discharge_result, best_lower_Purchase_result, best_lower_Inventory_result, best_lower_PVoutput_result, gap_records

best_obj_val, best_sol, \
best_lower_E_pv, best_lower_E_bat, best_lower_Total, \
best_lower_Charge_result, best_lower_Discharge_result, best_lower_Purchase_result, best_lower_Inventory_result, best_lower_PVoutput_result, \
gap_records = PIABO(7, 2, 500, 0.0001, Upper_constraint, Upper_obj_function, Get_upper_solution_population, Update_solution, Lower_optimization_model)

gap = []
for index in range(len(gap_records)):
    gap.append(["Gap" + str(index+1), gap_records[index]])

gap_df = pd.DataFrame(gap)

gap_df.to_excel("Gap.xlsx")

del best_lower_Charge_result[0]
del best_lower_Discharge_result[0]
del best_lower_Purchase_result[0]
del best_lower_Inventory_result[0]
del best_lower_PVoutput_result[0]

lower_obj_val = sum(C_charge*best_lower_Charge_result[index][1] + C_holding*best_lower_Inventory_result[index][1] + C_power_generation*best_lower_PVoutput_result[index][1] + Price[index]*best_lower_Purchase_result[index][1] for index in range(index_length)) -  best_sol[0]*best_lower_Total[0] - best_sol[1]*best_lower_E_bat - best_sol[2]*best_lower_Total[1]

other = []
other.append(["Lower_obj_value", lower_obj_val])
other.append(["E_pv", best_lower_E_pv])
other.append(["E_bat", best_lower_E_bat])
other.append(["Total_PVoutput", best_lower_Total[0]])
other.append(["Total_Purchase", best_lower_Total[1]])
other.append(["Total_Charge", best_lower_Total[2]])
other.append(["Total_Discharge", best_lower_Total[3]])
other.append(["Total_Inventory", best_lower_Total[4]])
other.append(["Total_Load", Total_load])

other.append([])

other.append(["Upper_obj_value", best_obj_val])
other.append(["S_pv", best_sol[0]])
other.append(["S_bat", best_sol[1]])
other.append(["S_purchase", best_sol[2]])

other.append([])

other.append(["C_holding", C_holding])
other.append(["C_power_generation", C_power_generation])
other.append(["C_charge", C_charge])

other.append([])

other.append(["CO2_pv", CO2_pv])
other.append(["CO2_bat", CO2_bat])
other.append(["CO2_purchase", CO2_purchase])

other_df = pd.DataFrame(other)

Theta_result, Load_result, Price_result = [], [], []
for index in range(index_length):
    Theta_result.append(["Theta[" + str(index) + "]", Theta[index]])  #光伏發電因數
    Load_result.append(["Load[" + str(index) + "]", Load[index]])  #負載量
    Price_result.append(["Price[" + str(index) + "]", Price[index]])  #購電價

result_df = pd.concat(
    [
        pd.DataFrame(best_lower_Charge_result),
        pd.DataFrame(best_lower_Discharge_result),
        pd.DataFrame(best_lower_Purchase_result),
        pd.DataFrame(best_lower_Inventory_result),
        pd.DataFrame(best_lower_PVoutput_result),
        pd.DataFrame(Theta_result),
        pd.DataFrame(Load_result),
        pd.DataFrame(Price_result)
    ],
    axis = 1
)

result = pd.concat([other_df, result_df], axis = 1)
result.to_excel("Result.xlsx")