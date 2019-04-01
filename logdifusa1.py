
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

food = ctrl.Antecedent(np.arange(0, 11, 1), 'food')
servicio = ctrl.Antecedent(np.arange(0, 11, 1), 'service')
tip = ctrl.Consequent(np.arange(5, 26, 1), 'propina')

servicio['Pobre'] = fuzz.gaussmf(servicio.universe, 0, 1.5)
servicio['Bueno'] = fuzz.gaussmf(servicio.universe, 5, 1.5)
servicio['Excelente'] = fuzz.gaussmf(servicio.universe, 10, 1.5)

food['Desagradable'] = fuzz.trapmf(food.universe, [0, 1, 3, 5])
food['Promedio'] = fuzz.trapmf(food.universe, [2, 4,  6, 8])
food['Deliciosa'] = fuzz.trapmf(food.universe, [5, 7, 9, 10])

tip['Tacaña'] = fuzz.trimf(tip.universe, [0, 5, 10])
tip['Promedio'] = fuzz.trimf(tip.universe, [5, 15, 25])
tip['Generosa'] = fuzz.trimf(tip.universe, [20, 25, 30])


food.view()

servicio.view()

tip.view()

rule1 = ctrl.Rule(servicio['Pobre'] | food['Desagradable'], tip['Tacaña'])

rule2 = ctrl.Rule(servicio['Bueno'], tip['Promedio'])
rule3 = ctrl.Rule(servicio['Excelente'] | food['Deliciosa'], tip['Generosa'])

rule1.view()

tipping_ctrl = ctrl.ControlSystem([rule1, rule2, rule3])

tipping = ctrl.ControlSystemSimulation(tipping_ctrl)

"""""
tipping.input['food'] = 1
tipping.input['service'] = 1

tipping.input['food'] = 8
tipping.input['service'] = 3
"""""
tipping.input['food'] = 10
tipping.input['service'] = 9

tipping.compute()

print(tipping.output['propina'])
tip.view(sim=tipping)

