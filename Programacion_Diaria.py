from pymoo.constraints.as_penalty import ConstraintsAsPenalty
from pymoo.core.evaluator import Evaluator
from pymoo.core.mixed import MixedVariableGA
from pymoo.core.termination import Termination
from pymoo.core.variable import Real, Integer
from pymoo.operators.mutation.pm import PolynomialMutation
from pymoo.optimize import minimize
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.variable import Real, Integer, Choice, Binary
from pymoo.visualization.scatter import Scatter
from pymoo.algorithms.moo.nsga2 import NSGA2, RankAndCrowdingSurvival
from pymoo.core.mixed import MixedVariableMating, MixedVariableGA, MixedVariableSampling, MixedVariableDuplicateElimination
from pymoo.optimize import minimize
from pymoo.visualization.pcp import PCP
from pymoo.indicators.gd import GD
import numpy as np
import matplotlib.pyplot as plt
#use TkAgg matplotlib backend
import matplotlib
matplotlib.use('TkAgg')
import pandas as pd
from pymoo.termination.default import DefaultSingleObjectiveTermination
from pymoo.termination import get_termination
from pruebagrafico3 import plot_schedule

import re

#create a surgery class with the operation room, the start time and the end time
class Surgery():
    def __init__(self, operation_room, start_time, end_time, personal):
        self.operation_room = operation_room
        self.start_time = int(start_time)
        self.end_time = int(end_time)
        self.duration = self.end_time - self.start_time
        self.personal = personal

    def __str__(self):
        return f"'Room: {self.operation_room} ST: {self.start_time} ET: {self.end_time} Personal: {self.personal}'"

    def __repr__(self):
        return f"'Room: {self.operation_room} ST: {self.start_time} ET: {self.end_time} Personal: {self.personal}'"

    def __eq__(self, other):
        return self.operation_room == other.operation_room and self.start_time == other.start_time and self.end_time == other.end_time and self.personal == other.personal

    def __hash__(self):
        return hash((self.operation_room, self.start_time, self.end_time, self.personal))

#create a class for a personal, only with an id and its role
class Personal():
    def __init__(self, id, role):
        self.id = id
        self.role = role

    def __str__(self):
        return f"ID: {self.id} Role: {self.role}"

    def __repr__(self):
        return f"ID: {self.id} Role: {self.role}"

    def __eq__(self, other):
        return self.id == other.id and self.role == other.role

    def __hash__(self):
        return hash((self.id, self.role))

#create a class for each activity, with the studie that its needed, the tipe of the activity and the duration and the personal needed for each activity
class Actividad():
    def __init__(self, estudio, tipo_actividad, personal, duration):
        self.estudio = estudio
        self.tipo_actividad = tipo_actividad
        self.personal = personal
        self.duration = duration

    def __str__(self):
        return f"'Estudio: {self.estudio} Tipo: {self.tipo_actividad} Personal: {self.personal} Duration: {self.duration}'"

    def __repr__(self):
        return f"'Estudio: {self.estudio} Tipo: {self.tipo_actividad} Personal: {self.personal} Duration: {self.duration}'"

    def __eq__(self, other):
        return self.estudio == other.estudio and self.tipo_actividad == other.tipo_actividad and self.personal == other.personal and self.duration == other.duration

    def __hash__(self):
        return hash((self.estudio, self.tipo_actividad, self.personal, self.duration))



class MixedVariableProblem(ElementwiseProblem):

    def __init__(self, **kwargs):
        self.tipos_actividades = {"R": "Recepción", "Cs": "Consentimiento", "ME": "Muestra Embarazo", "CM": "Consulta Médica", "LB": "Laboratorio", "Vc": "Vacunación", "Obs": "Observación", "PC": "Proceso de cierre"}

        self.tipo_actividades = ["R", "Cs", "ME", "CM", "LB", "Vc", "Obs", "PC"]
        self.cargo_actividad = {"R": "Enfermería", "Cs": "Enfermería", "ME": "Médico general", "CM": "Médico general", "LB": "Laboratorista", "Vc": "Médico seguridad", "Obs": "Enfermería", "PC": "Médico general"}

        self.duracion_actividad = {"R": 1, "Cs": 1, "ME": 1, "CM": 1, "LB": 1, "Vc": 1, "Obs": 1, "PC": 1}

        self.estudio = ["COVID", "Gripe"]

        #Create a object Actividad for each combination of estudio and tipos_actividades
        self.actividades = [Actividad(estudio, tipo_actividad, self.cargo_actividad[tipo_actividad], self.duracion_actividad[tipo_actividad]) for estudio in self.estudio for tipo_actividad in self.tipos_actividades]

        print(len(self.actividades))

        self.consultorios = ["C1", "C2", "C3", "C4", "C5", "C6"]
        self.horas = ["7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21", "22"]

        self.recursos = {"Médico general": 7, "Médico seguridad": 3, "Enfermería": 7, "Laboratorista": 3}

        #create a different list for every role of personal, with all the personal available for that role creating a Personal object for each one
        self.medicos_generales = [Personal(i, "Médico general") for i in range(1, self.recursos["Médico general"]+1)]
        self.medicos_seguridad = [Personal(i, "Médico seguridad") for i in range(1, self.recursos["Médico seguridad"]+1)]
        self.enfermeros = [Personal(i, "Enfermería") for i in range(1, self.recursos["Enfermería"]+1)]
        self.laboratoristas = [Personal(i, "Laboratorista") for i in range(1, self.recursos["Laboratorista"]+1)]

        self.personal = self.medicos_generales + self.medicos_seguridad + self.enfermeros + self.laboratoristas

        print(self.personal)

        vars = dict()
        self.combinations = []

        for i in self.consultorios:
            for j in self.personal:
                for k in self.horas:
                    for l in self.horas:
                        if int(k) < int(l) and (int(l) - int(k)) <= int(max(self.duracion_actividad.values())):
                            actividad = Actividad(i,k,l,j)
                            self.combinations.append(actividad)

        print(self.combinations)
        print(self.actividades)
        for i in self.actividades:
            vars[i] = Choice(options=self.combinations)

        super().__init__(vars=vars, n_obj=2, **kwargs)
    
    def _evaluate(self, X, out, *args, **kwargs):

        # Calculation of end time of each room, being X a dictionary containing the surgery number and the Surgery object

        # Initialize the end times of each room
        end_times = {"C1": 0, "C2": 0, "C3": 0, "C4": 0, "C5": 0, "C6": 0}

        # Iterate over each surgery in X
        for surgery in X.values():
            # Get the end time of the surgery
            end_time = surgery.end_time

            # Get the end time of the room
            room = surgery.operation_room
            room_end_time = end_times[room]

            # Update the end time of the room if necessary
            if end_time > room_end_time:
                end_times[room] = end_time

        penalty = 0
        
        # Restriction 1: No two surgeries can be in the same room at the same time
        penaltysameroomsametime = 0
        for room in self.consultorios:
            room_surgeries = [surgery for surgery in X.values() if surgery.operation_room == room]
            room_surgeries.sort(key=lambda surgery: surgery.start_time)
            start_time = 0
            for surgery in room_surgeries: 
                if surgery.start_time <= start_time:
                    penaltysameroomsametime += 1
                    penalty += 100
                start_time = surgery.start_time

        #Restriction 2: All surgeries have to have the right duration
        penaltyrightduration = 0
        for index, surgery in enumerate(X):
            if X[surgery].duration != surgery.duration:
                penalty += 100
                penaltyrightduration += 1

        #Restriction 3: All surgeries have to have the right personal
        penaltyrightpersonal = 0
        for index, surgery in enumerate(X):
            if X[surgery].personal.role != surgery.personal:
                penalty += 100
                penaltyrightpersonal += 1

        #Restriction 4: No personal can have two surgeries at the same time

        # Create a dictionary that maps each personal to a list of surgeries assigned to that personal
        personal_surgeries = {}
        for surgery in X.values():
            if surgery.personal not in personal_surgeries:
                personal_surgeries[surgery.personal] = []
            personal_surgeries[surgery.personal].append(surgery)

        # Check if a personal has two surgeries at the same time
        penaltypersonal = 0
        for personal in personal_surgeries:
            personal_surgeries[personal].sort(key=lambda surgery: surgery.start_time)
            end_time = None
            for surgery in personal_surgeries[personal]:
                if end_time is not None and surgery.start_time < end_time:
                    penalty += 100
                    penaltypersonal += 1
                end_time = surgery.end_time
        
        # Penalty: Restriction 5 - Activities must be completed in order

        penalty_restriction_6 = 0
        checked_surgeries = set()
        for index, i in enumerate(X):
            if i in checked_surgeries:
                continue
            checked_surgeries.add(i)
            for index2, j in enumerate(X):
                if j in checked_surgeries:
                    continue
                if i.estudio != j.estudio:
                    continue
                if self.tipo_actividades.index(i.tipo_actividad) <  self.tipo_actividades.index(j.tipo_actividad) and X[i].start_time > X[j].start_time:
                    penalty_restriction_6 += 100

        penalty += penalty_restriction_6

        # Penalty: Restriction 6 - Reduce number of empty hours
        empty_hours = 0
        for hour in self.horas:
            for surgery in X.values():
                if surgery.start_time == int(hour):
                    break
            else:
                "No surgery at this hour"
                empty_hours += 1
                if max(end_times.values()) > int(hour):
                    penalty += 100

        # Objective: Minimize the total waiting time for patients
        waiting_time = 0
        room_count = 0
        for room in self.consultorios:
            room_surgeries = [surgery for surgery in X.values() if surgery.operation_room == room]
            if room_surgeries != []:
                room_count += 1
            room_surgeries.sort(key=lambda surgery: surgery.start_time)
            end_time = 0
            for surgery in room_surgeries:
                waiting_time += max(0, surgery.start_time - end_time)
                end_time = max(end_time, surgery.end_time)

        f1 = penalty + max(end_times.values())
        f2 = penalty + waiting_time

        #ideas for objectives
        #f1 = penalty + max(end_times.values())
        #f2 = penalty + room_count
        #f3 = penalty + waiting_time
        #f4 = penalty + number of workers

        #print("Penalty = ", penalty)
        #print("Penalty Right Personal = ", penaltyrightpersonal, " Penalty Same Room Same Time = ", penaltysameroomsametime, " Penalty Right Duration = ", penaltyrightduration, "Penalty Personal = ", penaltypersonal)

        out["F"] = f1, f2


problem = MixedVariableProblem()

algorithm = NSGA2(pop_size=100,
                  sampling=MixedVariableSampling(),
                  mating=MixedVariableMating(eliminate_duplicates=MixedVariableDuplicateElimination()),
                  mutation=PolynomialMutation(prob=0.1),
                  eliminate_duplicates=MixedVariableDuplicateElimination(),
                  )

res = minimize(problem,
               algorithm,
               termination=('n_gen', 1),
               seed=1,
               verbose=True)

plot = Scatter()
plot.add(problem.pareto_front(), plot_type="line", color="black", alpha=0.7)
plot.add(res.F, facecolor="none", edgecolor="red")
plot.title = "Frente de Pareto"
plot.show()

print("Best solution found: \nX = %s\nF = %s" % (res.X, res.F))

X = res.pop
X = res.X
F = res.F


plt.figure(figsize=(7, 5))
plt.scatter(F[:, 0], F[:, 1], s=30, facecolors='none', edgecolors='blue')
plt.title("Objective Space")
plt.show()

fl = F.min(axis=0)
fu = F.max(axis=0)
print(f"Scale f1: [{fl[0]}, {fu[0]}]")
print(f"Scale f2: [{fl[1]}, {fu[1]}]")

approx_ideal = F.min(axis=0)
approx_nadir = F.max(axis=0)

plt.figure(figsize=(7, 5))
plt.scatter(F[:, 0], F[:, 1], s=30, facecolors='none', edgecolors='blue')
plt.scatter(approx_ideal[0], approx_ideal[1], facecolors='none', edgecolors='red', marker="*", s=100, label="Ideal Point (Approx)")
plt.scatter(approx_nadir[0], approx_nadir[1], facecolors='none', edgecolors='black', marker="p", s=100, label="Nadir Point (Approx)")
plt.title("Puntos Nadir e Ideal (Approx)")
plt.legend()
plt.show()

nF = (F - approx_ideal) / (approx_nadir - approx_ideal)

fl = nF.min(axis=0)
fu = nF.max(axis=0)
print(f"Scale f1: [{fl[0]}, {fu[0]}]")
print(f"Scale f2: [{fl[1]}, {fu[1]}]")

plt.figure(figsize=(7, 5))
plt.scatter(nF[:, 0], nF[:, 1], s=30, facecolors='none', edgecolors='blue')
plt.title("Frente de Pareto")
plt.show()

weights = np.array([0.7, 0.3])

from pymoo.decomposition.asf import ASF

decomp = ASF()

i = decomp.do(nF, 1/weights).argmin()

print("Best regarding ASF: Point \ni = %s\nF = %s \nX = %s" % (i, F[i], X[i]))

plt.figure(figsize=(7, 5))
plt.scatter(F[:, 0], F[:, 1], s=30, facecolors='none', edgecolors='blue')
plt.scatter(F[i, 0], F[i, 1], marker="x", color="red", s=200)
plt.title("Solución Subóptima Elegida por ASF")
plt.show()



print("Best solution found: \nX = %s\nF = %s" % (res.X, res.F))
