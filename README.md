# AutomaticR0
Calcul automatique des nombres de reproduction $\mathcal{R}\_0$ et $\mathcal{R}\_t$ à partir d'un fichier .json contenant un modèle. Ce code est présenté dans le cadre de la maîtrise en informatique.

# Usage
## Pour utiliser les fonctions
Importer automaticR0.py, numpy et pyplot.

## Functions
Les fonctions suivantes sont pertinentes pour étudier les modèles donnés:
- `plotCurves(xPoints: np.ndarray or list, curves: np.ndarray or list, toPlot: list, labels: list) -> None`
- `loadModel(name: str) -> dict`
- `initialize(model: dict, y0: dict, t: float, originalModel: dict = None, r0: bool = True) -> None`
- `getCompartments(model: dict) -> list`
- `solve(model: dict, tRange: tuple) -> tuple`
- `mod(model: dict) -> dict`
- `computeR0(modelName: str) -> tuple`
- `compare(modelName: str, t_span_rt: tuple, sub_rt: float = 1, R0: float = 0) -> None`
- `createLaTeX(model: dict)`

Voici un exemple de code qui utilise ces fonctions:
```python
# libraries
import automaticR0 as aR0
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('TkAgg')

# modèle original
SIR = aR0.loadModel('SIR')
solution, t_span = aR0.solve(SIR, (0, 60))
labels = aR0.getCompartments(SIR)

# modification des données pour le graphique
nullmPos = labels.index('Null_m')
nullnPos = labels.index('Null_n')

solution[:, nullmPos] = - solution[:, nullmPos]
curvesToPlot = list(range(solution.shape[1]))

diffNullm = np.max(np.abs(solution[:, nullmPos] - solution[0, nullmPos]))
diffNulln = np.max(np.abs(solution[:, nullnPos] - solution[0, nullnPos]))
if diffNullm < 1:
    curvesToPlot.remove(nullmPos)
if diffNulln < 1:
    curvesToPlot.remove(nullnPos)

# graphique des courbes
fig = plt.figure()
aR0.plotCurves(t_span,
               np.transpose(solution),
               toPlot=curvesToPlot,
               labels=labels)
plt.yscale('log')
plt.ylim(bottom=10 ** -3, top=10 ** int(np.log10(np.max(solution)) + 1))

# calcul de R0
# cette fonction renvoie le modèle original et le modèle modifié
SIR, SIRmod, _, R0 = aR0.computeR0('SIR')
print(f'Computation of R0: {R0}')

# modèle modifié
SIRmod = aR0.mod(SIR)
# initialisation
init = {key: solution[0, i]
        for i, key in enumerate(labels)}
aR0.initialize(SIRmod, init, 0, originalModel=SIR, r0=True)

solution, t_span = aR0.solve(SIRmod, (0, 60))
labels = aR0.getCompartments(SIRmod)

nullmPos = labels.index('Null_m')
nullnPos = labels.index('Null_n')

solution[:, nullmPos] = - solution[:, nullmPos]
curvesToPlot = list(range(solution.shape[1]))

diffNullm = np.max(np.abs(solution[:, nullmPos] - solution[0, nullmPos]))
diffNulln = np.max(np.abs(solution[:, nullnPos] - solution[0, nullnPos]))
if diffNullm < 1:
    curvesToPlot.remove(nullmPos)
if diffNulln < 1:
    curvesToPlot.remove(nullnPos)

fig = plt.figure()
aR0.plotCurves(t_span,
               np.transpose(solution),
               toPlot=curvesToPlot,
               labels=labels)
plt.yscale('log')
plt.ylim(bottom=10 ** -3, top=10 ** int(np.log10(np.max(solution)) + 1))

# comparaison des méthodes
# il faut entrer le R0 analytique manuellement pour comparer correctement
aR0.compare('SIR', (0, 60), sub_rt = 1, R0 = 4)

plt.show()

# création du fichier LaTeX pour ce modèle
aR0.createLaTeX(SIR)
print(open("LaTeX/SIR.tex").read())
```

D'autres fonctions (et plus d'information sur les sorties de chaque fonction) peuvent être retrouvées dans le fichier `own.nb` qui montre les applications principales du code.

## Models
Plusieurs exemples de modèles peuvent être retrouvés dans le dossier "models". Il est important de noter que ces fichiers doivent être formattés d'une manière précise, et ils doivent être placés dans ce dossier lorsque le code est executé.
