# AutomaticR0
Automatic $\mathcal{R}\_0$ computation from a .json model file.

# Usage
## Functions
Simply import automaticR0.py, numpy and pyplot and then use the following functions according to the situation:
- `plotCurves(xPoints: np.ndarray or list, curves: np.ndarray or list, toPlot: list, labels: list) -> None`
- `loadModel(name: str) -> dict`
- `initialize(model: dict, y0: dict, t: float, originalModel: dict = None, r0: bool = True) -> None`
- `getCompartments(model: dict) -> list`
- `solve(model: dict, tRange: tuple) -> tuple`
- `mod(model: dict) -> dict`
- `computeR0(modelName: str) -> tuple`
- `compare(modelName: str, t_span_rt: tuple, sub_rt: float = 1, R0: float = 0) -> None`
- `createLaTeX(model: dict)`

Here is an example code that uses every function above:
```python
import automaticR0 as aR0
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('TkAgg')

SIR = aR0.loadModel('SIR')
solution, t_span = aR0.solve(SIR, (0, 60))
labels = aR0.getCompartments(SIR)

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

SIR, SIRmod, _, R0 = aR0.computeR0('SIR')
print(f'Computation of R0: {R0}')

SIRmod = aR0.mod(SIR)
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

aR0.compare('SIR', (0, 60), sub_rt = 1, R0 = 4)

plt.show()

aR0.createLaTeX(SIR)
print(open("LaTeX/SIR.tex").read())
```

Other functions (and more information on the outputs of above functions) can be found in the notebook file which shows the main applications of the code.

## Models
Some model examples can be found in folder "models". It is important to note that all models have to be placed directly in the folder "models" relative to the notebook file when running.
