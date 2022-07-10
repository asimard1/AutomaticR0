# AutomaticR0
Automatic $\mathcal{R}_0$ computation from a configuration file.

# Usage
## Functions
Simply import the file functions and then use the following functions according to the situation:
- loadModel
- solve
- computeRt
- computeR0.

Here is an example for each function:
- `f.solve(f.loadModel('SIR'), (0, 200), 100)` will load the file with name "model/SIR.json" and then solve the associated model for times $0 < t < 200$ with precision $1/100$.
- `f.computeRt('SIR', (0, 100), write=True)` will compute all Rt values for times between $0$ and $100$ (another parameter can be added for variation between times) as well as create a new json file to contain the modified model.

- `f.computeR0('SIR', write=True)` will compute all $\mathcal{R}_0$ values for each contact in the model as well as store the modified model.

Other functions (and more information on the outputs of above functions) can be found in the notebook file which shows the main applications of the code.

## Models
Some model examples can be found in folder "models". It is important to note that all models have to be placed directly in the folder "models" relative to the notebook file when running.
