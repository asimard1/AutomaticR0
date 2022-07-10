# AutomaticR0
Automatic R0 computation from a configuration file.

# Usage
Simply import the file functions and then use functions "loadModel", "solve", "computeRt" or "computeR0" according to the situation.

Function "solve" can be used as such: f.solve(f.loadModel('SIR'), (0, 200), 100) will load the file with name "model/SIR.json" and then solve the associated model for times 0 < t < 200 with precision 1/100.

Function "computeRt" is used as: f.computeRt('SIR', (0, 100)) which will compute all Rt values for times between 0 and 100.

Function "computeR0" is used as: f.computeR0('SIR') which will compute all R0 values for each contact in the model.

Other functions (and more information on the outputs of above functions) can be found in the notebook file which shows the main applications of the code.
