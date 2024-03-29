import numpy as np
from dataclasses import dataclass
import math
from math import *
import matplotlib.pyplot as plt
import time
import json
from tqdm import tqdm
import os
from scipy.signal import argrelextrema

PLUM = '#8E4585'

# Use GPU if wanted
useTorch = False
if useTorch:
    from torchdiffeq import odeint
    import torch
    device = torch.device(
        'cuda:' + str(0) if torch.cuda.is_available() else 'cpu')
    print(
        f'Using device {device} with name {torch.cuda.get_device_name(device)}')
else:
    from scipy.integrate import odeint


@dataclass
class Delta:
    flux: list


@dataclass
class Flux:
    coef_indices: tuple([int, int])
    rate_index: list
    contact_index: list


types = [Flux]
functions = {}
derivatives = None


def storeFunctions(model: dict):
    """
    Stores all functions from model into dictionary.

    Inputs:
        model: dictionary
            Model for which we store the functions.

    Outputs:
        None.
    """
    global functions
    modelName = model['name']
    functions[modelName] = {}

    flows = model['flows']
    for flowType_index, flowType in enumerate(flows):
        for flow_index, flow in enumerate(flows[flowType]):
            functions[modelName][f"{flowType_index, flow_index}"] \
                = eval('lambda t: ' + str(flow['parameter']))


def removeDuplicates(liste: list) -> list:
    """
    Removes duplicates from list.

    Inputs:
        liste: list
            List to modify.

    Outputs:
        newList: list
            List without duplicates.
    """
    newList = list(dict.fromkeys(liste))
    return newList


def rreplace(s, old, new, n):
    """
    Replaces last n occurences in string.

    Inputs:
        s: string
            Original string to modify.
        old: string
            What to replace in string.
        new: string
            What to replace with.
        n: int
            Number of instances to replace.

    Outputs:
        newString: string
            Modified string.
    """
    li = s.rsplit(old, n)
    newString = new.join(li)
    return newString


def plotCurves(xPoints: np.ndarray or list, curves: np.ndarray or list,
               toPlot: list, labels: list,
               title: str = 'Infection curves', style: list = None,
               xlabel: str = 'Temps (jours)', ylabel: str = 'Nb. d\'individus',
               scales: list = ['linear', 'linear'],
               axes=plt, legendLoc: str = 'best',
               colors: list = None, ycolor: str = 'black') -> None:
    """
    Plots given curves. If xPoints are the same for all curves, give only np.ndarray.
    Otherwise, a list of np.ndarrays works, in which case it has to be given for every curve.
    Other options (title, labels, scales, etc.) are the same as for matplotlib.pyplot.plot function.

    Inputs:
        xPoints: np.ndarray or list
            Points to use for x axis.
        curves: np.ndarray or list
            Points to use for y axis.
        toPlot: list
            Curves to plot in given curves.
        labels: list
            Labels to use for given curves.
        title: string
            Title for graph.
        style: list
            Styles to use for each curve.
        xLabel: str
            Label of x axis.
        yLabel: str
            Label of y axis.
        scales: list
            Scales to use for each axis.
        axes: figure or axes
            Where to draw the curves.
        legendLoc: str
            Where to place legend.
        colors: list
            Colors to use for each curve.
        ycolor: str
            Color to use for y axis.

    Ouputs:
        None.
    """

    # Create missing lists if given None.
    liste = list(range(max(toPlot) + 1))
    if style == None:
        style = ['-' for _ in liste]
    if type(style) == type(''):
        style = [style for _ in liste]
    if colors == None:
        colors = [f'C{x}' for x in liste]

    k = 0
    # TODO rewrite using try except
    if type(xPoints) is np.ndarray:  # Only one set of x coordinates
        for curve in toPlot:
            if labels == None:
                axes.plot(xPoints,
                          curves[curve],
                          style[curve],
                          c=colors[curve])
                k += 1
            else:
                axes.plot(xPoints,
                          curves[curve],
                          style[curve],
                          label=labels[curve],
                          c=colors[curve])
                k += 1
    else:  # Different time scales
        for curve in toPlot:
            if labels == None:
                axes.plot(xPoints[curve],
                          curves[curve],
                          style[curve],
                          c=colors[curve])
                k += 1
            else:
                axes.plot(xPoints[curve],
                          curves[curve],
                          style[curve],
                          label=labels[curve],
                          c=colors[curve])
                k += 1

    if labels != None:
        axes.legend(loc=legendLoc)

    try:
        axes.title(title)
        axes.xlabel(xlabel)
        axes.ylabel(ylabel, color=ycolor)
        axes.xscale(scales[0])
        axes.yscale(scales[1])
    except:
        axes.set_title(title)
        axes.set_xlabel(xlabel)
        axes.set_ylabel(ylabel, color=ycolor)
        axes.set_xscale(scales[0])
        axes.set_yscale(scales[1])


def verifyModel(model: dict, modelName: str, printText: bool = True) -> None:
    """
    Verifies if model has the right properties. Might not be complete.

    Inputs:
        model: dict
            Model to verify.
        modelFile: str
            Name to compare model with.
        printText: bool
            Whether or not to print debug text.

    Ouputs:
        None.
    """

    # Try fixing the model if there are problems with nulls.
    missingNulln = False
    missingNullm = False
    if 'Null_n' not in model['compartments']:
        missingNulln = True
    if 'Null_m' not in model['compartments']:
        missingNullm = True

    if missingNulln or missingNullm or 'Null' in getCompartments(model):
        if printText:
            print('Fixing missing empty nodes.')
        flows = model['flows']
        model['compartments']['Null_n'] = 0
        model['compartments']['Null_m'] = 0

        # We don't need Null anymore
        if 'Null' in model['compartments']:
            del model['compartments']['Null']
        print('Model should be fixed...')

    # Solve Null nodes in edges. Replace Null with their right instances.
    flows = model['flows']
    problem = False
    for _, flowType in enumerate(flows):
        for _, flow in enumerate(flows[flowType]):
            for i, arg in enumerate(['to', 'from', 'rate', 'contact']):
                if flow[arg].lower() == 'null' and arg in ['to', 'from', 'rate', 'contact']:
                    problem = True
                    # i even: change null to null_n
                    # i odd: change null to null_m
                    flow[arg] = 'Null' + ('_n' if not i % 2 else '_m')
    if problem and printText:
        print('Les noeuds vides sont réglés dans les arrêtes')

    if model['name'] != modelName:
        raise Exception(f"Model doesn\'t have right name in file.\n"
                        + f"Name in file: {model['name']}. Wanted name: {modelName}.")

    missing = []
    flows = model['flows']
    for flowType in flows:
        for flow in flows[flowType]:
            # Check for missing keys in flows
            keys = list(flow.keys())
            for p in ['from', 'to', 'rate', 'contact', 'parameter']:
                if p not in keys and p not in missing:
                    missing.append(p)

            #     if p in keys and flow[p] == 'Null':
            #         if p in ['from', 'rate']:
            #             flow[p] = 'Null_n'

            # print(flow)

            # TODO check for flow informations to respect the structure defined

    if missing != []:
        missingStr = ', '.join(list(map(lambda x: f'"{x}"', missing)))
        missingStr = rreplace(missingStr, ', ', ' and ', 1)
        raise Exception(
            f'Some flows are missing parameters {missingStr} in model {modelName}.')

    if printText:
        print('Model verified.')

    return model


def loadModel(name: str, overWrite=True, printText: bool = True) -> dict:
    """
    Loads model from file.

    Inputs:
        name: str
            Name of model to load. Function will load model [name].json.
        overWrite: bool
            Whether or not to overwrite file after loading model.
        printText: bool
            Whether or not to print debug text.

    Outputs:
        model: dict
            Model loaded.
    """
    try:
        with open(f'models/{name}.json', 'r') as file:
            model = json.load(file)
    except:
        time.sleep(1)
        try:
            with open(f'models/{name}.json', 'r') as file:
                model = json.load(file)
        except:
            raise Exception(f'Could not open file {name}.json.')

    # Verify
    model = verifyModel(model, name, printText=printText)
    # Write to file
    writeModel(model, overWrite=overWrite, printText=printText)
    # Store functions in dictionary
    storeFunctions(model)
    return model


def initialize(model: dict, y0: dict, t: float, originalModel:
               dict = None, r0: bool = True, printText: bool = True,
               whereToAdd: str = 'to', scaled=False) -> None:
    """
    This modifies model variable, but doesn't modify the file it comes from.

    Inputs:
        model: dict
            Model which needs to be modified.
        y0: dict
            New values to initialize with.
        t: float
            Time at which we are initializing.
        scaled: bool
            Whether or not to rescale infected when initializing.
        originalModel: dict
            Dictionary to use as template for modified model. None if no modified model.
        printText: bool
            Whether or not to print debug text.
        whereToAdd: str
            Where to add new infections.

    Outputs:
        None.
    """

    if originalModel != None:
        # If we are initializing using another model
        if printText and r0:
            print(f'Initializing with values {roundDict(y0, 0)} at time {t}.')
        weWant = getCompartments(originalModel)
        if sorted(list(y0.keys())) != sorted(weWant):
            raise Exception("Initialization vector doesn't have right entries.\n"
                            + f"    Entries wanted:   {weWant}.\n"
                            + f"    Entries obtained: {list(y0.keys())}.")

        if scaled:
            infectious = infsScaled(originalModel, y0, t, whereToAdd)
        else:
            infectious = infs(originalModel, y0, t, whereToAdd)

        toDuplicate = isolated(originalModel)
        for compartment in y0:
            try:
                if compartment in toDuplicate:
                    model['compartments'][addI(
                        compartment, 0)] = infectious[compartment]
                model['compartments'][addI(
                    compartment, 1)] = y0[compartment] \
                    - infectious[compartment]
            except:
                model['compartments'][addI(
                    compartment, 1)] = y0[compartment]
    else:
        # No need for any ajustments. Simply write the values.
        for compartment in list(y0.keys()):
            model['compartments'][compartment] \
                = y0[compartment]


def getCompartments(model: dict) -> list:
    """
    List of compartments in model file.

    Inputs:
        model: dict
            Model of interest.

    Ouputs:
        compartments: list
            List of compartments in model.
    """
    compartments = list(model['compartments'].keys())
    return compartments


def getFlowsByCompartments(model: dict) -> list:
    """
    Get all inflows and outflows for model.

    Inputs:
        model: dict
            Model of interest.

    Ouputs:
        FBC: list
            Flows by compartments.
    """
    compartments = getCompartments(model)
    flows = model['flows']

    # FlowByCompartment
    FBC = [([], []) for _ in compartments]

    for flowType_index, flowType in enumerate(flows):
        for flow_index, flow in enumerate(flows[flowType]):
            to_i = compartments.index(
                flow['to'])
            from_i = compartments.index(
                flow['from'])

            rate_i = list(map(compartments.index,
                          [x for x in flow['rate'].split('+') if not x.startswith('Null')]))
            contact_i = list(map(compartments.index,
                                 [x for x in flow['contact'].split('+') if not x.startswith('Null')]))
            term = Flux((flowType_index, flow_index), rate_i, contact_i)

            try:
                # Inflow
                FBC[to_i][0].append(term)
            except:
                pass
            try:
                # Outflow
                FBC[from_i][1].append(term)
            except:
                pass

    # Create delta (list of flows) with each flux
    FBC = [[Delta(*[[f for f in flows if isinstance(f, T)] for T in types])
            for flows in compartment] for compartment in FBC]

    return FBC


def getPopNodes(model: dict) -> list:
    """
    Return compartments that are in the population only.

    Inputs:
        model: dict
            Model of interest.

    Outputs:
        weWant: list
            List of compartments in population.
    """
    compartments = getCompartments(model)
    weWant = [x for x in compartments if not x.startswith(('Rt', 'Null'))]
    return weWant


def getSusceptibleNodes(model: dict) -> list:
    """
    Return all susceptible nodes, from definition 1.2.3 in thesis.

    Inputs:
        model: dict
            Model of interest.

    Outputs:
        weWant: list
            list of susceptible nodes in population.
    """

    population = getPopNodes(model)
    flows = model['flows']

    weWant = []

    for category in flows:
        for flow in flows[category]:
            # On veut des noeuds vides pour considérer les entrées et sorties
            # Puisque les noeuds prennent de la place on les crée seulement lorsque nécessaire

            _, _, v_r, _ = flow['from'], flow['to'], flow['rate'], flow['contact']
            flowType = getFlowType(flow)

            if flowType == 'contact':  # we know v_r != Null
                rates = v_r.split('+')
                weWant += [node for node in rates if node not in weWant]

    problems = [x for x in weWant if x not in population]
    if problems != []:
        print(f'Found a susceptible not in population. Excluding {problems}.')

    return [x for x in population if x in weWant]


def getOtherNodes(model: dict) -> list:
    """
    Return compartments that are not in the population.

    Inputs:
        model: dict
            Model of interest.

    Outputs:
        weWant: list
            List of compartments not in population.
    """
    compartments = getCompartments(model)
    weWant = [x for x in compartments if x.startswith(('Rt', 'Null'))]
    return weWant


def getRtNodes(model: dict) -> list:
    """
    Return compartments that are used for computing Rt's.

    Inputs:
        model: dict
            Model of interest.

    Outputs:
        weWant: list
            List of compartments for Rt's.
    """
    compartments = getCompartments(model)
    weWant = [x for x in compartments if x.startswith(('Rt'))]
    return weWant


def getNodeValues(model: dict, state: np.ndarray or list, weWant: list) -> dict:
    """
    Get every value for nodes in weWant, as dictionary.

    Inputs:
        model: dict
            Model of interest.
        state: np.ndarray or list
            Value of each node in the model.
        weWant: list
            List of compartments of interest.

    Ouputs:
        dictNb: dict
            Value of each node in weWant.
    """
    if len(state.shape) != 1:
        # Prevent complete solutions from being given as input.
        # This is to be used on a single state.
        raise Exception('2nd argument should be a vector, not matrix.')

    dictNb = {}
    compartments = getCompartments(model)

    indexes = list(map(compartments.index, weWant))
    for i in indexes:
        dictNb[compartments[i]] = state[i]
    dictNb['Sum'] = sum(state[i] for i in indexes)

    return dictNb


def getPopulation(model: dict, state: np.ndarray or list) -> dict:
    """
    Get every value for nodes in population as dictionary.

    Inputs:
        model: dict
            Model of interest.
        state: np.ndarray or list
            Value of each node in the model.

    Ouputs:
        dictNb: dict
            Value of each node in population.
    """
    dictNb = getNodeValues(model, state, getPopNodes(model))
    return dictNb


def getPopChange(model: dict, solution: np.ndarray) -> float:
    """
    Get change in population from start to finish.

    Inputs:
        model: dict
            Model of interest.
        solution: np.ndarray
            Solution given by solve function.

    Outputs:
        popChange: float
            Change in population over solution given.
    """
    popChange = getPopulation(
        model, solution[-1])['Sum'] - getPopulation(model, solution[0])['Sum']
    return popChange


def getOthers(model: dict, state: np.ndarray or list) -> dict:
    """
    Get every value for nodes not in population as dictionary.

    Inputs:
        model: dict
            Model of interest.
        state: np.ndarray or list
            Value of each node in the model.

    Ouputs:
        dictNb: dict
            Value of each node not in population.
    """
    dictNb = getNodeValues(model, state, getOtherNodes(model))
    return dictNb


def getOtherChange(model: dict, solution: np.ndarray) -> float:
    """
    Get change in other nodes from start to finish.

    Inputs:
        model: dict
            Model of interest.
        solution: np.ndarray
            Solution given by solve function.

    Outputs:
        otherChange: float
            Change in other nodes over solution given.
    """
    otherChange = getOthers(
        model, solution[-1])['Sum'] - getOthers(model, solution[0])['Sum']
    return otherChange


def getCoefForFlux(model: dict, flux: Flux, t: float) -> float:
    """
    Gets the coefficient for flux from the config file.

    Inputs:
        model: dict
            Model of interest.
        flux: Flux
            Flux for which we need coefficient.
        t: float
            Time at which we compute coefficient.

    Outputs:
        value: float
            Value of coefficient at given time.
    """
    # Information read from stored functions.
    coef = functions[model['name']
                     ][f"{flux.coef_indices[0], flux.coef_indices[1]}"]
    value = coef(t)
    return value


def getCoefForFlow(flow: dict, t: float) -> float:
    """
    Gets the coefficient for flow given from the config file.

    Inputs:
        flow: dict
            Flow for which we need coefficient.
        t: float
            Time at which we compute coefficient.

    Outputs:
        value: float
            Value of coefficient at given time.
    """
    # Information read directly from flow dictionary.
    string = str(flow['parameter'])
    fonc = eval('lambda t: ' + string)
    value = fonc(t)

    return value


def evalDelta(model: dict, delta: Delta, state: np.ndarray or list,
              t: float, t0: float) -> float:
    """
    Computes the actual derivative for a delta (dataclass defined earlier).

    Inputs:
        model: dict
            Model of interest
        delta: Delta
            List of Flows.
        state: np.ndarray or list
            Value of each node in the model.
        t: float
            Time at which we compute the derivative.

    Outputs:
        somme: float
            Sum of all flux in delta.
    """

    compartments = getCompartments(model)

    N = sum(state[i] for i, comp in enumerate(compartments)
            if not comp.startswith(('Null', 'Rt')))

    rateInfluence = [sum(state[x] for x in flux.rate_index)
                     if len(flux.rate_index) > 1
                     else (state[flux.rate_index[0]]
                           if len(flux.rate_index) == 1
                           else 1)
                     for flux in delta.flux]
    contactInfluence = [sum(state[x] for x in flux.contact_index) / N
                        if len(flux.contact_index) > 1
                        else (state[flux.contact_index[0]] / N
                              if len(flux.contact_index) == 1
                              else 1)
                        for flux in delta.flux]

    coefsInfluence = [getCoefForFlux(model, flux, t if t0 is None else t0)
                      for flux in delta.flux]

    somme = np.einsum('i,i,i', rateInfluence, contactInfluence, coefsInfluence)

    return somme


def derivativeFor(model: dict, compartment: str):
    """
    Get the derivative for a compartment as a function of state and time.

    Inputs:
        model: dict
            Model of interest.
        compartments: str
            Compartment for which we want to get the derivative.

    Outputs:
        derivativeForThis: function
            Function for derivative according to state and time.
    """

    compartments = getCompartments(model)
    i = compartments.index(compartment)

    FBC = getFlowsByCompartments(model)

    def derivativeForThis(state, t, t0):
        # evalDelta takes care of getting the right coefficient for the derivative
        inflows = evalDelta(model, FBC[i][0], state, t, t0)
        outflows = evalDelta(model, FBC[i][1], state, t, t0)
        return inflows - outflows
    return derivativeForThis


def model_derivative(t: float, state: list or torch.Tensor or np.ndarray, t0: float) -> list:
    """
    Gets the derivative functions for every compartments evaluated at given state.

    Inputs:
        state: list or torch.Tensor or np.ndarray
            Value of each node in the model.
        t: float
            Time at which we are evaluating.
        derivatives: list
            List of functions containing the derivative of each compartment.

    Outputs:
        dstate_dt: float
            Evaluation of the derivative of each compartment.
    """
    # state = [x if x > 0 else 0 for x in state]
    global derivatives

    if useTorch:
        dstate_dt = torch.FloatTensor(
            [derivatives[i](state, t, t0) for i in range(len(state))])
    else:
        [t, state] = [state, t]
        dstate_dt = np.array(
            [derivatives[i](state, t, t0) for i in range(len(state))])
    return dstate_dt


def solve(model: dict, tRange: tuple, refine: int = 100, printText=False, t0=None) -> tuple:
    """
    Model solver, uses odeint from scipy.integrate.

    Inputs:
        model: dict
            Model of interest.
        tRange: tuple
            Time range considered.
        refine: int
            Precision used for integration.
        printText: bool
            Whether or not to print debug text.

    Outputs:
        solution: np.ndarray
            Solver solution for each compartment.
        t_span: np.ndarray
            All time values for the generated solution.
    """
    ti = time.time()

    global derivatives

    compartments = getCompartments(model)
    steps = (tRange[1] - tRange[0]) * refine + 1
    t_span = np.linspace(tRange[0], tRange[1], num=math.ceil(steps))

    derivatives = [derivativeFor(model, c)
                   for c in compartments]

    y0 = np.array([model['compartments'][comp]
                   for comp in compartments])
    if useTorch:
        solution = odeint(model_derivative, torch.FloatTensor(y0),
                          torch.FloatTensor(t_span), args=(t0,))
    else:
        solution = odeint(model_derivative, y0, t_span, args=(t0,))

    if printText:
        print(f'Model took {time.time() - ti:.1e} seconds to solve.')

    return solution, t_span


def getFlowType(flow: dict) -> str:
    """
    Returns the type of a given flow.

    Inputs:
        flow: dict
            Flow for which we want the type.

    Outputs:
        type: str
            Type of the flow.
    """
    if flow['rate'].startswith('Null'):
        if flow['contact'].startswith('Null'):
            return 'batch'
        else:
            raise Exception(f'Invalid flow type: {flow}.')
    else:
        if flow['contact'].startswith('Null'):
            return 'rate'
        else:
            return 'contact'


def addI(node: str, i: int) -> str:
    """
    Adds layer number to a given node.

    Inputs:
        node: str
            Node name to modify.
        i: int
            Number to add.

    Outputs:
        newNode: str
            Modified node name.
    """
    if node.startswith(('Null', 'Rt')):
        newNode = node
    else:
        newNode = (node + f'^{i}')
    return newNode


def removeI(node: str) -> str:
    """
    Get base node for a numbered one.

    Inputs:
        node: str
            Node of interest.

    Outputs:
        newNode: str
            Base node name.
    """
    if len(node) > 1:
        newNode = node[:-2] if node[-2] == '^' else node
    else:
        newNode = node
    return newNode


def getI(node: str) -> str:
    """
    Get layer number for node.

    Inputs:
        node: str
            Node of interest.

    Outputs:
        i: int
            Layer containing node.
    """
    remove = len(removeI(node))

    if remove == len(node):
        return -1
    else:
        return int(node[remove + 1:])


def joinNodeSum(nodes: list) -> str:
    """
    Joins a node list with a sum.

    Inputs:
        nodes: list
            List of nodes to join.

    Outputs:
        sum: string
            Sum of nodes as a string.
    """
    return '+'.join(removeDuplicates(nodes))


def addIList(nodes: list, i: int) -> str:
    """
    Adds i to all nodes in list "nodes".

    Inputs:
        nodes: list
            List of nodes to update.
        i: int
            Superior index to add.

    Outputs:
        newNodes: list
            Modified nodes.
    """
    newNodes = joinNodeSum(list(map(
        lambda x: addI(x, i),
        nodes
    )))
    return newNodes


def splitVrVc(nodes, newCompartments) -> str:
    """
    Splits vr or vc node if there is something to split to.

    Inputs:
        nodes: list
            List of nodes to split.
        newCompartments: list
            List of compartments in the new model.

    Outputs:
        newVrVc: str
            String containing split nodes.
    """
    def splitIfExists(x):
        if addI(x, 0) in newCompartments:
            return joinNodeSum([addI(x, j) for j in range(2)])
        else:
            return addI(x, 1)
    newVrVc = joinNodeSum(list(map(splitIfExists,
                                   nodes)))

    return newVrVc


def subSet(model, start: str, end: str):
    """
    Gets all edges in graph as a dictionary.

    Inputs:
        model: dict
            Model of interest.
        u: str
            Start point for search.
        d: str
            Destination for search

    Outputs:
        edges: dict
            All edges in model.
    """
    compartments = getCompartments(model)
    visited = {comp: False for comp in compartments}
    edges = edgesCut(model)

    allPaths = []
    cycles = []

    searchPaths(start, end, visited, edges, [], allPaths, cycles)

    subSet = []

    for path in allPaths:
        subSet += [node for node in path if node not in subSet]

    for cycle in [cycle for cycle in cycles if cycle[-1] in subSet]:
        subSet += [node for node in cycle if node not in subSet]

    return [node for node in compartments if node in subSet]  # reorder


def subSetVc(model, u: str, vc: list):
    """
    Gets all edges in graph as a dictionary.

    Inputs:
        model: dict
            Model of interest.
        u: str
            Start point for search.
        vc: list
            List of destinations for search

    Outputs:
        edges: dict
            All edges in model.
    """

    allNodes = []
    for d in vc:
        for node in subSet(model, u, d):
            if node not in allNodes:
                allNodes.append(node)

    return allNodes


def searchPaths(u: str, d: str, visited: dict, edges: dict,
                path: list, allPaths: list, cycles: list):
    """
    Returns the subSet in the model that lies between u and v (without loops).
    See https://www.geeksforgeeks.org/find-paths-given-source-destination/.

    Inputs:
        u: str
            Start point for search.
        d: str
            Destination point for search.
        visited: dict
            Dictionary containing information on searched vertices.

    Outputs:
        subSet: list
            List of nodes that are in the subSet.
    """

    visited[u] = True
    path.append(u)

    if u == d:
        allPaths.append(path.copy())
    else:
        for descendant in edges[u]:
            if not visited[descendant]:
                searchPaths(descendant,
                            d,
                            visited,
                            edges,
                            path,
                            allPaths,
                            cycles)
            else:
                cycles.append((path + [descendant]).copy())

    path.pop()
    visited[u] = False


def edgesCut(model):
    """
    Gets all edges that do not go to u in graph as a dictionary.

    Inputs:
        model: dict
            Model of interest.
        u: str
            Start point for search.

    Outputs:
        edges: dict
            Some edges in model.
    """

    flows = model['flows']
    compartments = getCompartments(model)

    edges = {comp: [] for comp in compartments}

    for _, flowName in enumerate(flows):
        for flow in flows[flowName]:
            if getFlowType(flow) != 'contact' and\
                    not flow['to'] in edges[flow['from']]:
                edges[flow['from']].append(flow['to'])

    return edges


def isolated(model):
    """
    Returns the isolated layer of a model, to be duplicated.
    Takes the union of all isolations over contact edges.

    Inputs:
        model: dict
            Given model being modified.

    Outputs:
        toDuplicate: list
            List of nodes to be duplicated in isolated layer.
    """
    flows = model['flows']

    toDuplicate = []
    for flowName in flows:
        for flow in flows[flowName]:
            newFlow = {
                "from": "Null_n",
                "to": "Null_m",
                "rate": "Null_n",
                "contact": "Null_m",
                "parameter": "0"
            }

            u = flow['from']
            v = flow['to']
            vr = flow['rate'].split('+')
            vc = flow['contact'].split('+')

            if getFlowType(flow) == 'contact':
                for node in subSetVc(model, v, vc):
                    if node not in toDuplicate:
                        toDuplicate.append(node)

    return toDuplicate


def mod(model: dict,
        printText: bool = False,
        write=True, overWrite=True) -> dict:
    """
    This function modifies the given model to let us compute Rt.

    Inputs:
        model: dict
            Given model to modify.
        printWarnings: bool
            Whether or not to print warning text.
        printText: bool
            Whether or not to print debug text.
        autoInfections: bool
            Whether or not to include autoinfections in modification.
        write: bool
            Whether or not to write modified model to json file.
        overWrite: bool
            Whether or not to overwrite old file when writing.

    Outputs:
        newModel: dict
            Modified model.
    """
    if printText:
        print('\nCreating new model!')

    newModel = {"name": model['name'] + '_mod',
                "compartments": {}, "flows": {}}

    compartments = getCompartments(model)
    flows = model['flows']

    # Verify if structure is already well implemented
    if 'Rt' in list(map(lambda x: x[:2], compartments)):
        raise Exception(
            f"There is already a node called 'Rt...', please change its name. Model {model['name']}.")
    if 'Null_n' not in compartments:
        raise Exception(
            "There are no nodes called 'Null_n'. Cannot identify empty nodes.")
    if 'Null_m' not in compartments:
        raise Exception(
            "There are no nodes called 'Null_m'. Cannot identify empty nodes.")

    # Add base compartments
    for compartment in compartments:
        if not compartment.startswith('Null'):
            newModel["compartments"][addI(compartment, 1)] \
                = model["compartments"][compartment]

    # Add isolated layer

    for node in isolated(model):
        newModel["compartments"][addI(node, 0)] = 0

    newCompartments = getCompartments(newModel)
    # print(newCompartments)
    # print(toDuplicate)

    # Add edges and their informations
    for flowName in flows:
        newModel['flows'][flowName] = []
        for flow in flows[flowName]:
            newFlow = {
                "from": "Null_n",
                "to": "Null_m",
                "rate": "Null_n",
                "contact": "Null_m",
                "parameter": "0"
            }

            # Informations from original flow
            u = flow['from']
            v = flow['to']
            vr = flow['rate'].split('+')
            vc = flow['contact'].split('+')

            ### RATES ###
            if getFlowType(flow) == 'rate':
                uPrime = addI(u, 1)
                vPrime = addI(v, 1)
                # Find vr' and vc'
                if u.startswith('Null'):
                    # births
                    rateNode = splitVrVc(vr, newCompartments)
                else:
                    # normal rates
                    rateNode = addIList(vr, 1)
                contactNode = 'Null_m'

                newFlow['from'] = uPrime
                newFlow['to'] = vPrime
                newFlow['rate'] = rateNode
                newFlow['contact'] = contactNode
                newFlow['parameter'] = flow['parameter']

                newModel['flows'][flowName].append(newFlow.copy())

                if addI(u, 0) in newCompartments:
                    uPrime = addI(u, 0)
                    if addI(v, 0) in newCompartments:
                        vPrime = addI(v, 0)
                    else:
                        vPrime = addI(v, 1)
                    rateNode = addIList(vr, 0)
                    contactNode = 'Null_m'

                    newFlow['from'] = uPrime
                    newFlow['to'] = vPrime
                    newFlow['rate'] = rateNode
                    newFlow['contact'] = contactNode
                    newFlow['parameter'] = flow['parameter']

                    newModel['flows'][flowName].append(newFlow.copy())
            ### BATCHES ###
            if getFlowType(flow) == 'batch':

                uPrime = addI(u, 1)
                vPrime = addI(v, 1)
                rateNode = 'Null_n'
                contactNode = 'Null_m'

                newFlow['from'] = uPrime
                newFlow['to'] = vPrime
                newFlow['rate'] = rateNode
                newFlow['contact'] = contactNode
                newFlow['parameter'] = flow['parameter']

                newModel['flows'][flowName].append(newFlow.copy())
            ### CONTACTS ###
            if getFlowType(flow) == 'contact':
                # first layer
                uPrime = addI(u, 1)
                vPrime = addI(v, 1)
                rateNode = addIList(vr, 1)
                contactNode = splitVrVc(vc, newCompartments)

                newFlow['from'] = uPrime
                newFlow['to'] = vPrime
                newFlow['rate'] = rateNode
                newFlow['contact'] = contactNode
                newFlow['parameter'] = flow['parameter']

                newModel['flows'][flowName].append(newFlow.copy())

                if addI(u, 0) in newCompartments:
                    # isolated layer
                    uPrime = addI(u, 0)
                    vPrime = addI(v, 0)
                    rateNode = addIList(vr, 0)
                    contactNode = splitVrVc(vc, newCompartments)

                    newFlow['from'] = uPrime
                    newFlow['to'] = vPrime
                    newFlow['rate'] = rateNode
                    newFlow['contact'] = contactNode
                    newFlow['parameter'] = flow['parameter']

                    newModel['flows'][flowName].append(newFlow.copy())

                # COMPUTE RT
                # RT NODES
                compartName = f"Rt({flow['from']},{flow['to']})"
                newModel["compartments"][compartName] = 0

                uPrime = 'Null_n'
                vPrime = compartName
                rateNode = splitVrVc(vr, newCompartments)
                contactNode = addIList(vc, 0)

                newFlow['from'] = uPrime
                newFlow['to'] = vPrime
                newFlow['rate'] = rateNode
                newFlow['contact'] = contactNode
                newFlow['parameter'] = flow['parameter']

                newModel['flows'][flowName].append(newFlow.copy())

    # Add Null compartments
    for compartment in compartments:
        if compartment.startswith('Null'):
            newModel["compartments"][compartment] \
                = model["compartments"][compartment]

    if write:
        writeModel(newModel, overWrite, printText)
    storeFunctions(newModel)
    return newModel


def printModel(dictionary: dict) -> None:
    """
    Prints given dictionary in a formatted manner.

    Inputs:
        dictionary: dict
            Given model to print.

    Ouputs:
        None.
    """
    print(json.dumps(dictionary, sort_keys=True, indent=2))


def roundDict(dictionary: dict, i: int) -> dict:
    """
    Returns the same dictionary with rounded values.

    Inputs:
        dictionary: dict
            Given model to round.
        i: int
            Number of digits to round to.

    Outputs:
        roundedDict: dict
            Rounded dictionary.
    """
    roundedDict = {key: round(dictionary[key], i) for key in dictionary}

    if i < 1:
        roundedDict = {key: int(roundedDict[key]) for key in roundedDict}

    return roundedDict


def computeRt(modelName: str, t_span_rt: tuple, sub_rt: float = 1,
              t_span_sim: tuple = (0, 100), sub_sim: int = 5,
              scaledInfs=False,
              verification: bool = True, write: bool = True,
              overWrite: bool = False, whereToAdd: str = 'to',
              printText=True, printInit=False, printWarnings=True,
              r0=False, scaleMethod: str = 'Total',
              printR0: bool = False, useTqdm: bool = True) -> tuple:
    """
    Returns a dictionary with Rt values, as well as models and solutions.

    Inputs:
        modelName: str
            Name of model to simulate.
        t_span_rt: tuple
            Time range for which we require rt values.
        sub_rt: float
            Time precision with which to compute rt.
        t_span_sim: tuple
            Time range used for simulations.
        sub_sim: int
            Time precision for simulations.
        scaledInfs: bool
            Whether or not to rescale infected when initializing.
        verification: bool
            Whether or not to confirm that modified and original have same dynamic.
        write: bool
            Whether or not to write json files.
        overWrite: bool
            Whether or not to overwrite json files.
        whereToAdd: str
            Where to add new infected individuals.
        printText: bool
            Whether or not to print debug text.
        printInit: bool
            Whether or not to print initialization text.
        printWarnings: bool
            Whether or not to print warning text.
        r0: bool
            Whether or not we are computing R0 (false if Rt).
        scaleMethod: str
            Total: scale all rt values together, PerVariant: variant-wise.
        printR0: bool
            Whether or not to print R0 values.
        useTqdm: bool
            Whether or not to show progress bar on rt computation.

    Outputs:
        modelOld: dict
            Old (original) model.
        newModel: dict
            Modified model.
        solutionOld: np.ndarray
            Integration solution (only keep time values that fit with t_span_rt)
        t_spanOld: np.ndarray
            Integration time points (only keep time values that fit with t_span_rt)
        values: dict
            Dictionary containing all rt values of interest.
        toKeep: list
            List of values to keep in solution.
    """

    if printText:
        if r0:
            print('\nComputation of R0')
        else:
            print('\nComputation of Rt')

    if printWarnings:
        if sub_rt > sub_sim:
            sub_sim = sub_rt

    modelOld = loadModel(modelName, printText=printText)
    solutionOld, t_spanOld = solve(modelOld, (0, t_span_rt[1]), 100)
    oldCompartments = getCompartments(modelOld)

    newModel = mod(modelOld, printText,
                   write=write, overWrite=overWrite)
    solution, _ = solve(newModel, (0, t_span_rt[1]), 100)
    compartments = getCompartments(newModel)

    # Vérification!
    if verification:
        allGood = True
        problems = []
        for comp in getPopNodes(modelOld):
            if comp[:4] != 'Null':
                array1 = solutionOld[:, oldCompartments.index(comp)]
                try:
                    if useTorch:
                        array2 = torch.sum(
                            [solution[:, compartments.index(addI(comp, i))]
                             for i in range(2)], axis=0)
                    else:
                        array2 = np.sum(
                            [solution[:, compartments.index(addI(comp, i))]
                             for i in range(2)], axis=0)
                except:
                    array2 = solution[:, compartments.index(addI(comp, 1))]
            else:
                array1 = solutionOld[:, oldCompartments.index(comp)]
                if useTorch:
                    array2 = torch.sum(
                        [solution[:, compartments.index(x)]
                         for x in getOtherNodes(newModel)], axis=0)
                else:
                    array2 = np.sum(
                        [solution[:, compartments.index(x)]
                         for x in getOtherNodes(newModel)], axis=0)

            # print(f"{comp + ':':<{length}}", np.allclose(array1, array2))
            a1 = array1 + .1
            a2 = array2 + .1
            if useTorch:
                condition = not torch.allclose(a1, a2) \
                    or not torch.allclose(a1, a2)
            else:
                condition = not np.allclose(a1, a2) \
                    or not np.allclose(a1, a2)
            if condition:
                allGood = False
                problems.append(comp)

        if not allGood and printWarnings:
            print('Il semble que les modèles aient des résultats différents.')
            print('On continue l\'expérience quand même, à vérifier.')
            print(f'Problèmes: {problems}.')
        else:
            if printText:
                print('Véfication faite, les deux modèles sont identiques.')

    values = {}
    # No need for progress bar if only computing R0
    iterations = np.arange(t_span_rt[0],
                           t_span_rt[1] + .5 / sub_rt,
                           1 / sub_rt)
    iterator = tqdm(iterations) if len(iterations) > 1 \
        and useTqdm else iterations
    for t in iterator:

        values[t] = {}
        pointIndex = find_nearest(t_spanOld, t)
        pointTime = t_spanOld[pointIndex]
        init = {key: solutionOld[pointIndex, i]
                for i, key in enumerate(oldCompartments)}

        initialize(newModel, init, pointTime, modelOld, r0=r0,
                   printText=printText, whereToAdd=whereToAdd,
                   scaled=scaledInfs)

        solutionTemp, _ = solve(newModel, t_span_sim, sub_sim, t0=t)

        initialCond = solutionOld[pointIndex]
        initialCond = {comp: initialCond[i]
                       for i, comp in enumerate(getCompartments(modelOld))}
        initialCond = infs(modelOld, initialCond, t, whereToAdd='to')
        initialCond = {comp: initialCond[comp]
                       for comp in initialCond if initialCond[comp] > 0}

        if scaleMethod == 'Total':
            if scaledInfs:
                denom = 1
            else:
                denom = sum(initialCond[x] for x in initialCond)

        RtNodes = getRtNodes(newModel)

        if printR0:
            length = max(len(x) for x in RtNodes)

            if t == t_span_rt[0]:
                print(f"Time  {'Node':<{length + 6}}Value     Divide    Rt")

        for x in RtNodes:

            if scaleMethod == 'PerVariant':
                compartment = x.split(',')[1].split(')')[0]
                if scaledInfs:
                    denom = 1 if compartment in initialCond else 0
                else:
                    denom = initialCond[compartment] if compartment in initialCond else 0

            value = solutionTemp[-1, getCompartments(newModel).index(x)]
            newValue = value / (denom if denom != 0 else 1)

            if printR0:
                # if t == t_span_rt[0]:
                print(f"{t:4.0f}, {x + ', ':<{length + 2}}{value:9.1f}, "
                      + f"{denom:9.1f}, {newValue:4.2f}")
            values[t][x] = newValue
        # print(f'{sum(values[t_spanOld[i]]):.2f} ', end='')

    if printText:
        if r0:
            print(f'R0 computation done: R0 = {values[0]}\n')
        else:
            print('Rt computation done\n')

    toKeep = np.where(np.logical_and(t_span_rt[0] <= t_spanOld,
                                     t_spanOld <= t_span_rt[1]))[0]
    return modelOld, newModel, solutionOld[toKeep], t_spanOld[toKeep], values, toKeep


def computeR0(modelName: str, t_span_sim: tuple = (0, 100),
              sub_sim: float = 100, scaledInfs=False,
              verification: bool = True,
              write: bool = False,
              overWrite: bool = False, whereToAdd: str = 'to',
              printText=True, printInit: bool = True,
              printWarnings: bool = True, scaleMethod: str = 'Total',
              printR0: bool = False) -> tuple:
    """
    Returns a dictionary with R0 values, as well as models and initial conditions.

    Inputs:
        modelName: str
            Name of model to simulate.
        t_span_sim: tuple
            Time range used for simulations.
        sub_sim: int
            Time precision for simulations.
        scaledInfs: bool
            Whether or not to rescale infected when initializing.
        verification: bool
            Whether or not to confirm that modified and original have same dynamic.
        write: bool
            Whether or not to write json files.
        overWrite: bool
            Whether or not to overwrite json files.
        whereToAdd: str
            Where to add new infected individuals.
        printText: bool
            Whether or not to print debug text.
        printInit: bool
            Whether or not to print initialization text.
        printWarnings: bool
            Whether or not to print warning text.
        scaleMethod: str
            Total: scale all rt values together, PerVariant: variant-wise.
        printR0: bool
            Whether or not to print R0 values.

    Outputs:
        modelOld: dict
            Old (original) model.
        newModel: dict
            Modified model.
        initialConds: np.ndarray
            Values of compartments at time 0.
        values: dict
            Dictionary containing all R0 values of interest.
    """

    modelOld, newModel, solutionOld, _, values, _ = \
        computeRt(modelName, (0, 0), 1, t_span_sim,
                  sub_sim, scaledInfs=scaledInfs, verification=verification,
                  write=write, overWrite=overWrite, whereToAdd=whereToAdd,
                  printInit=printInit, r0=True,
                  printWarnings=printWarnings, printText=printText,
                  scaleMethod=scaleMethod, printR0=printR0)

    initialConds = solutionOld[0]
    return modelOld, newModel, initialConds, values[0]


def evaluateCurve(curve: np.ndarray, idx: int or float) -> any:
    """
    Evaluates curve at index or pair of index (average of both indices).

    Inputs:
        curve: np.ndarray
            Curve to evaluate.
        idx: int or float
            Index to evaluate at.

    Outputs:
        output: any
            Output of array.
    """

    try:
        output = curve[idx]
    except:
        p = idx - int(idx)
        output = (1 - p) * curve[int(idx)] \
            + p * curve[int(idx + 1)]

    return output


def compare(modelName: str,
            t_span_rt: tuple, sub_rt: float = 1,
            R0: float = 0,
            t_span_sim: tuple = (0, 100), sub_sim: float = 100,
            verification: bool = True, write: bool = True,
            overWrite: bool = False, whereToAdd: str = 'to',
            printText=False, printInit=False,
            plotANA: bool = True,
            title: str = None,
            scaleMethod: str = 'Total',
            plotIndividual: bool = True,
            plotInfected: bool = True,
            printR0: bool = False,
            scaledInfectedPlot=False,
            supressGraph: bool = False,
            useTqdm: bool = True,
            legendLoc: str = 'best',
            legendRtCurve: str = None,
            whereToPlot: tuple = None,
            useLog: bool = True,
            plotStyle: str = None,
            forceColors: bool = False,
            drawVertical: bool = True,
            saveGraph: bool = True,
            graphName: str = None,
            addToLegends: str = '',
            scale: float = 1,
            plotFrom: float = 0,
            colors: tuple = ('C1', PLUM)) -> None:
    """
    Runs through all steps and produces a graph (call plt.plot() to show after).

    Inputs:
        modelName: str
            Name of model to simulate.
        t_span_rt: tuple
            Time range for which we require rt values.
        sub_rt: float
            Time precision with which to compute rt.
        R0: float
            Analytical value of R0 to compare to.
        t_span_sim: tuple
            Time range used for simulations.
        sub_sim: int
            Time precision for simulations.
        verification: bool
            Whether or not to confirm that modified and original have same dynamic.
        write: bool
            Whether or not to write json files.
        overWrite: bool
            Whether or not to overwrite json files.
        whereToAdd: str
            Where to add new infected individuals.
        printText: bool
            Whether or not to print debug text.
        printInit: bool
            Whether or not to print initialization text.
        plotANA: bool
            Whether or not to plot analytical Rt.
        plotANA_v2: bool
            Whether or not to plot the 2nd version of analytical Rt.
        infected: list
            List of infected compartments (integers).
        scaleMethod: str
            Total: scale all rt values together, PerVariant: variant-wise.
        plotIndividual: bool
            Whether or not to plot individual Rt lines (e.g. for variants).
        plotBound: bool
            Whether or not to plot suspected lower bound.
        printR0: bool
            Whether or not to print R0 values.
        scaledInfectedPlot: bool
            Whether or not to plot incident cases scaled (better visualization).
        supressGraph: bool
            Whether or not to suppress graph at the end of loop.
        useTqdm: bool
            Whether or not to show progress bar on rt computation.

    Outputs:
        rt_times: np.ndarray
            Times for which we compute Rt.
        rtCurves: dict
            Dictionary containing all rtCurves of interest.
        infsNotScaled
    """

    WIDTH = 1
    HEIGHT = WIDTH * 3/4
    DASH = (3, 3)

    if whereToPlot is None:
        fig, ax1 = plt.subplots(figsize=(6.4*scale, 4.8*scale))
        # In article, scale is 5/8
    else:
        fig, ax1, _ = whereToPlot
    if plotInfected and not scaledInfectedPlot:
        if whereToPlot is None:
            ax2 = ax1.twinx()
        else:
            _, _, ax2 = whereToPlot
        if useLog:
            ax1.axhline(y=0, linestyle=':', color='gray',
                        linewidth=HEIGHT, dashes=DASH)
            ax2.set_yscale('log')
        else:
            ax2.axhline(y=0, linestyle=':', color='gray',
                        linewidth=HEIGHT, dashes=DASH)
        ax2.set_ylabel('Nb. d\'individus')
    ax1.axhline(y=1, linestyle=':', color='white', gapcolor='gray',
                linewidth=HEIGHT, dashes=DASH)
    ax1.set_ylabel('Nb. de reproduction')
    ax1.set_xlabel('Temps (jours)')

    rtCurves = {}

    model, _, solution, t_span, values, _ = computeRt(
        modelName, t_span_rt, sub_rt,
        t_span_sim=t_span_sim, sub_sim=sub_sim,
        verification=verification, whereToAdd=whereToAdd,
        scaledInfs=False, write=write, overWrite=overWrite,
        printText=printText, printInit=printInit,
        printWarnings=True, scaleMethod=scaleMethod,
        printR0=printR0, useTqdm=useTqdm)

    toKeep_t_span = np.where(t_span >= plotFrom - 1.5 / sub_sim)[0]

    N = np.array([getPopulation(model, x)['Sum']
                  for x in solution])

    infsScaled = infCurveScaled(model, solution, t_span)
    infsNotScaled = infCurve(model, solution, t_span)

    if plotInfected:
        ax2.plot(t_span[toKeep_t_span], infsNotScaled[toKeep_t_span],
                 label='$\\nu(t)$' + addToLegends,
                 linewidth=None if plotStyle is None else WIDTH,
                 ls='--' if plotStyle is None else plotStyle,
                 c=PLUM if forceColors else colors[1])

    compartments = getCompartments(model)
    susceptibles = [compartments.index(x) for x in getSusceptibleNodes(model)]

    if useTorch:
        susceptiblesDivPop = torch.sum(solution[:, susceptibles], axis=1) / N
        # infectedDivPop = torch.sum(solution[:, infected], axis=1) / N
    else:
        susceptiblesDivPop = np.sum(solution[:, susceptibles], axis=1) / N
        # infectedDivPop = np.sum(solution[:, infected], axis=1) / N

    # R0 needs to be table, tablewise multiplication
    # or simply a number for normal multiplication
    rt_ANA = R0 * susceptiblesDivPop

    rt_times = np.array([key for key in values])
    toKeep_rt_times = np.where(rt_times >= plotFrom - 1.5 / sub_rt)

    rt = np.zeros_like(rt_times, dtype='float64')
    for i, rtNode in enumerate(getRtNodes(mod(model, False))):
        rt_rtNode = np.array([values[key][rtNode] for key in values])
        rtCurves[rtNode] = rt_rtNode
        rt += rt_rtNode
    rtCurves['Sum'] = rt

    # Rt simulated
    if doesIntersect(rt, 1):
        idx_rt = find_intersections(rt, 1)
        xTimeRt = []
        for idx in idx_rt:
            xTimeRt.append(evaluateCurve(rt_times, idx))
        xTimeRt = np.array(xTimeRt)

        peaks = argrelextrema(infsScaled, np.greater, order=5)
        valleys = argrelextrema(infsScaled, np.less, order=5)
        extrema = np.union1d(peaks, valleys)
        xTimeInfs = []
        for idx in extrema:
            if abs((infsScaled[idx + 1] - infsScaled[idx - 1]) / infsScaled[idx]) < .1:
                xTimeInfs.append(evaluateCurve(t_span, idx))
        xTimeInfs = np.array(xTimeInfs)

        if printText:
            print(f'Infected extrema at {[np.round(x, 2) for x in xTimeInfs]}')
            print(f'Rt = 1 at {[np.round(x, 2) for x in xTimeRt]}')
            if xTimeRt.shape == xTimeInfs.shape:
                print(
                    f'Time difference: {[np.round(x, 2) for x in np.abs(xTimeInfs - xTimeRt)]}')

        if plotInfected and drawVertical:
            for point in xTimeInfs:
                try:
                    ax1.axvline(x=point, linestyle=':', color=PLUM if forceColors else colors[1],
                                linewidth=WIDTH if plotStyle is None else HEIGHT,
                                dashes=DASH if plotStyle is None else (1, 1))
                except Exception as e:
                    print(e)
            for point in xTimeRt:
                try:
                    ax1.axvline(x=point, linestyle=':',
                                color=(0, 0, 0, 0),
                                gapcolor='C1' if forceColors else colors[0],
                                linewidth=WIDTH if plotStyle is None else HEIGHT,
                                dashes=DASH if plotStyle is None else (1, 1))
                except Exception as e:
                    print(e)
    elif printText:
        print('Time difference is not relevant, '
              + 'no intersection between rt and 1.')

    # RT CURVES
    if plotANA:
        ax1.plot(t_span[toKeep_t_span], rt_ANA[toKeep_t_span],
                 label='$\\mathcal{R}_{t}^{ana}$' + addToLegends, ls=plotStyle,
                 c='tab:blue' if forceColors else None)
    if len(getRtNodes(mod(model, False))) > 1 and plotIndividual:
        for i, rtNode in enumerate(getRtNodes(mod(model, False))):
            rt_rtNode = np.array([values[key][rtNode] for key in values])
            rtCurves[rtNode] = rt_rtNode
            rt_rtNode[toKeep_rt_times]
            nodeLegend = rtNode.replace('Rt', '\\mathcal{R}_t')
            ax1.plot(rt_times[toKeep_rt_times], rt_rtNode[toKeep_rt_times],
                     label=f"${nodeLegend}$", ls=':', color=f'C{i+2}',
                     linewidth=HEIGHT, dashes=DASH)
    ax1.plot(rt_times[toKeep_rt_times], rt[toKeep_rt_times],
             label=('$\\mathcal{R}_{t}^{sim}$' if legendRtCurve is None
                    else legendRtCurve) + addToLegends,
             linewidth=None if plotStyle is None else WIDTH,
             linestyle=plotStyle,
             c='C1' if forceColors else colors[0])

    if title is None:
        ax1.set_title(modelName)
    else:
        ax1.set_title(title)

    if plotInfected and not scaledInfectedPlot:
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines + lines2, labels + labels2,
                   loc=legendLoc, framealpha=.9)

        # ax1.set_ylim(bottom=0)
        ax2.set_ylim(bottom=1)
        if ax2.get_yscale() == 'linear':
            ax2.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    else:
        ax2.legend(framealpha=.9)

    yintMax = math.ceil(max(np.max(rt), np.max(rt_ANA)) + 1)
    yint = range(yintMax)
    ax1.set_yticks(yint)

    if supressGraph:
        plt.close()

    if saveGraph:
        if graphName is None:
            fig.savefig(f'graphs/{modelName}.png', bbox_inches='tight')
        else:
            fig.savefig(f'graphs/{graphName}.png', bbox_inches='tight')

    print('Rt at start, ana and sim:', rt_ANA[0], rt[0])
    print('Rt at end, ana and sim:', rt_ANA[-1], rt[-1])

    return rt_times, rtCurves, infsNotScaled


def infs(model: dict, y0: dict, t: float, whereToAdd: str = 'to') -> dict:
    """
    Returns incidence value for y0.

    Inputs:
        model: dict
            Model of interest.
        y0: dict
            Value for each compartment of model.
        t: float
            Time at which we compute incidences.
        whereToAdd: str
            Where to add new infections.

    Outputs:
        newInfections: dict
            Incidences created in each compartment of the model.
    """

    weWant = getCompartments(model)
    if sorted(list(y0.keys())) != sorted(weWant):
        raise Exception("Initialization vector doesn't have right entries.\n"
                        + f"Entries wanted:   {weWant}.\n"
                        + f"Entries obtained: {list(y0.keys())}.")

    newInfections = {key: 0 for key in weWant}
    flows = model['flows']
    N = sum(y0[x] for x in getPopNodes(model))
    for _, flowType in enumerate(flows):
        for _, flow in enumerate(flows[flowType]):
            if getFlowType(flow) == 'contact':
                v_r = flow['rate'].split('+')
                v_c = flow['contact'].split('+')

                rateImpact = sum(y0[x]
                                 for x in v_r if not x.startswith('Null'))
                contactImpact = sum(y0[x]
                                    for x in v_c if not x.startswith('Null'))
                # Normally v_r and v_c should not be null. We can use both directly.
                param = getCoefForFlow(flow, t)
                contactsFlow = param * rateImpact * contactImpact / N
                node = flow[whereToAdd]
                if node.startswith('Null'):
                    node = node + '_m'
                newInfections[node] += contactsFlow

    return newInfections


def infsScaled(model: dict, y0: dict, t: float, whereToAdd: str = 'to') -> dict:
    """
    Returns scaled incidence value for y0.

    Inputs:
        model: dict
            Model of interest.
        y0: dict
            Value for each compartment of model.
        t: float
            Time at which we compute incidences.
        whereToAdd: str
            Where to add new infections.

    Outputs:
        newInfections: dict
            Scaled incidences created in each compartment of the model.
    """

    infections = infs(model, y0, t, whereToAdd)
    weWant = getCompartments(model)

    sumInfections = sum(infections[node] for node in weWant)
    denom = sumInfections if sumInfections != 0 else 1

    scaledInfs = {key: infections[key] / denom for key in infections}

    return scaledInfs


def totInfs(model: dict, state: np.ndarray, t: float) -> np.ndarray:
    """
    Returns total incidence for a state.

        model: dict
            Model of interest.
        state: np.ndarray
            Value for each compartment of model.
        t: float
            Time at which we compute incidences.

    Outputs:
        infectious: float
            Number of incidences created in the model at time t.
    """

    if len(state.shape) > 1:
        raise Exception(
            f'Function can only be used on single state, not solution.')
    weWant = getCompartments(model)
    y0 = {comp: state[i] for i, comp in enumerate(weWant)}
    infections = infs(model, y0, t, whereToAdd='to')

    infectious = sum(infections[comp] for comp in weWant)
    return infectious


def infCurve(model: dict, solution: np.ndarray, t_span: np.ndarray) -> np.ndarray:
    """
    Returns curve of incidence for given solution.

    Inputs:
        model: dict
            Model of interest.
        solution: np.ndarray
            Solution given by solve function.
        t_span: np.ndarray
            Time frame for solution.

    Outputs:
        curve: np.ndarray
            Curve of newly infected at each time.
    """

    if useTorch:
        curve = torch.FloatTensor([totInfs(model, x, t_span[i])
                                   for i, x in enumerate(solution)])
    else:
        curve = np.array([totInfs(model, x, t_span[i])
                          for i, x in enumerate(solution)])
    return curve


def infCurveScaled(model: dict, solution: np.ndarray, t_span: np.ndarray) -> np.ndarray:
    """
    Returns scaled curve of incidence for given solution.

    Inputs:
        model: dict
            Model of interest.
        solution: np.ndarray
            Solution given by solve function.
        t_span: np.ndarray
            Time frame for solution.

    Outputs:
        curve: np.ndarray
            Scaled curve of newly infected at each time.
    """

    curve = infCurve(model, solution, t_span)
    if useTorch:
        curve = curve / torch.max(curve)
    else:
        curve = curve / np.max(curve)
    return curve


def writeModel(model: dict, overWrite: bool = True, printText: bool = True) -> None:
    """
    Write given model to file.

    Inputs:
        newModel: dict
            Model to write to file.
        overWrite: bool
            Whether or not to overwrite existing file.
        printText: bool
            Whether or not to print debug text.

    Outputs:
        None.
    """
    modelName = model['name']
    newFileName = modelName + '.json'
    if printText:
        print(f'Writing model to file models/{newFileName}.')
    if not os.path.isfile(f'models/{newFileName}'):
        # File doesn't exist
        try:
            with open(f'models/{newFileName}', 'w') as file:
                json.dump(model, file, indent=4)
            if printText:
                print('Model written.')
        except:
            if printText:
                print('Problem when writing file.')
    else:
        # File exists already
        if printText:
            print('File name already exists.')
        if overWrite:
            if printText:
                print('Overwriting file.')
            # os.remove(f'models/{newFileName}')
            try:
                with open(f'models/{newFileName}', 'w') as file:
                    json.dump(model, file, indent=4)
            except:
                if printText:
                    print('Problem when writing file.')


def find_nearest(array: np.ndarray, value: float) -> int:
    """
    Find index of array for which array value is closest to given value.

    Inputs:
        array: np.ndarray
            Array of interest.
        value: float
            Value to find.

    Outputs:
        idx: int
            Position of closest element.
    """
    if useTorch:
        array = torch.FloatTensor(array)
        idx = torch.argmin((torch.abs(array - value)))
    else:
        array = np.array(array)
        idx = np.argmin((np.abs(array - value)))
    return idx


def find_intersections(array: np.ndarray, value: float, eps=10**-5) -> list:
    """
    Finds all intersections between curve and value.

    Inputs:
        array: np.ndarray
            Array of interest.
        value: float
            Value to find.
        eps: float
            Permitted error.

    Outputs:
        newWhere: list
            List of floats.
    """

    newCurve = array - value
    where = np.where(abs(newCurve) > eps)[0]

    newWhere = []

    newCurveEps = newCurve[where]

    for i in range(len(newCurveEps) - 1):
        product = newCurveEps[i] * newCurveEps[i + 1]

        if product < 0:
            if where[i + 1] - where[i] > 1:
                newWhere.append((where[i] + where[i + 1]) / 2)
            else:
                num = value - array[where[i]]
                denom = array[where[i + 1]] - array[where[i]]
                newWhere.append(num / denom + where[i])

    return newWhere


def find_intersections_curves(curve1: np.ndarray, curve2: np.ndarray, eps=10**-5) -> list:
    """
    Finds all intersections between curve and value.

    Inputs:
        curve1: np.ndarray
            First array to test.
        curve2: np.ndarray
            Second array to test.
        eps: float
            Permitted error.

    Outputs:
        newWhere: list
            List of floats.
    """

    newCurve = curve1 - curve2
    where = np.where(abs(newCurve) > eps)[0]

    newWhere = []

    newCurveEps = newCurve[where]

    for i in range(len(newCurveEps) - 1):
        product = newCurveEps[i] * newCurveEps[i + 1]

        if product < 0:
            if where[i + 1] - where[i] > 1:
                newWhere.append((where[i] + where[i + 1]) / 2)
            else:
                num = - newCurve[where[i]]
                denom = newCurve[where[i + 1]] - newCurve[where[i]]
                newWhere.append(num / denom + where[i])

    return newWhere


def doesIntersect(curve: np.ndarray, value: int, eps=10**-5):
    """
    Checks if curve intersects y = value.

    Inputs:
        curve: np.ndarray
            Array of interest.
        value: float
            Value to find.
        eps: float
            Permitted error.

    Outputs:
        doesInter: bool
            Whether or not the curve intersects the value given.
    """
    doesInter = len(find_intersections(curve, value, eps)) > 0
    return doesInter


def createLaTeX(model: dict, layerDistance: float = 1,
                nodeDistance: float = 2, varDistance: float = .25,
                nullDistance: float = .8, baseAngle: int = 10,
                contactPositions: tuple = ("2/5", "3/5"), scale=1) -> None:
    """
    Produces tikzfigure for a model. Places automatically in file.

    Needs some definitions in preamble:

    ```latex
    usepackage[usenames,dvipsnames]{xcolor}
    usepackage{tikz}
    usetikzlibrary{calc, positioning, arrows.meta, shapes.geometric}

    tikzset{Square/.style={draw=black, rectangle, rounded corners=0pt, align=center, minimum height=1cm, minimum width=1cm}}
    tikzset{Text/.style={rectangle, rounded corners=0pt, inner sep=0, outer sep=0, draw=none, sloped}}
    tikzset{Empty/.style={rectangle, inner sep=0, outer sep=0, draw=none}}

    tikzset{Arrow/.style={-Latex, line width=1pt}}
    tikzset{Dashed/.style={Latex-, dashed, line width=1pt}}
    tikzset{Dotted/.style={Latex-, dotted, line width=1pt}}
    ```

    Inputs:
        model: dict
            Model of interest.
        layerDistance: float
            Vertical distance (in cm) for modified model layer.
        nodeDistance: float
            Horizontal distance (in cm) for nodes.
        varDistance: float
            Vertical distance (in cm) for variant nodes.
        nullDistance: float
            Diagonal distance (in cm) for births and deaths.
        baseAngle: float
            Base angle for all arrows.
        contactPositions: tuple
            Position for rate and contact liaison on arrow.
        scale: float
            Scale of the figure.

    Outputs:
        None.
    """
    # variables préliminaires
    tab = ' ' * 4
    modelName = model['name']
    compartments = model['compartments']
    flows = model['flows']
    modified = False
    joint = {}
    colors = {
        'rate': 'Green',
        'contact': 'Plum',
        'batch': 'Cyan'
    }

    layerDistance = scale * layerDistance
    nodeDistance = scale * nodeDistance
    varDistance = scale * varDistance
    nullDistance = scale * nullDistance

    for x in compartments:
        # Is the model to graph a modified version?
        if x.endswith(('^0, ^1')) or x.startswith(('Rt')):
            modified = True
            break

    for x in compartments:
        # Join all variant informations together
        if '_' in x and not x.startswith(('Null', 'Rt')):
            if modified:
                i = int(x.split('^')[1])
                compBase = x.split('_')[0]
                joint[x] = [x for x in compartments
                            if x.startswith(compBase + '_') and x.endswith(f'^{i}')] \
                    + [addI(compBase, i)]

            else:
                compBase = x.split('_')[0]
                joint[x] = [
                    x for x in compartments if x.startswith(compBase + '_')] \
                    + [compBase]

    # String containing LaTeX code
    LaTeX = f"\\begin{{figure}}[H]\n{tab}\\centering\n{tab}" \
        + f"\\begin{{tikzpicture}}[scale={scale}, every node/.style=" + "{scale=" \
        + f"{scale}" + "}]\n"

    # Book-keeping
    layer0 = []
    layer1 = []
    others = []
    bases0 = []
    bases1 = []
    for x in compartments:
        # Create edges
        if (x.endswith('^1') or (not modified and
                                 not x.startswith(('Null', 'Rt')))) \
                and x not in layer1:
            # Layer 0 (or normal layer in non-modified)
            if len(layer1) != 0:
                # Où placer le noeud par rapport à ceux qui existent
                if layer1[-1] not in joint:
                    where = f"[right={nodeDistance}cm of {layer1[-1]}] "
                else:
                    where = f"[right={nodeDistance}cm of {bases1[-1]}] "
            else:
                where = " "

            if x not in joint:
                # Si pas un variant, on le place simplement
                LaTeX += f"{tab * 2}\\node [Square] ({x}) " \
                    + where + f"{{${x}$}};\n"

                layer1.append(x)
            else:
                # Si on a variant, il faut placer noeud du centre
                # et mettre les variants autour
                base = joint[x][-1]
                toPlace = joint[x][:-1]
                positions = [2 * i - (len(toPlace) - 1)
                             for i, _ in enumerate(toPlace)]

                LaTeX += f"{tab * 2}\\node [Empty] ({base}) " \
                    + where + f"{{}};\n"

                for i, node in enumerate(toPlace):
                    if positions[i] < 0:
                        where2 = f"[above={- positions[i]*varDistance}cm of {base}] "
                    elif positions[i] > 0:
                        where2 = f"[below={positions[i]*varDistance}cm of {base}] "
                    else:
                        where2 = f"[right={positions[i]*varDistance}cm of {base}] "
                    LaTeX += f"{tab * 2}\\node [Square] ({node}) " \
                        + where2 + f"{{${node}$}};\n"

                    layer1.append(node)
                bases1.append(base)
        elif x.endswith('^0') and x not in layer0:
            # Layer 0, this should only be for a few compartments
            # On utilise layer 1 pour construire
            equivalent1 = addI(removeI(x), 1)
            where = f"[above={layerDistance}cm of {equivalent1}] "

            if x not in joint:
                LaTeX += f"{tab * 2}\\node [Square] ({x}) " + \
                    where + f"{{${x}$}};\n"

                layer0.append(x)
            else:
                base = joint[x][-1]
                toPlace = joint[x][:-1]
                positions = [2 * i - (len(toPlace) - 1)
                             for i, _ in enumerate(toPlace)]

                LaTeX += f"{tab * 2}\\node [Empty] ({base}) " \
                    + where + f"{{}};\n"

                for i, node in enumerate(toPlace):
                    if positions[i] < 0:
                        where2 = f"[above={- positions[i]*varDistance}cm of {base}] "
                    elif positions[i] > 0:
                        where2 = f"[below={positions[i]*varDistance}cm of {base}] "
                    else:
                        where2 = f"[right={positions[i]*varDistance}cm of {base}] "
                    LaTeX += f"{tab * 2}\\node [Square] ({node}) " \
                        + where2 + f"{{${node}$}};\n"

                    layer0.append(node)
                bases0.append(base)

        elif x not in layer0 and x not in layer1 and not x.startswith('Null'):
            # Ceux qui ne sont pas dans une layer, donc Rt
            # (Nulls sont placés ici aussi)

            # Variables pour nom (pas de parenthèses dans Tikz) et texte à écrire
            nameNoProblem = x.replace(
                '(', '').replace(')', '').replace(',', '')
            if x.startswith('Rt'):
                textNoProblem = '$\\\\$('.join(x.split('('))
                textNoProblem = '_'.join([textNoProblem[0], textNoProblem[1:]])
            else:
                textNoProblem = x

            # Place to the left of layer 1
            if len(others) == 0:
                if layer1[0] not in joint:
                    reference = layer1[0]
                else:
                    reference = bases1[0]
                pos = 'left'
                dist = nodeDistance
            else:
                reference = others[0]
                pos = 'above'
                dist = nodeDistance
            LaTeX += f"{tab * 2}\\node [Square] ({nameNoProblem}) " \
                + f"[{pos}={dist}cm of {reference}] " + \
                f"{{${textNoProblem}$}};\n"

            others.append(nameNoProblem)

    layer0 = removeDuplicates(layer0)
    layer1 = removeDuplicates(layer1)
    others = removeDuplicates(others)

    LaTeX += '\n'
    for flowType in flows:
        for flow in flows[flowType]:
            # On veut des noeuds vides pour considérer les entrées et sorties
            # Puisque les noeuds prennent de la place on les crée seulement lorsque nécessaire

            u, v, v_r, v_c = flow['from'], flow['to'], flow['rate'], flow['contact']
            if u.startswith('Null'):
                nameNoProblem = v.replace(
                    '(', '').replace(')', '').replace(',', '')
                LaTeX += f"{tab * 2}\\node [Empty] (Nulln_{nameNoProblem}) " \
                    + f"[above left={nullDistance}cm of {nameNoProblem}] {{}};\n"
            if v.startswith('Null'):
                nameNoProblem = u.replace(
                    '(', '').replace(')', '').replace(',', '')
                LaTeX += f"{tab * 2}\\node [Empty] (Nullm_{nameNoProblem}) " \
                    + f"[below right={nullDistance}cm of {nameNoProblem}] {{}};\n"

    # L'ensemble des flèches
    Arrow = []  # E(G)
    Dotted = []  # s(E(G))
    Dashed = []  # t(E(G))
    for flowType in flows:
        for flow in flows[flowType]:
            # Couleur du flot (voir document)
            color = colors[getFlowType(flow)]
            u, v, v_r, v_c = flow['from'], flow['to'], flow['rate'], flow['contact']

            # On modifie le nom des noeuds Rt pour marcher avec avant
            if u.startswith('Rt'):
                u = u.replace(
                    '(', '').replace(')', '').replace(',', '')
            if v.startswith('Rt'):
                v = v.replace(
                    '(', '').replace(')', '').replace(',', '')

            # Orientation des flèches pointillées et segmentées
            if u.startswith('Null'):
                u = u.replace('_', '')
                u += '_' + v
                bend = 'left'
            if v.startswith('Null'):
                v = v.replace('_', '')
                v += '_' + u
                bend = 'left'

            # Orientation et angle des flèches pleines
            bendBase = 'left'
            angle = baseAngle
            try:
                # If both nodes in layer 0 and far away
                if abs(layer0.index(u) - layer0.index(v)) > 1 and \
                        u not in joint and v not in joint:
                    angle = 25
            except:
                try:
                    # If both nodes in layer 1 and far away
                    if abs(layer1.index(u) - layer1.index(v)) > 1 and \
                            u not in joint and v not in joint:
                        angle = 25
                except:
                    pass

            # Position des attaches sur les arrêtes pleines
            if v.startswith('Rt'):
                pos1 = "3/5"
                pos2 = "2/5"
            elif getFlowType(flow) != 'contact':
                pos1 = "2/5"
                pos2 = "3/5"
            else:
                pos1 = contactPositions[0]
                pos2 = contactPositions[1]

            # Définition des couleurs additionnelles
            if getFlowType(flow) == 'rate' and not u.startswith('Null'):
                color = 'Red'
            Arrow.append(f"({u}) edge [bend {bendBase}={angle}, {color}] node [Empty, pos={pos1}] ({u}-{v}-{v_r}-{v_c}-r) {{}} " +
                         f"node [Empty, pos={pos2}] ({u}-{v}-{v_r}-{v_c}-c) {{}} node [Empty, pos=1/2, above={.05*scale}cm] ({u}{v}{v_r}{v_c}) {{}} ({v})")

            # Créer arrêtes pointillées
            for r in v_r.split('+'):
                bend = 'right'
                angle = 30
                if v.startswith('Null') or u.startswith('Null'):
                    bend = 'left'
                    if r == v or r == u:
                        angle = 45

                # Ajouter l'arrête
                if not r.startswith('Null'):
                    if u.startswith('Null'):
                        Dotted.append(
                            f"({u}-{v}-{v_r}-{v_c}-r) edge [bend {bend}={angle}] ({r})")

            # Créer arrêtes segmentées
            for c in v_c.split('+'):
                bend = 'left'
                angle = 30
                if getI(c) == getI(v) and modified:
                    bend = 'right'
                if v.startswith('Rt'):
                    angle = 20

                if not c.startswith('Null'):
                    Dashed.append(
                        f"({u}-{v}-{v_r}-{v_c}-c) edge [bend {bend}={angle}] ({c})")

    names = ['Arrow', 'Dotted', 'Dashed']
    for i, table in enumerate([Arrow, Dotted, Dashed]):
        if len(table) > 0:
            LaTeX += '\n' + tab * 2 + '\\begin{pgfonlayer}{bg}'
            LaTeX += f'\n{tab * 2}\\path [{names[i]}, line width={scale}pt]'
            for arrow in table:
                LaTeX += '\n' + tab * 3 + arrow
            LaTeX += ';'
            LaTeX += '\n' + tab * 2 + '\\end{pgfonlayer}'
            LaTeX += '\n'

    label = "\\label{fig:" + modelName + "_Tikz}"
    LaTeX += f"{tab}\\end{{tikzpicture}}\n{tab + label}\n\\end{{figure}}"

    if not os.path.isdir('LaTeX'):
        os.mkdir('LaTeX')
    with open('LaTeX/' + modelName + '.tex', 'w') as file:
        file.write(LaTeX)

    print(f'Tikz created for {modelName}')
