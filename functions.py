import numpy as np
from scipy.integrate import odeint
from dataclasses import dataclass
from typing import Tuple
from math import *
import matplotlib.pyplot as plt
import time
import json
from tqdm.notebook import tqdm
import os


@dataclass
class Delta:
    flux: list


@dataclass
class Flux:
    coef_indices: Tuple[int, int]
    rate_index: list
    contact_index: list


types = [Flux]


functions = {}


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
                = eval('lambda t: ' + flow['parameter'])


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
               xlabel: str = 'Time', ylabel: str = 'Number of people',
               scales: list = ['linear', 'linear'],
               fig=plt, legendLoc: str = 'best',
               colors: list = None, ycolor: str = 'black') -> None:
    """
    Plots given curves. If xPoints are the same for all curves, give only np.ndarray.
    Otherwise, a list of np.ndarrays works, in which case it has to be given for every curve.
    Other options (title, labels, scales, etc.) are the same as for matplotlib.pyplot.plot function.
    Need to use 

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
        fig: figure or axes
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
    if colors == None:
        colors = [None for _ in liste]

    k = 0
    # TODO rewrite using try except
    if type(xPoints) is np.ndarray:  # Only one set of x coordinates
        for curve in toPlot:
            if labels == None:
                fig.plot(xPoints,
                         curves[curve],
                         style[curve],
                         c=colors[curve])
                k += 1
            else:
                fig.plot(xPoints,
                         curves[curve],
                         style[curve],
                         label=labels[curve],
                         c=colors[curve])
                k += 1
    else:  # Different time scales
        for curve in toPlot:
            if labels == None:
                fig.plot(xPoints[curve],
                         curves[curve],
                         style[curve],
                         c=colors[curve])
                k += 1
            else:
                fig.plot(xPoints[curve],
                         curves[curve],
                         style[curve],
                         label=labels[curve],
                         c=colors[curve])
                k += 1

    if labels != None:
        fig.legend(loc=legendLoc)

    try:
        fig.title(title)
        fig.xlabel(xlabel)
        fig.ylabel(ylabel, color=ycolor)
        fig.xscale(scales[0])
        fig.yscale(scales[1])
    except:
        fig.set_title(title)
        fig.set_xlabel(xlabel)
        fig.set_ylabel(ylabel, color=ycolor)
        fig.set_xscale(scales[0])
        fig.set_yscale(scales[1])


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

    if "Null_n" not in model['compartments'] or "Null_m" not in model['compartments']:
        raise Exception('Model doesn\'t have both Null nodes.')

    if model['name'] != modelName:
        raise Exception(f"Model doesn\'t have right name in file. "
                        + f"Name in file: {model['name']}. Wanted name: {modelName}.")

    missing = []
    flows = model['flows']
    compartments = getCompartments(model)
    for flowType_index, flowType in enumerate(flows):
        for flow_index, flow in enumerate(flows[flowType]):
            # Check for missing keys in flows
            keys = list(flow.keys())
            for p in ['from', 'to', 'rate', 'contact', 'parameter']:
                if p not in keys and p not in missing:
                    missing.append(p)

    if missing != []:
        missingStr = ', '.join(list(map(lambda x: f'"{x}"', missing)))
        missingStr = rreplace(missingStr, ', ', ' and ', 1)
        raise Exception(f'Some flows are missing parameters {missingStr}.')

    if printText:
        print('Model verified.')


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
    with open(f'models/{name}.json') as file:
        model = json.load(file)

    # Try fixing the model if there are problems with nulls.
    missingNulln = False
    missingNullm = False
    if 'Null_n' not in model['compartments']:
        missingNulln = True
    if 'Null_m' not in model['compartments']:
        missingNullm = True

    if missingNulln or missingNullm or 'Null' in getCompartments(model):
        print('Fixing missing empty nodes.')
        flows = model['flows']
        model['compartments']['Null_n'] = {
            "susceptibility": 1,
            "contagiousness": 1,
            "initial_condition": 0
        }
        model['compartments']['Null_m'] = {
            "susceptibility": 1,
            "contagiousness": 1,
            "initial_condition": 0
        }

        for _, flowType in enumerate(flows):
            for _, flow in enumerate(flows[flowType]):
                for i, arg in enumerate(flow):
                    if flow[arg] == 'Null':
                        # i even: change null to null_n
                        # i odd: change null to null_m
                        flow[arg] = 'Null' + ('_n' if not i % 2 else '_m')

        # We don't need Null anymore
        if 'Null' in model['compartments']:
            del model['compartments']['Null']

        print('Model should be fixed...')

    # Verify
    verifyModel(model, name, printText=printText)
    # Write to file
    writeModel(model, overWrite=overWrite, printText=printText)
    # Store functions in dictionary
    storeFunctions(model)
    return model


def initialize(model: dict, y0: dict, t: float, scaled=False, originalModel:
               dict = None, printText: bool = True,
               whereToAdd: str = 'to') -> None:
    """
    This modifies "model", but doesn't modify the file it comes from.

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
        if printText:
            print(f'Initializing with values {roundDict(y0, 2)}.')
        weWant = getCompartments(originalModel)
        if sorted(list(y0.keys())) != sorted(weWant):
            raise Exception("Initialization vector doesn't have right entries.\n"
                            + f"    Entries wanted:   {weWant}.\n"
                            + f"    Entries obtained: {list(y0.keys())}.")

        if scaled:
            scaledInfs = infsScaled(originalModel, y0, t, whereToAdd)
        else:
            scaledInfs = infs(originalModel, y0, t, whereToAdd)

        # ! this will not work with new modification...
        for compartment in y0:
            model['compartments'][addI(
                compartment, 0)]["initial_condition"] = scaledInfs[compartment]
            model['compartments'][addI(
                compartment, 1)]["initial_condition"] = y0[compartment] - scaledInfs[compartment]

        if printText:
            print(f'NewDelta: {[round(scaledInfs[x], 2) for x in scaledInfs]}')
            print(f"Init done. Values for layer 0: " +
                  f"{[round(model['compartments'][addI(x, 0)]['initial_condition'], 2) for x in weWant if not x.startswith('Null')]}")
            print(f"           Values for layer 1: " +
                  f"{[round(model['compartments'][addI(x, 1)]['initial_condition'], 2) for x in weWant if not x.startswith('Null')]}")

    else:
        # No need for any ajustments. Simply write the values.
        for compartment in list(y0.keys()):
            model['compartments'][compartment]["initial_condition"] \
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
    weWant = [x for x in getOtherNodes(model) if x[:4] != 'Null']
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
    string = flow['parameter']
    fonc = eval('lambda t: ' + string)
    value = fonc(t)

    return value


def evalDelta(model: dict, delta: Delta, state: np.ndarray or list,
              t: float) -> float:
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

    susceptibility = [model['compartments'][comp]
                      ['susceptibility'] for comp in compartments]
    contagiousness = [model['compartments'][comp]
                      ['contagiousness'] for comp in compartments]

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

    coefsInfluence = [getCoefForFlux(model, flux, t)
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

    def derivativeForThis(state, t):
        # evalDelta takes care of getting the right coefficient for the derivative
        inflows = evalDelta(model, FBC[i][0], state, t)
        outflows = evalDelta(model, FBC[i][1], state, t)
        return inflows - outflows
    return derivativeForThis


def model_derivative(state: np.ndarray or list, t: float, derivatives: list) -> list:
    """
    Gets the derivative functions for every compartments evaluated at given state.

    Inputs:
        state: np.ndarray or list
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
    dstate_dt = [derivatives[i](state, t) for i in range(len(state))]
    return dstate_dt


def solve(model: dict, tRange: tuple, refine: int, printText=False) -> tuple:
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

    compartments = getCompartments(model)
    steps = (tRange[1] - tRange[0]) * refine + 1
    t_span = np.linspace(tRange[0], tRange[1], num=ceil(steps))

    derivatives = [derivativeFor(model, c)
                   for c in compartments]

    solution = odeint(model_derivative, [
        model['compartments'][comp]['initial_condition'] for comp in compartments
    ], t_span, args=(derivatives,))

    if printText:
        print(f'Model took {time.time() - ti:.1e} seconds to solve.')

    return solution, t_span


def getFlowType(flow: dict) -> str:
    """batches, rates, contacts or u-contacts"""
    if flow['rate'].startswith('Null'):
        if flow['contact'].startswith('Null'):
            return 'batch'
        else:
            return 'u-contact'
    else:
        if flow['contact'].startswith('Null'):
            return 'rate'
        else:
            return 'contact'


def addI(node: str, i: int) -> str:
    """Pour ne pas ajouter des indices à Null ou R0."""
    newNode = (node + f'^{i}') if not (node[:4]
                                       == 'Null' or node[:2] == 'Rt') else (node)
    return newNode


def removeI(node: str) -> str:
    """Pour retrouver un noeud initial."""
    if len(node) > 1:
        newNode = node[:-2] if node[-2] == '^' else node
    else:
        newNode = node
    return newNode


def getI(node: str) -> str:
    """Pour retrouver un noeud initial."""
    remove = len(removeI(node))

    if remove == len(node):
        return -1
    else:
        return int(node[remove + 1:])


def joinNodeSum(nodes: list) -> str:
    return '+'.join(removeDuplicates(nodes))


def mod(model: dict, printWarnings: bool = True,
        printText: bool = False, autoInfections: bool = True,
        write=True, overWrite=False) -> dict:
    """
    This function is the main point of the research.
    Creates the modified model from the base one.
    TODO this function is very long and needs to be reworked.
    """
    if printText:
        print('\nCreating new model!')
    ti = time.time()

    newModel = {"name": model['name'] + '_mod',
                "compartments": {}, "flows": {}}

    compartments = getCompartments(model)
    flows = model['flows']

    # Verify if structure is already well implemented
    if 'Rt' in list(map(lambda x: x[:2], compartments)):
        raise Exception(
            "There is already a node called 'Rt...', please change its name.")
    if 'Null_n' not in compartments:
        raise Exception(
            "There are no nodes called 'Null_n'. Cannot identify empty nodes.")
    if 'Null_m' not in compartments:
        raise Exception(
            "There are no nodes called 'Null_m'. Cannot identify empty nodes.")

    # Add compartment copies and Null node
    for compartment in compartments:
        if compartment[:4] != 'Null':
            for i in range(2):
                newModel["compartments"][compartment + f'^{i}'] \
                    = model["compartments"][compartment].copy()
                # Pour le moment, on place toute la population dans la couche 1
                # Il faudra initialiser le modèle avec les bonnes valeurs pour régler ceci
                # Bonne solution temporaire puisqu'on peut utiliser le modèle comme normal
                if i == 0:
                    newModel["compartments"][compartment +
                                             f'^{i}']["initial_condition"] = 0
        else:
            # Null node is not duplicated
            newModel["compartments"][compartment] \
                = model["compartments"][compartment].copy()

    # Ajouter les nouvelles arrêtes et leurs informations
    for _, flowName in enumerate(flows):
        newModel['flows'][flowName] = []

        newFlow = {
            "from": "Null_n",
            "to": "Null_m",
            "rate": "Null_n",
            "contact": "Null_m",
            "parameter": "0"
        }

        for flow in flows[flowName]:
            # Information du flot original
            u = flow['from']
            v = flow['to']
            vr = flow['rate'].split('+')
            vc = flow['contact'].split('+')

            ### RATES ###
            if getFlowType(flow) == 'rate':
                for i in range(2):
                    if not (i == 0 and u[:4] == 'Null'):
                        uPrime = addI(u, i)
                        vPrime = addI(v, i)
                        # Find vr' and vc'
                        if u[:4] == 'Null':
                            rateNode = joinNodeSum(list(map(
                                lambda x: joinNodeSum([addI(x, j)
                                                   for j in range(2)]),
                                vr)))
                        else:
                            rateNode = joinNodeSum(list(map(
                                lambda x: addI(x, i),
                                vr
                            )))
                        contactNode = 'Null_m'

                        newFlow['from'] = uPrime
                        newFlow['to'] = vPrime
                        newFlow['rate'] = rateNode
                        newFlow['contact'] = contactNode
                        newFlow['parameter'] = flow['parameter']

                        # print('  ', newFlow)

                        newModel['flows'][flowName].append(newFlow.copy())
            ### BATCHES ###
            if getFlowType(flow) == 'batch':
                ### ATTENTION ####
                # Si une batch crée des infections, il faudra s'assurer qu'au
                # moins une infection est créée dans la couche 0. Il faut donc
                # jouer un peu sur ces paramètres...
                # splitBatch = False
                # for _, flowName2 in enumerate(flows):
                #     for flow2 in flows[flowName2]:
                #         if flow2['to'] == v and getFlowType(flow2) == 'contact':
                #             splitBatch = True
                #             if printWarnings:
                #                 print(f'Warning: had to double a batch '
                #                       + f'from {u} to {v}.')

                uPrime = addI(u, 1)
                vPrime = addI(v, 1)
                rateNode = 'Null_n'
                contactNode = 'Null_m'

                newFlow['from'] = uPrime
                newFlow['to'] = vPrime
                newFlow['rate'] = rateNode
                newFlow['contact'] = contactNode
                newFlow['parameter'] = flow['parameter']

                # newFlow['split'] = False

                # newBatch_split = {
                #     'from': addI(u, 0),
                #     'to': addI(v, 0),
                #     'rate': rateNode,
                #     'contact': contactNode,
                #     'parameter': flow['parameter'],
                #     'split': True
                # }

                newModel['flows'][flowName].append(newFlow.copy())
                # newModel['flows'][flowName].append(newBatch_split.copy())
            ### CONTACTS ###
            if getFlowType(flow) == 'contact':
                uPrime = addI(u, 1)
                vPrime = addI(v, 1)
                rateNode = joinNodeSum(list(map(
                    lambda x: joinNodeSum([addI(x, j)
                                           for j in range(2)]),
                    vr)))
                contactNode = joinNodeSum(list(map(
                    lambda x: joinNodeSum([addI(x, j)
                                           for j in range(2)]),
                    vc)))

                newFlow['from'] = uPrime
                newFlow['to'] = vPrime
                newFlow['rate'] = rateNode
                newFlow['contact'] = contactNode
                newFlow['parameter'] = flow['parameter']

                # print('  ', newFlow)

                newModel['flows'][flowName].append(newFlow.copy())

                compartName = f"Rt({flow['from']},{flow['to']})"
                newModel["compartments"][compartName] = {
                    "susceptibility": 1,
                    "contagiousness": 1,
                    "initial_condition": 0
                }

                uPrime = 'Null_n'
                vPrime = compartName
                if autoInfections:
                    rateNode = joinNodeSum(list(map(
                        lambda x: joinNodeSum([addI(x, 0), addI(x, 1)]),
                        vr
                    )))
                else:
                    rateNode = joinNodeSum(list(map(
                        lambda x: addI(x, 1),
                        vr
                    )))
                contactNode = joinNodeSum(list(map(
                    lambda x: addI(x, 0),
                    vc
                )))

                newFlow['from'] = uPrime
                newFlow['to'] = vPrime
                newFlow['rate'] = rateNode
                newFlow['contact'] = contactNode
                newFlow['parameter'] = flow['parameter']

                # print('  ', newFlow)

                newModel['flows'][flowName].append(newFlow.copy())

    if printText:
        print(f'New model created in {time.time() - ti:.1e} seconds.\n')

    writeModel(newModel, overWrite, printText)
    storeFunctions(newModel)
    return newModel


def printModel(model: dict) -> None:
    """Imprime tout le dictionaire du modèle de manière formattée."""
    print(json.dumps(model, sort_keys=True, indent=2))


def roundDict(dictionary: dict, i: int) -> dict:
    """Renvoie le dictionaire arrondi, utile pour les tailles des noeuds."""
    roundedDict = {key: round(dictionary[key], i) for key in dictionary}

    if i < 1:
        roundedDict = {key: int(roundedDict[key]) for key in roundedDict}

    return roundedDict


def analysis(model: dict, solution: np.ndarray, nodes: bool = False,
             changes: bool = False, end: bool = False, R0: bool = True,
             maximums: bool = True) -> None:
    """Prints important information on solutions, all togglable."""

    if nodes:
        print(f"Compartments: {getCompartments(model)}")
        print(f"Population:  {getPopNodes(model)}")
        print(f"Other nodes: {getOtherNodes(model)}")

    if changes:
        print(f"Population change:  {getPopChange(model, solution):+.2f}")
        print(f"Other nodes change: {getOtherChange(model, solution):+.2f}")

    if end:
        print(f"Population at end:  " +
              f"{roundDict(getPopulation(model, solution[-1]), 2)}")
        print(f"Other nodes at end: " +
              f"{roundDict(getOthers(model, solution[-1]), 2)}")

    if R0:
        RtNodes = getRtNodes(model)
        compartments = getCompartments(model)
        length = max(list(map(len, RtNodes))) + 1
        for x in RtNodes:
            compName = f"{x + ':':<{length}}"
            value = f'{solution[-1, compartments.index(x)]: .2f}'
            print(f"{compName}{value}")

    if maximums:
        maximums = {}
        for i in list(map(lambda y: getCompartments(model).index(y),
                          [x for x in getCompartments(model) if x.endswith(('^0'))])):
            maximums[getCompartments(model)[i]] = np.max(solution[:, i])
        print(f'Maximums: {roundDict(maximums, 2)}')


def computeRt(modelName: str, t_span_rt: tuple, sub_rt: float = 1,
              t_span_sim: tuple = (0, 100), sub_sim: float = 100,
              scaledInfs=False, autoInfections=True,
              verification: bool = True, write: bool = True,
              overWrite: bool = False, whereToAdd: str = 'to',
              printText=True, printInit=False, printWarnings=True,
              r0=False, scaleMethod: str = 'Total',
              printR0: bool = False) -> tuple:
    """
    Returns a dictionary with Rt values,
    as well as models and solutions.
    """

    if printText:
        if r0:
            print('\nComputation of R0 ' +
                  ('with autoInfections' if autoInfections else ''))
        else:
            print('\nComputation of Rt ' +
                  ('with autoInfections' if autoInfections else ''))

    if printWarnings:
        if sub_rt > sub_sim:
            print('Warning: rt precision too high.')

    modelOld = loadModel(modelName, printText=printText)
    solutionOld, t_spanOld = solve(modelOld, (0, t_span_rt[1]), sub_sim)
    oldCompartments = getCompartments(modelOld)

    newModel = mod(modelOld, printWarnings, printText, autoInfections=autoInfections,
                   write=write, overWrite=overWrite)
    solution, _ = solve(newModel, (0, t_span_rt[1]), sub_sim)
    compartments = getCompartments(newModel)

    # Vérification!
    if verification:
        allGood = True
        problems = []
        for comp in getPopNodes(modelOld):
            if comp[:4] != 'Null':
                array1 = solutionOld[:, oldCompartments.index(comp)]
                array2 = np.sum(np.array(
                    [solution[:, compartments.index(addI(comp, i))]
                     for i in range(2)]), axis=0)
            else:
                array1 = solutionOld[:, oldCompartments.index(comp)]
                array2 = np.sum(np.array(
                    [solution[:, compartments.index(x)]
                     for x in getOtherNodes(newModel)]), axis=0)

            # print(f"{comp + ':':<{length}}", np.allclose(array1, array2))
            if not np.allclose(array1, array2) or not np.allclose(array2, array1):
                allGood = False
                problems.append(comp)
        if not allGood and printWarnings:
            print('Il semble que les modèles aient des résultats différents.')
            print('On continue l\'expérience quand même, à vérifier.')
            print(f'Problèmes: {problems}.')
        else:
            if printText:
                print('Véfication faite, les deux modèles sont identiques.')

    flows = newModel['flows']

    values = {}
    # No need for progress bar if only computing R0
    iterations = np.arange(t_span_rt[0],
                           t_span_rt[1] + .5 / sub_rt,
                           1 / sub_rt)
    iterator = tqdm(iterations) if len(iterations) > 1 else iterations
    for t in iterator:

        values[t] = {}
        pointIndex = find_nearest(t_spanOld, t)
        pointTime = t_spanOld[pointIndex]
        init = {key: solutionOld[pointIndex, i]
                for i, key in enumerate(oldCompartments)}
        initialize(newModel, init, pointIndex, scaledInfs, modelOld,
                   printText=printInit and t == t_span_rt[0], whereToAdd=whereToAdd)

        solutionTemp, _ = solve(newModel, t_span_sim, sub_sim)

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
                print(f"{'Node':<{length + 3}}Value  Divide  Rt")

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
                if t == t_span_rt[0]:
                    print(f"{x + ', ':<{length + 2}}{value:5.2f}, "
                          + f"{denom:6.2f}, {newValue:5.2f}")
            values[t][x] = newValue
        # print(f'{sum(values[t_spanOld[i]]):.2f} ', end='')

    if printText:
        if r0:
            print('R0 computation done\n')
        else:
            print('Rt computation done\n')

    toKeep = np.where(np.logical_and(t_span_rt[0] <= t_spanOld,
                                     t_spanOld <= t_span_rt[1]))[0]
    return modelOld, newModel, solutionOld[toKeep], t_spanOld[toKeep], values


def computeR0(modelName: str, t_span_sim: tuple = (0, 100),
              sub_sim: float = 100, scaledInfs=False,
              autoInfections: bool = True, write: bool = False,
              overWrite: bool = False, whereToAdd: str = 'to',
              printText=True, printInit: bool = True,
              printWarnings: bool = True, scaleMethod: str = 'Total',
              printR0: bool = False) -> dict:
    """Computes R0 associated with all contact nodes.
    However, if a variant is not present at start, R0 will be 0.
    This is because it was impossible at t=0 to know that variant would appear."""

    modelOld, newModel, solutionOld, _, values = \
        computeRt(modelName, (0, 0), 1, t_span_sim,
                  sub_sim, scaledInfs=scaledInfs, verification=True,
                  write=write, overWrite=overWrite, whereToAdd=whereToAdd,
                  printInit=printInit, r0=True, autoInfections=autoInfections,
                  printWarnings=printWarnings, printText=printText,
                  scaleMethod=scaleMethod, printR0=printR0)

    initialConds = solutionOld[0]
    return modelOld, newModel, initialConds, values[0]


def compare(modelName: str, t_span_rt: tuple, sub_rt: float = 1,
            R0: float = 0, autoToPlot=[True], scaledToPlot=[False],
            t_span_sim: tuple = (0, 100), sub_sim: float = 100,
            verification: bool = False, write: bool = False,
            overWrite: bool = False, whereToAdd: str = 'to',
            printText=False, printInit=False,
            plotANA: bool = True,
            susceptibles: list = [0],
            plotANA_v2: bool = False,
            infected: list = [1],
            scaleMethod: str = 'Total',
            plotIndividual: bool = False,
            plotBound: bool = False,
            printR0: bool = False, plotScaled=True) -> None:
    """Does all possible scenarios"""

    WIDTH = .5
    DASH = (10, 10)
    DOTS = (1, 2)

    fig = plt.figure()
    # plt.yscale('log')

    plt.axhline(y=0, linestyle='--', color='grey',
                linewidth=WIDTH, dashes=DASH)
    plt.axhline(y=1, linestyle='--', color='grey',
                linewidth=WIDTH, dashes=DASH)

    i = 0
    rtCurves = {i: {} for i in range(4)}
    plotedInfsLine = False

    for auto in autoToPlot:
        for scaled in scaledToPlot:
            model, newModel, solution, t_span, values = computeRt(
                modelName, t_span_rt, sub_rt, autoInfections=auto,
                t_span_sim=t_span_sim, sub_sim=sub_sim,
                verification=verification, whereToAdd=whereToAdd,
                scaledInfs=scaled, write=write, overWrite=overWrite,
                printText=printText, printInit=printInit,
                printWarnings=(i == 0), scaleMethod=scaleMethod,
                printR0=printR0)

            if i == 0:
                susceptiblesDivPop = np.sum(solution[:, susceptibles], axis=1) / \
                    np.array([getPopulation(model, x)['Sum']
                              for x in solution])
                infectedDivPop = np.sum(solution[:, infected], axis=1) / \
                    np.array([getPopulation(model, x)['Sum']
                              for x in solution])
                rt_ANA = R0 * susceptiblesDivPop
                if plotANA:
                    plt.plot(t_span, rt_ANA, label='ANA')
                rt_ANA_v2 = R0 * (susceptiblesDivPop - infectedDivPop)
                if plotANA_v2:
                    plt.plot(t_span, rt_ANA_v2, label='ANA_v2')
                bound = rt_ANA * (1 - R0 * infectedDivPop)
                if plotBound:
                    plt.plot(t_span, bound, label='Bound')

                infsScaled = infCurveScaled(model, solution, t_span)
                infsNotScaled = infCurve(model, solution, t_span)
                plt.plot(
                    t_span, infsScaled if plotScaled else infsNotScaled, label='Inci (scaled)' if plotScaled else 'Inci')

                rt_times = np.array([key for key in values])

            rt = np.zeros_like(rt_times, dtype='float64')
            for rtNode in getRtNodes(mod(model, False, False)):
                rt_rtNode = np.array([values[key][rtNode] for key in values])
                rtCurves[i][rtNode] = rt_rtNode
                if len(getRtNodes(mod(model, False, False))) > 1 \
                        and i == 0 \
                        and plotIndividual:
                    plt.plot(rt_times, rt_rtNode, label=rtNode)
                rt += rt_rtNode

            rtCurves[i]['Sum'] = rt

            print(f'Scaled: {scaled}, Auto: {auto}')
            if doesIntersect(rt, 1):
                idx_infs = find_nearest(infsScaled, 1)
                xTimeInfs = t_span[idx_infs]
                idx_rt = find_intersections(rt, 1)[0]
                try:
                    xTimeRt = rt_times[idx_rt]
                except:
                    xTimeRt = (rt_times[int(idx_rt)] +
                               rt_times[int(idx_rt + 1)]) / 2
                print(f'Rt = 1 at {xTimeRt:.3f}')
                if doesIntersect(rt_ANA_v2, 1):
                    idx_rt_ANA_v2 = find_intersections(rt_ANA_v2, 1)[0]
                    try:
                        xTimeRt_ANA_v2 = t_span[idx_rt_ANA_v2]
                    except:
                        xTimeRt_ANA_v2 = (t_span[int(idx_rt_ANA_v2)] +
                                          t_span[int(idx_rt_ANA_v2 + 1)]) / 2
                    print(f'Rt_ANA_v2 = 1 at {xTimeRt_ANA_v2:.3f}')

                print(f'Lower bound respected? ' +
                      ('Yes' if xTimeRt_ANA_v2 < xTimeRt else 'No'))

                idxs_after_rt_eq_1 = np.where(t_span >= xTimeRt)[0]
                times_after_rt_eq_1 = t_span[idxs_after_rt_eq_1]

                # CHECK BOUND !!
                diff_at_moment = np.sum(
                    solution[idxs_after_rt_eq_1[0] - 1, susceptibles]) - np.sum(
                    solution[idxs_after_rt_eq_1[0] - 1, infected])
                susceptibles_after = np.sum(
                    solution[idxs_after_rt_eq_1][:, susceptibles], axis=1)
                idx_problems = np.where(susceptibles_after < diff_at_moment)[0]

                prob_times = times_after_rt_eq_1[idx_problems]
                print(f'No. of moments where susceptibles < difference (i.e. problems): '
                      + f'{len(prob_times)} / {len(susceptibles_after)}')

                print(
                    f'Difference (S - I) at time of importance: {diff_at_moment:.3f}')
                if len(idx_problems) > 0:
                    print(f'Susceptibles at problem time:             ' +
                          f'{susceptibles_after[idx_problems[0]]:.3f}')
                    print(f'Susceptibles just before:                 ' +
                          f'{susceptibles_after[idx_problems[0] - 1]:.3f}')

                print(f'Time difference: {np.abs(xTimeInfs - xTimeRt)}')
                if not plotedInfsLine:
                    plt.axvline(x=xTimeInfs, linestyle=':', color='grey',
                                linewidth=2.5 * WIDTH, dashes=DOTS)
                    plotedInfsLine = True
                plt.axvline(x=xTimeRt, linestyle='--', color='grey',
                            linewidth=WIDTH, dashes=DASH)

                # print(f'rt time: {xTimeRt}, inf time: {xTimeInfs}')
                # print(f'rt intersections: {find_intersections(rt, 1)}')
            else:
                print('Time difference is not relevant, '
                      + 'no intersection between rt and 1.')

            ls = ['-', '--', '-.', ':'][i % 4]
            plt.plot(rt_times, rt, label='SIM' +
                     (' w/ auto' if auto else ' no auto') +
                     (', scaled' if scaled else ', raw'),
                     linestyle=ls)

            i += 1
    plt.title(modelName)

    # plt.ylim(bottom=.1)
    plt.legend(loc='best')

    return rt_times, rtCurves, infsNotScaled


def infs(model: dict, y0: dict, t: float, whereToAdd: str = 'to') -> dict:
    """Returns incidences."""

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
    """Returns scaled incidences, sums to 1."""

    infections = infs(model, y0, t, whereToAdd)
    weWant = getCompartments(model)

    sumInfections = sum(infections[node] for node in weWant)
    denom = sumInfections if sumInfections != 0 else 1

    scaledInfs = {key: infections[key] / denom for key in infections}

    return scaledInfs


def totInfs(model: dict, state: np.ndarray, t: float) -> np.ndarray:
    """Returns total infected for a state."""

    if len(state.shape) > 1:
        raise Exception(
            f'Function can only be used on single state, not solution.')
    weWant = getCompartments(model)
    y0 = {comp: state[i] for i, comp in enumerate(weWant)}
    infections = infs(model, y0, t, whereToAdd='to')

    return sum(infections[comp] for comp in weWant)


def infCurve(model: dict, solution: np.ndarray, t_span: np.ndarray) -> np.ndarray:
    """Returns curve of total infected."""

    curve = np.array([totInfs(model, x, t_span[i])
                     for i, x in enumerate(solution)])
    return curve


def infCurveScaled(model: dict, solution: np.ndarray, t_span: np.ndarray) -> np.ndarray:
    """Returns curve of total infected, scaled so max = 1."""

    curve = infCurve(model, solution, t_span)
    curve = curve / np.max(curve)
    return curve


def writeModel(newModel: dict, overWrite: bool = False, printText: bool = True) -> None:
    """Write model to file. This is useful to save modified models."""
    modelName = newModel['name']
    newFileName = modelName + '.json'
    if printText:
        print(f'Writing model to file models/{newFileName}.')
    if not os.path.isfile(f'models/{newFileName}'):
        # File doesn't exist
        try:
            with open(f'models/{newFileName}', 'w') as file:
                json.dump(newModel, file, indent=4)
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
            print('Overwriting file.')
            try:
                with open(f'models/{newFileName}', 'w') as file:
                    json.dump(newModel, file, indent=4)
            except:
                if printText:
                    print('Problem when writing file.')


def find_nearest(array: np.ndarray, value: float) -> int:
    """Retourne l'indice avec la valueur la plus proche."""
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


def find_intersections(array: np.ndarray, value: float, eps=10**-5) -> list:
    """Finds intersection between curve and value."""

    newCurve = array - value
    where = np.where(abs(newCurve) > eps)[0]

    newWhere = []

    newCurveEps = newCurve[where]

    for i in range(len(newCurveEps) - 1):
        product = newCurveEps[i] * newCurveEps[i + 1]

        if product < 0:
            newWhere.append((where[i] + where[i+1]) / 2)

    return newWhere


def doesIntersect(curve: np.ndarray, value: int, eps=10**-3):
    """Checks if curve intersects y = value."""

    return len(find_intersections(curve, value, eps)) > 0


def createLaTeX(model: dict, layerDistance: str = ".8cm",
                nodeDistance: str = "2cm", varDistance: str = ".1cm",
                nullDistance: str = "1cm", baseAngle: int = 10,
                contactPositions: tuple = ("2/5", "3/5")) -> str:
    """
    Produces tikzfigure for a model. Needs some definitions in preamble:
    usepackage[usenames,dvipsnames]{xcolor}
    usepackage{tikz}
    usetikzlibrary{calc, positioning, arrows.meta, shapes.geometric}

    tikzset{Square/.style={draw=black, rectangle, rounded corners=0pt, align=center, minimum height=1cm, minimum width=1cm}}
    tikzset{Text/.style={rectangle, rounded corners=0pt, inner sep=0, outer sep=0, draw=none, sloped}}
    tikzset{Empty/.style={rectangle, inner sep=0, outer sep=0, draw=none}}

    tikzset{Arrow/.style={-Latex, line width=1pt}}
    tikzset{Dashed/.style={Latex-, dashed, line width=1pt}}
    tikzset{Dotted/.style={Latex-, dotted, line width=1pt}}
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
        'batch': 'Cyan',
    }

    for x in compartments:
        # On veut savoir si le modèle est modifié ou non pour savoir
        # combien de couches il faut faire
        if x.endswith(('^0, ^1')) or x.startswith(('Rt')):
            modified = True

    for x in compartments:
        # Ici on veut regrouper les variants ensemble pour les mettre proche
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

            if len(joint[x]) > 3:
                # Il faudra trouver comment gérer le code tikz si on a beaucoup de variants...
                print("This code doesn't work for 3 variants or more yet.")

    # Création du string qui va contenir le code LaTeX
    LaTeX = f"\\begin{{figure}}[H]\n{tab}\\centering\n{tab}\\begin{{tikzpicture}}\n"

    # Book-keeping
    layer0 = []
    layer1 = []
    others = []
    bases0 = []
    bases1 = []
    for x in compartments:
        # On créé ici E' (l'ensemble des arrêtes)
        if (x.endswith('^0') or (not modified and
                                 not x.startswith(('Null', 'Rt')))) \
                and x not in layer0:
            # Layer 0 (or normal layer in non-modified)
            if len(layer0) != 0:
                # Où placer le noeud par rapport à ceux qui existent
                if layer0[-1] not in joint:
                    where = f"[right={nodeDistance} of {layer0[-1]}]"
                else:
                    where = f"[right={nodeDistance} of {bases0[-1]}]"
            else:
                where = ""

            if x not in joint:
                # Si pas un variant, on le place simplement
                LaTeX += f"{tab * 2}\\node [Square] ({x}) " \
                    + f"{where} " \
                    + f"{{${x}$}};\n"

                layer0.append(x)
            else:
                # Si on a variant, il faut placer noeud du centre
                # et mettre les variants autour
                base = joint[x][2]
                above = joint[x][0]
                below = joint[x][1]

                LaTeX += f"{tab * 2}\\node [Empty] ({base}) " \
                    + f"{where} " \
                    + f"{{}};\n"

                # Placer les noeuds de variants
                LaTeX += f"{tab * 2}\\node [Square] ({above}) " \
                    + f"[above={varDistance} of {base}] " + f"{{${above}$}};\n"
                LaTeX += f"{tab * 2}\\node [Square] ({below}) " \
                    + f"[below={varDistance} of {base}] " + f"{{${below}$}};\n"

                layer0.append(above)
                layer0.append(below)
                bases0.append(base)
        elif x.endswith('^1') and x not in layer1:
            # on est dans layer 1
            if len(layer1) != 0:
                # On regarde si on a point de référence ou pas dans layer 1
                where = f"[right={nodeDistance} of {layer1[-1]}]"
            else:
                # Sinon, on utiliser layer 0 pour construire
                equivalent0 = addI(removeI(x), 0)
                where = f"[below={layerDistance} of {equivalent0}]"

            if x not in joint:
                equivalent0 = addI(removeI(x), 0)
                LaTeX += f"{tab * 2}\\node [Square] ({x}) " + \
                    f"{where} " + \
                    f"{{${x}$}};\n"
                layer1.append(x)
            else:
                base = joint[x][2]
                above = joint[x][0]
                below = joint[x][1]

                LaTeX += f"{tab * 2}\\node [Empty] ({base}) " \
                    + f"{where} " + f"{{}};\n"

                LaTeX += f"{tab * 2}\\node [Square] ({above}) " \
                    + (f"[above={varDistance} of {base}] " if len(layer0) != 0 else '') \
                    + f"{{${above}$}};\n"
                LaTeX += f"{tab * 2}\\node [Square] ({below}) " \
                    + (f"[below={varDistance} of {base}] " if len(layer0) != 0 else '') \
                    + f"{{${below}$}};\n"

                layer1.append(above)
                layer1.append(below)
                bases1.append(base)

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

            # On place à gauche de layer 1 ou vers le haut (causera problème pour
            # trop de Rts différents à calculer)
            if len(others) == 0:
                reference = layer1[0]
                pos = 'left'
                dist = nodeDistance
            else:
                reference = layer0[0]
                pos = 'left'
                dist = nodeDistance
            LaTeX += f"{tab * 2}\\node [Square] ({nameNoProblem}) " \
                + f"[{pos}={dist} of {reference}] " + \
                f"{{${textNoProblem}$}};\n"

            others.append(nameNoProblem)

    layer0 = removeDuplicates(layer0)
    layer1 = removeDuplicates(layer1)
    others = removeDuplicates(others)

    LaTeX += '\n'
    for x in layer0 + layer1 + others:
        # On veut des noeuds vides pour considérer les entrées et sorties
        # Puisque les noeuds ne sont pas montrés on en crée pour chaque autre noeud
        LaTeX += f"{tab * 2}\\node [Empty] (Nulln_{x}) " \
            + f"[above left={nullDistance} of {x}] {{}};\n"
        LaTeX += f"{tab * 2}\\node [Empty] (Nullm_{x}) " \
            + f"[below right={nullDistance} of {x}] {{}};\n"

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
                if abs(layer0.index(u) - layer0.index(v)) > 1 and \
                        u not in joint and v not in joint:
                    angle = 30
            except:
                try:
                    if abs(layer1.index(u) - layer1.index(v)) > 1 and \
                            u not in joint and v not in joint:
                        angle = 30
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
            if getFlowType(flow) == 'contact' and v.startswith('Rt'):
                color = 'Black'
            Arrow.append(f"({u}) edge [bend {bendBase}={angle}, {color}] node [Empty, pos={pos1}] ({u}-{v}-r) {{}} " +
                         f"node [Empty, pos={pos2}] ({u}-{v}-c) {{}} ({v})")

            # Créer arrêtes pointillées
            for r in v_r.split('+'):
                bend = 'right'
                if v.startswith('Null') or u.startswith('Null'):
                    bend = 'left'

                if u.startswith('Null') and r != v:
                    # On a un rate de naissances qui vient d'ailleurs
                    angle = 20
                else:
                    # Dans les autres cas
                    angle = 30

                if getFlowType(flow) == 'contact' and getI(r) == 0 \
                        and not v.startswith('Rt'):
                    # On a un contact (dans layer 1) créé par contact avec
                    # compartiment de layer 0, mais sans considérer les Rt
                    angle = 10
                    # on inverse le bend de la flèche
                    bend = 'right' if bend == 'left' else (
                        'left' if bend == 'right' else bend)
                # Ajouter l'arrête
                if r != 'Null_n':
                    Dotted.append(
                        f"({u}-{v}-r) edge [bend {bend}={angle}] ({r})")

            # Créer arrêtes segmentées
            for c in v_c.split('+'):
                bend = 'right'
                if v.startswith('Null') or u.startswith('Null'):
                    bend = 'left'

                if c == u:
                    angle = 45
                elif v.startswith('Null') and c != u:
                    angle = 20
                else:
                    angle = 30
                bend = 'right' if c == u else 'left'
                if getFlowType(flow) == 'contact' and getI(c) == 0 \
                        and not v.startswith('Rt'):
                    angle = 10
                    bend = 'right' if bend == 'left' else (
                        'left' if bend == 'right' else bend)
                if c != 'Null_m':
                    Dashed.append(
                        f"({u}-{v}-c) edge [bend {bend}={angle}] ({c})")

    names = ['Arrow', 'Dotted', 'Dashed']
    for i, table in enumerate([Arrow, Dotted, Dashed]):
        if len(table) > 0:
            LaTeX += f'\n{tab * 2}\\path [{names[i]}]'
            for arrow in table:
                LaTeX += '\n' + tab * 3 + arrow
            LaTeX += ';\n'

    LaTeX += f"{tab}\\end{{tikzpicture}}\n\\end{{figure}}"

    if not os.path.isdir('LaTeX'):
        os.mkdir('LaTeX')
    with open('LaTeX/' + modelName + '.tex', 'w') as file:
        file.write(LaTeX)
