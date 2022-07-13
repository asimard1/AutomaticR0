import numpy as np
from scipy.integrate import odeint
from dataclasses import dataclass
from typing import Tuple
from math import ceil, inf
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


def removeDuplicates(liste: list) -> list:
    """Retourne la liste sans répétitions."""
    return list(dict.fromkeys(liste))


def rreplace(s, old, new, n):
    """Taken from stack exchange, replaces last n occurences."""
    li = s.rsplit(old, n)
    return new.join(li)


def find_nearest(array: np.ndarray, value: float) -> int:
    """Retourne l'indice avec la valueur la plus proche."""
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


def find_intersection(array: np.ndarray, value: float) -> tuple:
    """Returns the pair of value that bound the intersection more closely."""
    idx = find_nearest(array, value)

    if array[idx] > value:
        if array[idx - 1] < value:
            toReturn = (idx - 1, idx)
        elif array[idx + 1] < value:
            toReturn = (idx, idx + 1)
        else:
            toReturn = (idx, idx)
    else:
        if array[idx - 1] > value:
            toReturn = (idx - 1, idx)
        elif array[idx + 1] > value:
            toReturn = (idx, idx + 1)
        else:
            toReturn = (idx, idx)

    valueNearest = array[idx]
    valueMean = (array[toReturn[0]] + array[toReturn[1]]) / 2

    if np.abs(valueNearest - value) < np.abs(valueMean - value):
        print(valueNearest, valueMean)
        return (idx, idx)
    else:
        print(valueNearest, valueMean)
        return toReturn


def plotCurves(xPoints: np.ndarray or list, curves, toPlot: list, labels: list,
               title: str = 'Infection curves', style: list = None,
               xlabel: str = 'Time', ylabel: str = 'Number of people',
               scales: list = ['linear', 'linear'],
               fig=plt, yTitle: str = None, fontsize: int = 15,
               legendLoc: str = 'best', colors: list = None,
               ycolor: str = 'black') -> None:
    """
    Plots given curves. If xPoints are the same for all curves, give only np.ndarray.
    Otherwise, a list of np.ndarrays works, in which case it has to be given for every curve.
    Other options (title, labels, scales, etc.) are the same as for matplotlib.pyplot.plot function.
    """

    liste = list(range(max(toPlot) + 1))
    if style == None:
        style = ['-' for _ in liste]
    if colors == None:
        colors = [None for _ in liste]

    k = 0
    if type(xPoints) is np.ndarray:  # Only one set of x coordinates
        for curve in toPlot:
            if labels == None:
                fig.plot(xPoints, curves[curve], style[curve], c=colors[curve])
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
                fig.plot(xPoints[curve], curves[curve],
                         style[curve], c=colors[curve])
                k += 1
            else:
                fig.plot(xPoints[curve], curves[curve],
                         style[curve], label=labels[curve], c=colors[curve])
                k += 1

    if labels != None:
        fig.legend(loc=legendLoc)

    try:
        fig.title(title, fontsize=fontsize, y=yTitle)
        fig.xlabel(xlabel)
        fig.ylabel(ylabel, color=ycolor)
        fig.xscale(scales[0])
        fig.yscale(scales[1])
    except:
        fig.set_title(title, fontsize=fontsize, y=yTitle)
        fig.set_xlabel(xlabel)
        fig.set_ylabel(ylabel, color=ycolor)
        fig.set_xscale(scales[0])
        fig.set_yscale(scales[1])


def verifyModel(model: dict) -> None:
    """Verifies if model has the right properties. Might not be complete."""
    if "Null_n" not in model['compartments'] or "Null_m" not in model['compartments']:
        raise Exception('Model doesn\'t have both Null nodes.')

    missing = []
    flows = model['flows']
    compartments = getCompartments(model)
    for flowType_index, flowType in enumerate(flows):
        for flow_index, flow in enumerate(flows[flowType]):
            # TODO check if flows don't have v_r and v_c that contain nulls but aren't nulls!
            if 'rate' in flow:
                v_r = flow['rate'].split('+')
                for x in v_r:
                    if x not in compartments:
                        raise Exception(f'Compartment {x} found in rates, not in compartment list.')
                    if x.startswith('Null'):
                        if len(v_r) > 1:
                            raise Exception(f'Some flow has a rate which is a sum containing {x}.')
            if 'contact' in flow:
                v_c = flow['contact'].split('+')
                for x in v_c:
                    if x not in compartments:
                        raise Exception(f'Compartment {x} found in contacts, not in compartment list.')
                    if x.startswith('Null'):
                        if len(v_c) > 1:
                            raise Exception(f'Some flow has a contact which is a sum containing {x}.')

            keys = list(flow.keys())
            for p in ['from', 'to', 'rate', 'contact', 'parameter', 'ti', 'tf']:
                if p not in keys and p not in missing:
                    missing.append(p)
    if missing != []:
        missingStr = ', '.join(list(map(lambda x: f'"{x}"', missing)))
        missingStr = rreplace(missingStr, ', ', ' and ', 1)
        raise Exception(f'Some flows are missing parameters {missingStr}.')

    print('Model verified.')


def loadModel(name: str) -> dict:
    """
    Envoie un dictionaire contenant le fichier json "name.json".
    Seulement besoin d'appeler sur le nom du modèle.
    Par exemple, loadModel('SIR') fonctionne.
    """
    with open(f'models/{name}.json') as file:
        model = json.load(file)

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

        for flowType_index, flowType in enumerate(flows):
            for flow_index, flow in enumerate(flows[flowType]):
                for i, arg in enumerate(flow):
                    if flow[arg] == 'Null':
                        # i even: change null to null_n
                        flow[arg] = 'Null' + ('_n' if not i % 2 else '_m')

        if 'Null' in model['compartments']:
            del model['compartments']['Null']

        print('Model should be fixed...')

    verifyModel(model)

    writeModel(model, name, overWrite=False)
    return model


def initialize(model: dict, y0: dict, t: float, t0: float, originalModel:
               dict = None, printText: bool = True,
               whereToAdd: str = 'contact') -> None:
    """This modifies "model", but doesn't modify the file it comes from."""

    if originalModel != None:
        if printText:
            print(f'Initializing with values {roundDict(y0, 2)}.')
        weWant = getCompartments(originalModel)
        if sorted(list(y0.keys())) != sorted(weWant):
            raise Exception("Initialization vector doesn't have right entries.\n"
                            + f"Entries wanted:   {weWant}.\n"
                            + f"Entries obtained: {list(y0.keys())}.")

        # TODO add test to see if need normalized or not!!!
        scaledInfs = infs(originalModel, y0, t, t0, whereToAdd)

        for compartment in y0:
            model['compartments'][addI(
                compartment, 0)]["initial_condition"] = scaledInfs[compartment]
            model['compartments'][addI(
                compartment, 1)]["initial_condition"] = y0[compartment] - scaledInfs[compartment]

    else:
        # No need for any ajustments. Simply write the values.
        for compartment in list(y0.keys()):
            model['compartments'][compartment]["initial_condition"] \
                = y0[compartment]

    if printText:
        print(f'NewDelta: {[round(scaledInfs[x], 2) for x in scaledInfs]}')
        print(f"Init done. Values for layer 0: " +
              f"{[round(model['compartments'][addI(x, 0)]['initial_condition'], 2) for x in weWant if x[:4] != 'Null']}")
        print(f"           Values for layer 1: " +
              f"{[round(model['compartments'][addI(x, 1)]['initial_condition'], 2) for x in weWant if x[:4] != 'Null']}")


def newInfections(model: dict, state: np.ndarray or list) -> float:
    """Returns new infections at state for given model."""
    compartments = getCompartments(model)
    weWant = removeDuplicates(
        [removeI(x) for x in compartments if not x.startswith(('Rt'))])
    Delta = model_derivative([x for x in state], 0, model, 0)
    Delta = [max(x, 0) for x in Delta]
    for x in [x for x in compartments if x.startswith('Null')]:
        Delta[weWant.index(x)] = 0

    return sum(Delta)


def getCompartments(model: dict) -> list:
    """List of compartments in model file."""
    return list(model['compartments'].keys())


def getFlowsByCompartments(model: dict) -> list:
    """
    Ceci nous permet d'appeler sur un modèle et directement obtenir tous les flots.
    Aucun besoin de répéter cette structure sur tous nos modèles,
    comme c'était fait avec les définitions de modèles.
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

            rate_i = list(map(lambda x: compartments.index(x),
                          flow['rate'].split('+')))
            contact_i = list(map(lambda x: compartments.index(x),
                                 flow['contact'].split('+')))
            term = Flux((flowType_index, flow_index), rate_i, contact_i)

            try:
                FBC[to_i][0].append(term)
            except:
                pass
            try:
                FBC[from_i][1].append(term)
            except:
                pass

    FBC = [[Delta(*[[f for f in flows if isinstance(f, T)] for T in types])
            for flows in compartmentflows] for compartmentflows in FBC]

    return FBC


def popTot(model: dict, state: np.ndarray or list) -> float:
    """Total population at state with given model structure."""
    compartments = getCompartments(model)

    N = sum([state[i] if (compartment[:4] != 'Null' and compartment[:2] != 'Rt') else 0
             for i, compartment in enumerate(compartments)])

    return N


def getPopNodes(model: dict) -> list:
    """Return compartments that are in the population only."""
    compartments = getCompartments(model)
    weWant = [x for x in compartments if not x.startswith(('Rt', 'Null'))]
    return weWant


def getOtherNodes(model: dict) -> list:
    """Return compartments that are not in the population."""
    compartments = getCompartments(model)
    weWant = [x for x in compartments if x.startswith(('Rt', 'Null'))]
    return weWant


def getRtNodes(model: dict) -> list:
    return [x for x in getOtherNodes(model) if x[:4] != 'Null']


def getNodeValues(model: dict, state: np.ndarray or list, weWant: list) -> dict:
    """Get every value for nodes in weWant as dictionary."""
    if len(state.shape) != 1:
        raise Exception('2nd argument should be a vector, not matrix.')

    dictNb = {}
    compartments = getCompartments(model)

    indexes = list(map(lambda x: compartments.index(x), weWant))
    for i in indexes:
        dictNb[compartments[i]] = state[i]
    dictNb['Sum'] = sum([state[i] for i in indexes])

    return dictNb


def getPopulation(model: dict, state: np.ndarray or list) -> dict:
    """Get every value for nodes in population as dictionary."""
    return getNodeValues(model, state, getPopNodes(model))


def getPopChange(model: dict, solution: np.ndarray) -> float:
    """Get change in population from start to finish."""
    return getPopulation(model, solution[-1])['Sum'] - getPopulation(model, solution[0])['Sum']


def getOthers(model: dict, state: np.ndarray or list) -> dict:
    """Get every value for nodes NOT in population as dictionary."""
    return getNodeValues(model, state, getOtherNodes(model))


def getOtherChange(model: dict, solution: np.ndarray) -> float:
    """Get change in other nodes from start to finish."""
    return getOthers(model, solution[-1])['Sum'] - getOthers(model, solution[0])['Sum']


def getCoefForFlux(model: dict, flux: Flux, t: float, t0: float) -> float:
    """
    Gets the coefficient for flux from the config file.
    """
    flows = model['flows']
    flowTypes = list(model['flows'].keys())
    flowType = flowTypes[flux.coef_indices[0]]
    flowJson = flows[flowType][flux.coef_indices[1]]

    ti = flowJson['ti'] if flowJson['ti'] != None else - inf
    tf = flowJson['tf'] if flowJson['tf'] != None else inf

    if 'split' in list(flowJson.keys()):
        # Pour le flot qui créé un cas index, on ne l'ajoute pas si
        # on ne sait pas déjà qu'il existe.
        if flowJson['split'] == 'New':
            if t0 < ti:
                return 0
        # Pour le flot ajouté à la dynamique, il faut l'enlever si
        # on le met dans les infectés.
        else:
            if t0 > ti:
                return 0

    return flowJson['parameter'] if ti <= t <= tf else 0


def getCoefForFlow(flow: dict, t: float, t0: float) -> float:
    """
    Gets the coefficient for flow from the config file.
    """
    ti = flow['ti'] if flow['ti'] != None else - inf
    tf = flow['tf'] if flow['tf'] != None else inf

    if 'split' in list(flow.keys()):
        # Pour le flot qui créé un cas index, on ne l'ajoute pas si
        # on ne sait pas déjà qu'il existe.
        if flow['split'] == 'New':
            if t0 < ti:
                return 0
        # Pour le flot ajouté à la dynamique, il faut l'enlever si
        # on le met dans les infectés.
        else:
            if t0 > ti:
                return 0

    return flow['parameter'] if ti <= t <= tf else 0


def evalDelta(model: dict, delta: Delta, state: np.ndarray or list,
              t: float, t0: float) -> float:
    """
    Computes the actual derivative for a delta (delta is the dataclass defined earlier).
    """

    compartments = getCompartments(model)

    N = popTot(model, state)

    # A bit useless for the moment...
    susceptibility = [model['compartments'][comp]
                      ['susceptibility'] for comp in compartments]
    contagiousness = [model['compartments'][comp]
                      ['contagiousness'] for comp in compartments]

    rateInfluence = [sum(state[x] for x in flux.rate_index)
                     if len(flux.rate_index) != 1
                     else (state[flux.rate_index[0]]
                           if compartments[flux.rate_index[0]][:4] != 'Null'
                           else 1)
                     for flux in delta.flux]
    contactInfluence = [sum(state[x] for x in flux.contact_index) / N
                        if len(flux.contact_index) != 1
                        else (state[flux.contact_index[0]] / N
                              if compartments[flux.contact_index[0]][:4] != 'Null'
                              else 1)
                        for flux in delta.flux]
    # Influence of time on parameter (i.e. different vaccination in time) could
    # be added in this function: getCoefForFlux. But otherwise, it's too complicated.
    coefsInfluence = [getCoefForFlux(model, flux, t, t0)
                      for flux in delta.flux]

    somme = np.sum(np.array(rateInfluence) *
                   np.array(contactInfluence) *
                   np.array(coefsInfluence))

    return somme

    # b = sum([getCoefForFlux(model, batch) for batch in delta.batches])
    # r = sum([getCoefForFlux(model, rate) * state[rate.from_index]
    #         for rate in delta.rates])
    # m = sum([getCoefForFlux(model, contact) *
    #          state[contact.from_index] * susceptibility[contact.from_index] *
    #          state[contact.contact_index] *
    #          contagiousness[contact.contact_index] / N
    #          for contact in delta.contacts])
    # tb = sum(np.hstack([list(map(lambda x: x[2] if x[0] <= t <= x[1] else 0,
    #                              getCoefForFlux(model, timelyBatch)))
    #                     for timelyBatch in delta.timelybatches])) \
    #     if [0 for _ in delta.timelybatches] != [] else 0

    # return b + r + m + tb


def derivativeFor(model: dict, compartment: str, t: float,
                  t0: float):
    """
    Get the derivative for a compartment as a function.
    """

    compartments = getCompartments(model)
    i = compartments.index(compartment)

    FBC = getFlowsByCompartments(model)

    def derivativeForThis(x, t):
        inflows = evalDelta(model, FBC[i][0], x, t, t0)
        outflows = evalDelta(model, FBC[i][1], x, t, t0)
        return inflows - outflows
    return derivativeForThis


def model_derivative(state: np.ndarray or list, t: float, model: dict,
                     t0: float) -> list:
    """
    Gets the derivative functions for every compartments evaluated at given state.
    """

    compartments = getCompartments(model)
    derivatives = [derivativeFor(model, c, t, t0)
                   for c in compartments]

    # state = [x if x > 0 else 0 for x in state]
    dstate_dt = [derivatives[i](state, t) for i in range(len(state))]
    return dstate_dt


def solve(model: dict, range: tuple, refine: int, printText=False) -> tuple:
    """
    Model solver. Eventually, we would want the first element of range to be used\n
    in order to determine if we need to consider timed elements in the compuation\n
    of R0 (e.g. we don't know whether a new variant will appear or not).
    """

    ti = time.time()

    compartments = getCompartments(model)
    steps = (range[1] - range[0]) * refine + 1
    t_span = np.linspace(range[0], range[1], num=ceil(steps))

    solution = odeint(model_derivative, [
        model['compartments'][comp]['initial_condition'] for comp in compartments
    ], t_span, args=(model, range[0]))

    if printText:
        print(f'Model took {time.time() - ti:.1e} seconds to solve.')

    return solution, t_span


def getFlowType(flow: dict) -> str:
    """batches, rates, contacts or u-contacts"""
    if flow['rate'] == 'Null_n':
        if flow['contact'] == 'Null_m':
            return 'batch'
        else:
            return 'u-contact'
    else:
        if flow['contact'] == 'Null_m':
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


def joinNodeSum(nodes: list) -> str:
    return '+'.join(removeDuplicates(nodes))


def mod(model: dict, printWarnings: bool = True, printText: bool = True,
        autoInfections: bool = False) -> dict:
    """
    This function is the main point of the research.
    Creates the modified model from the base one.
    TODO this function is very long and needs to be reworked.
    """
    if printText:
        print('Creating new model!')
    ti = time.time()

    newModel = {"compartments": {}, "flows": {}}

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
            "parameter": 0,
            "ti": None,
            "tf": None
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
                        newFlow['ti'] = flow['ti']
                        newFlow['tf'] = flow['tf']

                        # print('  ', newFlow)

                        newModel['flows'][flowName].append(newFlow.copy())
            ### BATCHES ###
            if getFlowType(flow) == 'batch':
                uPrime = addI(u, 1)
                vPrime = addI(v, 1)
                rateNode = 'Null_n'
                contactNode = 'Null_m'

                #### ATTENTION ####
                # Si une batch crée des infections, il faudra s'assurer qu'au
                # moins une infection est créée dans la couche 0. Il faut donc
                # jouer un peu sur ces paramètres...
                splitBatch = False
                for _, flowName2 in enumerate(flows):
                    for flow2 in flows[flowName2]:
                        if flow2['to'] == v and getFlowType(flow2) == 'contact':
                            splitBatch = True
                            if printWarnings:
                                print(f'Warning: had to split a batch '
                                      + f'from {u} to {v}.')

                newFlow['from'] = uPrime
                newFlow['to'] = vPrime
                newFlow['rate'] = rateNode
                newFlow['contact'] = contactNode
                if not splitBatch or flow['parameter'] == 0:
                    newFlow['parameter'] = flow['parameter']
                    newFlow['ti'] = flow['ti']
                    newFlow['tf'] = flow['tf']
                else:
                    newFlow['parameter'] = flow['parameter']
                    newFlow['ti'] = flow['ti'] + 1 / newFlow['parameter']
                    newFlow['tf'] = flow['tf']
                    newFlow['split'] = False

                    newBatch = {
                        'from': addI(u, 0),
                        'to': addI(v, 0),
                        'rate': 'Null_n',
                        'contact': 'Null_m',
                        'parameter': flow['parameter'],
                        'ti': flow['ti'],
                        'tf': flow['ti'] + 1 / newFlow['parameter'],
                        'split': 'New'
                    }
                    newBatch2 = {
                        'from': addI(u, 1),
                        'to': addI(v, 1),
                        'rate': 'Null_n',
                        'contact': 'Null_m',
                        'parameter': flow['parameter'],
                        'ti': flow['ti'],
                        'tf': flow['ti'] + 1 / newFlow['parameter'],
                        'split': 'Original'
                    }
                    newModel['flows'][flowName].append(newBatch.copy())
                    newModel['flows'][flowName].append(newBatch2.copy())

                    if printWarnings:
                        print(f'From:  {flow},')
                        print(f'  to:  {newFlow},')
                        print(f'Added: {newBatch}')
                        print(f' and:  {newBatch2}.')

                # print('  ', newFlow)

                newModel['flows'][flowName].append(newFlow.copy())
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
                newFlow['ti'] = flow['ti']
                newFlow['tf'] = flow['tf']

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
                newFlow['ti'] = flow['ti']
                newFlow['tf'] = flow['tf']

                # print('  ', newFlow)

                newModel['flows'][flowName].append(newFlow.copy())

    if printText:
        print(f'New model created in {time.time() - ti:.1e} seconds.')

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
              verification: bool = True, write: bool = False,
              overWrite: bool = False, whereToAdd: str = 'contact',
              printInit=False, r0=False, autoInfections=False) -> tuple:
    """This is an important part. Returns a dictionary with Rt values,
    as well as models and solutions."""

    if r0:
        print('\nComputation of R0 ' + ('with autoInfections' if autoInfections else ''))
    else:
        print('\nComputation of Rt ' + ('with autoInfections' if autoInfections else ''))

    if sub_rt > sub_sim:
        print('Warning: rt precision too high.')

    modelOld = loadModel(modelName)
    solutionOld, t_spanOld = solve(modelOld, t_span_rt, sub_sim)
    oldCompartments = getCompartments(modelOld)

    newModel = mod(modelOld, True, True, autoInfections=autoInfections)
    solution, t_span = solve(newModel, t_span_rt, sub_sim)
    compartments = getCompartments(newModel)

    if write:
        writeModel(newModel, modelName + '_mod', overWrite=overWrite)

    # Vérification!
    if verification:
        allGood = True
        length = max(list(map(len, oldCompartments))) + 1
        for comp in oldCompartments:
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
        if not allGood:
            print('Il semble que les modèles aient des résultats différents.')
            print('On continue l\'expérience quand même, à vérifier.')
        else:
            print('Véfication faite, les deux modèles sont identiques.')

    values = {}
    # No need for progress bar if only computing R0
    iterations = np.arange(t_span_rt[0],
                           t_span_rt[1] + .5 / sub_rt,
                           1 / sub_rt)
    iterator = tqdm(iterations) if len(iterations) > 1 else iterations
    for t in iterator:
        values[t] = {}
        pointIndex = find_nearest(t_spanOld, t)
        init = {key: solutionOld[pointIndex, i]
                for i, key in enumerate(oldCompartments)}
        if printInit:
            print(f'init: {init}')
        initialize(newModel, init, pointIndex, t_span_rt[0], modelOld,
                   printText=printInit, whereToAdd=whereToAdd)
        solutionTemp, t_spanTemp = solve(newModel, t_span_sim, sub_sim)

        initialCond = solutionOld[pointIndex]
        initialCond = {comp: initialCond[i]
                       for i, comp in enumerate(getCompartments(modelOld))}
        initialCond = infs(modelOld, initialCond, t, t_span_rt[0], whereToAdd='to')
        initialCond = {comp: initialCond[comp]
                       for comp in initialCond if initialCond[comp] > 0}

        for x in getRtNodes(newModel):
            compartment = x.split(',')[1].split(')')[0]
            denom = initialCond[compartment] if compartment in initialCond else 1
            value = solutionTemp[-1, getCompartments(newModel).index(x)]
            # print(f'{compartment}, {value}, {denom}')
            value = value / denom
            values[t][x] = value
        # print(f'{sum(values[t_spanOld[i]]):.2f} ', end='')

    if r0:
        print('R0 computation done\n')
    else:
        print('Rt computation done\n')

    return modelOld, newModel, solutionOld, t_spanOld, values


def computeR0(modelName: str, t_span_sim: tuple = (0, 100),
              sub_sim: float = 100, write: bool = False,
              overWrite: bool = False, whereToAdd: str = 'contact',
              printInit: bool = True, autoInfections: bool = False) -> dict:
    """Computes R0 associated with all contact nodes.
    However, if a variant is not present at start, R0 will be 0.
    This is because it was impossible at t=0 to know that variant would appear."""
    modelOld, newModel, solutionOld, _, values = \
        computeRt(modelName, (0, 0), 1, t_span_sim,
                  sub_sim, verification=False, write=write,
                  overWrite=overWrite, whereToAdd=whereToAdd,
                  printInit=printInit, r0=True, autoInfections=autoInfections)

    initialConds = solutionOld[0]
    return modelOld, newModel, initialConds, values[0]


def infs(model: dict, y0: dict, t: float, t0: float, whereToAdd: str = 'contact') -> dict:
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

                rateImpact = sum([y0[x] for x in v_r])
                contactImpact = sum([y0[x] for x in v_c])
                # Normally v_r and v_c should not be null. We can use both directly.
                param = getCoefForFlow(flow, t, t0)
                contactsFlow = param * rateImpact * contactImpact / N
                node = flow[whereToAdd]
                newInfections[node] += contactsFlow

    return newInfections


def infsScaled(model: dict, y0: dict, t: float, t0: float, whereToAdd: str = 'contact') -> dict:
    """Returns scaled incidences, sums to 1."""

    infections = infs(model, y0, t, t0, whereToAdd)
    weWant = getCompartments(model)

    sumInfections = sum([infections[node] for node in weWant])
    denom = sumInfections if sumInfections != 0 else 1

    scaledInfs = {key: infections[key] / denom for key in infections}

    return scaledInfs


def totInfs(model: dict, state: np.ndarray, t:float, t0: float) -> np.ndarray:
    """Returns total infected for a state."""

    if len(state.shape) > 1:
        raise Exception(
            f'Function can only be used on single state, not solution.')
    weWant = getCompartments(model)
    y0 = {comp: state[i] for i, comp in enumerate(weWant)}
    infections = infs(model, y0, t, t0)

    return sum([infections[comp] for comp in weWant])


def infCurve(model: dict, solution: np.ndarray, t_span: np.ndarray) -> np.ndarray:
    """Returns curve of total infected."""

    curve = np.array([totInfs(model, x, t_span[i], t_span[i]) for i, x in enumerate(solution)])
    return curve


def infCurveScaled(model: dict, solution: np.ndarray) -> np.ndarray:
    """Returns curve of total infected, scaled so max = 1."""

    curve = infCurve(model, solution)
    curve = curve / np.max(curve)
    return curve


def writeModel(newModel: dict, modelName: str, overWrite: bool = False) -> None:
    """Write model to file. This is useful to save modified models."""
    newFileName = modelName + '.json'
    print(f'Writing new model to file models/{newFileName}.')
    if not os.path.isfile(f'models/{newFileName}'):
        # File doesn't exist
        try:
            with open(f'models/{newFileName}', 'w') as file:
                json.dump(newModel, file, indent=4)
            print('Model written.')
        except:
            print('Problem when writing file.')
    else:
        # File exists already
        print('File name already exists.')
        if overWrite:
            print('Overwriting file.')
            try:
                with open(f'models/{newFileName}', 'w') as file:
                    json.dump(newModel, file, indent=4)
            except:
                print('Problem when writing file.')


def doesIntersect(curve: np.ndarray, value: int, eps=10**4):
    """Checks if curve intersects y = value."""

    newCurve = curve - value
    products = newCurve[1:] * curve[:-1]

    where = np.where(products < - eps)[0]

    return len(where) > 0
