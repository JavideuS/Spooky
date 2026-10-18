import numpy as np
import time
# import neal # neal.SimulatedAnnealingSampler is depricated, instead use dwave.samplers.SimulatedAnnealingSampler
import dwave
import itertools
import dwave_networkx as dnx
import networkx as nx
import minorminer
import dimod
import dwave.preprocessing
from dwave.embedding import embed_bqm, unembed_sampleset
from dwave.embedding.chain_breaks import broken_chains
from dwave.embedding.chain_strength import uniform_torque_compensation
from dwave.system import DWaveSampler, EmbeddingComposite, AutoEmbeddingComposite, CutOffComposite, FixedEmbeddingComposite, LazyFixedEmbeddingComposite
from dwave.system import DWaveCliqueSampler
from dwave.system import LeapHybridCQMSampler
from dwave.samplers import SimulatedAnnealingSampler # neal.SimulatedAnnealingSampler is depricated, instead use dwave.samplers.SimulatedAnnealingSampler
try:
    simulated_sampler = SimulatedAnnealingSampler()
    import dwave_token
    import dwave_token_JSC
    # DWave_sampler = EmbeddingComposite(DWaveSampler(token=dwave_token.value))
    DWave_sampler = AutoEmbeddingComposite(DWaveSampler(token=dwave_token.value))
    DWave_sampler2 = AutoEmbeddingComposite(DWaveSampler(token=dwave_token.value, solver='Advantage2_system1'))
    DWave_sampler2e = AutoEmbeddingComposite(DWaveSampler(token=dwave_token_JSC.value, solver='Advantage2_system2', region="eu-central-1"))
    clique_sampler = DWaveCliqueSampler(token=dwave_token.value)
    clique_sampler2 = DWaveCliqueSampler(token=dwave_token.value, solver='Advantage2_system1')
    clique_sampler2e = DWaveCliqueSampler(token=dwave_token_JSC.value, solver='Advantage2_system2', region="eu-central-1")
    hybrid_sampler = LeapHybridCQMSampler(token=dwave_token.value)
    lazy_sampler = LazyFixedEmbeddingComposite(DWaveSampler(token=dwave_token.value))
    Advantage1_graph = dnx.pegasus_graph(16, node_list=DWaveSampler(token=dwave_token.value).nodelist, edge_list=DWaveSampler(token=dwave_token.value).edgelist)
    Advantage2_graph = dnx.zephyr_graph(12, 4, node_list=DWaveSampler(token=dwave_token.value).nodelist, edge_list=DWaveSampler(token=dwave_token.value).edgelist)
except Exception as e:
    print(e)
    if e.__str__() == "name 'dwave_token' is not defined" or e.__str__() == "No module named 'dwave_token'":
        print('D-Wave Token unspecified. Can not choose any quantum solver')

__all__ = [
    'solution_from_sampleset',
    'anneal_QUBO',
    'logical_to_physical_bqm',
    'physical_to_logical_sampleset',
    'dict_QUBO_to_array',
    'roof_duality',
    'replace_fixed_vars'    
]

embedding_timeout = 2

def solver_to_sampler(solver, QUBO, embedding=None, print_info:bool=False):
    solver = solver.lower()
    assert solver in ['sa', 'simulated annealing', 'simulated_annealing', 'annealing', 'dwave', 'qa', 'quantum_annealing', 'initial_clique', 'clique', 'lazy', 'annealing2', 'dwave2', 'qa2', 'quantum_annealing2', 'initial_clique2', 'clique2', 'sa_clique', 'sa_clique2', 'sa_search', 'sa_search2', 'qa2e', 'clique2e', 'sa_search2e', 'sa_clique2e'], 'Unknown solver specified'
    if solver == 'clique':
        sampler = clique_sampler
    elif solver == 'clique2':
        sampler = clique_sampler2
    elif solver == 'clique2e':
        sampler = clique_sampler2e
    elif solver in ['sa', 'simulated_annealing', 'sa_clique', 'sa_clique2', 'sa_search', 'sa_search2', 'sa_search2e', 'sa_clique2e']:
        sampler = simulated_sampler
    elif solver == 'lazy':
        sampler = lazy_sampler
        sampler.embedding_parameters = {'timeout': embedding_timeout}.items()
    elif solver[-1] == '2':
        sampler = DWave_sampler2
    elif solver.endswith('2e'):
        sampler = DWave_sampler2e
        sampler.embedding_parameters = {'timeout': embedding_timeout}.items()
    else:
        sampler = DWave_sampler
        sampler.embedding_parameters = {'timeout': embedding_timeout}.items()
    if solver[-1] == '2':
        graph = dnx.zephyr_graph(12, 4, node_list=DWave_sampler2.child.nodelist, edge_list=DWave_sampler2.child.edgelist)
    elif solver.endswith('2e'):
        graph = dnx.zephyr_graph(12, 4, node_list=DWave_sampler2e.child.nodelist, edge_list=DWave_sampler2e.child.edgelist)
    elif sampler != simulated_sampler or solver in ['sa_clique', 'sa_search']:
        graph = dnx.pegasus_graph(16, node_list=DWave_sampler.child.nodelist, edge_list=DWave_sampler.child.edgelist)
    else:
        graph = None
    if solver[:len('initial_clique')] == 'initial_clique':
        sampler.embedding_parameters = {
            'timeout': embedding_timeout, 
            'initial_chains': minorminer.busclique.find_clique_embedding(len(QUBO), graph)
        }.items()
    return(sampler, graph)

def solution_from_sampleset(sampleset):
    if len(sampleset) == 0:
        return np.array([]), None
    objective = sampleset.first.energy
    is_sorted = lambda a: (a == np.arange(len(a))).all()
    assert is_sorted(list(sampleset.first.sample.keys())), 'Variables not sorted properly.'
    solution = np.array(list(sampleset.first.sample.values()))
    return solution, objective

def anneal_QUBO(QUBO, solver='SA', shots=100, embedding=None, annealing_parameters=None, back_up_clique_embedding=False, **kwargs):
    '''solves a QUBO by a D-Wave annealing method.
    Inputs:
        QUBO: tuple of a QUBO-matrix as np.ndarray and an offset as float
        solver: str, specifying the ampler to be used. Use "SA" for simulated annealing, "clique" for the clique sampler, "QA" ("initial clique") for searching for an embedding with the AutoEmbeddingComposite (with clique embedding as initial guess). To use the Advantage2 system, add 2 to the end of the argument. Default is "SA".
        shots: int, number of reads in the annealing process. Default is 100.
        embedding: dict, mapping variables to the hardware graph. You can also specify a part of an embedding to search for the remainder. This does not work with the clique or simulated annealing sampler. Default is {}.
        annealing_parameters: dict, parameters passed to the sample method if a quantum annealing solver is chosen. Please do not include key 'embedding_parameters', as this one will be specified by other arguments. To vary the annealing time, use key "annealing_time". Default is {}.
        back_up_clique_embedding: bool, if True, clique embedding is attempted if finding a manual embedding fails. This should only be relevant for solver in ["QA", "QA2", "initial_clique", "initial_clique2"]. Default is False.
    Outputs:
        objective (float): objective of best found solution of the QUBO
        solution (array): binary solution vector of best found solution
        computation_time (float) wall-clock time of sampling process, including embedding and other overheads
        sampleset: dimod sampleset object containing all samples, energies, number of occurences, as well as timing and embedding information (check sampleset.info['embedding_context'])
        broken_chain_fraction (array): Relative number of broken chains for each sample
    '''
    
    if embedding is None:
        embedding = {}

    if annealing_parameters is None:
        annealing_parameters = {}

    if isinstance(QUBO, tuple):
        QUBO, C = QUBO
    elif isinstance(QUBO, np.ndarray):
        C = 0
    else:
        raise TypeError('Please provide a suitible input for the QUBO argument.')
    sampler, graph = solver_to_sampler(solver, QUBO)
    bqm = dimod.BinaryQuadraticModel.from_qubo(QUBO, C)

    if 'preprocessing' in kwargs and 'roof_duality' in kwargs['preprocessing']:
        if 'roof_strict' in kwargs:
            strict = kwargs['roof_strict']
        else:
            strict = True

        lb, fixed_vars = roof_duality(bqm, strict=strict)
        bqm = replace_fixed_vars(bqm, fixed_vars)
    else:
        fixed_vars = {}

    if 'num_reads' not in annealing_parameters:
        annealing_parameters['num_reads'] = shots

    a = time.time()
    if isinstance(sampler, SimulatedAnnealingSampler) and len(bqm.variables) == 0: # to omit warnings of zero variables with SA
        sampleset = generate_empty_sampleset(bqm.offset, shots)
        broken_chain_fraction = None
    elif solver[:len('sa_clique')].lower() == 'sa_clique' or solver[:len('sa_search')].lower() == 'sa_search':
        chain_strength = annealing_parameters.get('chain_strength', None)
        if solver[:len('sa_clique')].lower() == 'sa_clique':
            embedding = minorminer.busclique.find_clique_embedding(bqm.variables, graph)
        else:
            source_edgelist = list(itertools.chain(bqm.quadratic, ((v, v) for v in bqm.linear)))
            embedding = minorminer.find_embedding(source_edgelist, graph, timeout=embedding_timeout)
        if not embedding and bqm.num_variables:
            raise ValueError("Cannot embed given BQM (size {})".format(len(bqm.variables)))
        physical_bqm = logical_to_physical_bqm(logical_bqm=bqm, embedding=embedding, target_graph=graph, chain_strength=chain_strength)
        physical_sampleset = sampler.sample(
            physical_bqm,
            **annealing_parameters
        )
        sampleset, num_broken_chains = physical_to_logical_sampleset(physical_sampleset, embedding, bqm)
        sampleset.info["embedding_context"] = {
            "embedding": embedding,
            "chain_strength": uniform_torque_compensation(bqm, embedding) if chain_strength is None else chain_strength,
        }
        broken_chain_fraction = np.array(num_broken_chains) / len(sampleset.variables)
    elif isinstance(sampler, SimulatedAnnealingSampler):
        sampleset = sampler.sample(
            bqm,
            **annealing_parameters
        )
        broken_chain_fraction = None
    elif isinstance(sampler, DWaveCliqueSampler):
        sampleset = sampler.sample(
            bqm,
            return_embedding=True,
            **annealing_parameters,
        )
        broken_chain_fraction = sampleset.record.chain_break_fraction
    else:
        try:
            sampleset = sampler.sample(
                bqm,
                embedding_parameters={
                    **dict(sampler.embedding_parameters),
                    'fixed_chains': embedding
                    },
                return_embedding=True,
                **annealing_parameters,
            )
            # Check if a valid embedding was found
            if not sampleset:
                raise ValueError("No embedding found within the timeout.")
        except (ValueError, TimeoutError) as e:
            if back_up_clique_embedding:
                print(e)
                print('Reverting to Clique embedding')
                # Fallback to clique embedding if no embedding found
                sampleset = sampler.sample(
                    bqm,
                    embedding_parameters={
                        **dict(sampler.embedding_parameters),
                        'fixed_chains': minorminer.busclique.find_clique_embedding(bqm.variables, graph)
                    },
                    return_embedding=True,
                    **annealing_parameters,
                )
            else:
                raise
        broken_chain_fraction = sampleset.record.chain_break_fraction
    
    # adjust sampleset to account for fixed variables
    sampleset = dimod.SampleSet.append_variables(sampleset, fixed_vars, True)

    computation_time = time.time() - a
    solution, objective = solution_from_sampleset(sampleset)
    return objective, solution, computation_time, sampleset, broken_chain_fraction

def logical_to_physical_bqm(logical_bqm, embedding, target_graph='Advantage1', chain_strength=None, verbose=False):
    '''Converts a bqm in logical variables to the embedded problem in physical qubits
    Inputs:
        logical_bqm (dimod.BinaryQuadraticModel): bqm in logical variables
        embedding (dict): dict, mapping variables to the hardware graph.
        target_graph (str, dict, nx.Graph): string in ['Advantage1', 'Advantage2'] to load the current physical hardware graphs of the D-Wave annealers, or dict or nx.Graph to specify them. Default is 'Advantage1'
        chain_strength (float/mapping/callable, optional): specify the coupling strength of chains in the embedding. Default is None, which will trigger the standard argument and set the chain strengths according to dwave.embedding.chain_strength.uniform_torque_compensation. 
        verbose (bool): bool, for printing info. Default is False.
    Output:
        embedded_bqm (dimod.BinaryQuadraticModel): embedded bqm of physical variables
    '''
    if target_graph == 'Advantage1':
        target_graph = Advantage1_graph
    elif target_graph == 'Advantage2':
        target_graph = Advantage2_graph
    else:
        assert isinstance(target_graph, (dict, nx.Graph)), 'Specify target graph of sufficient type.'
    target_nodes = target_graph.nodes if isinstance(target_graph, nx.Graph) else target_graph.keys()
    assert {val for vals in embedding.values() for val in vals} <= set(target_nodes)  , 'Embedding not matching with target graph!'
    embedded_bqm = embed_bqm(logical_bqm, embedding=embedding, target_adjacency=target_graph, chain_strength=chain_strength)
    if verbose:
        print('QUBO embedded in {} physical bits'.format(len(embedded_bqm.variables)))
    return embedded_bqm

def physical_to_logical_sampleset(physical_sampleset, embedding, logical_bqm):
    '''Given an embedding, transforms a sampleset of an embedded QUBO to a sampleset of the corresponding logical QUBO.
    This is basically reversing the embedding.
    Inputs:
        physical_sampleset (sampleset): sampleset of physical_QUBO
        embedding (dict): dict, mapping variables to the hardware graph
        logical_bqm (dimod.BinaryQuadraticModel): The output logical_sampleset will correspond to the solution of this bqm.
    Outputs:
        logical_sampleset (sampleset): sampleset of logical_QUBO
        num_broken_chains (list): list of how many chains broke in each sample
    '''
    logical_sampleset = unembed_sampleset(physical_sampleset, embedding, source_bqm=logical_bqm)
    # calculate number of broken chains
    chains = [embedding[v] for v in logical_bqm.variables]
    num_broken_chains = [np.count_nonzero(sample) for sample in broken_chains(physical_sampleset, chains)]
    return logical_sampleset, num_broken_chains

def dict_QUBO_to_array(dict_QUBO):
    '''Transforms a QUBO in dictionary form to a QUBO in array form. This does not include an additive constant.
    Input:
        dict_QUBO (dict[(var, var): float]): QUBO in dictionary form, with pairs of variables as keys and QUBO coefficients as values.
    Outputs:
        QUBO_array (nd.array): QUBO in quadratic array form
        variables (list): list of variables to understand ordering of QUBO_array entries
    '''
    variables = sorted(set(var for pair in dict_QUBO.keys() for var in pair))
    # Initialize the quadratic array with zeros
    size = len(variables)
    QUBO_array = np.zeros((size, size), dtype=float)
    # Fill the quadratic array with coefficients from the dictionary
    for (var1, var2), value in dict_QUBO.items():
        i = variables.index(var1)
        j = variables.index(var2)
        QUBO_array[tuple(sorted((i, j)))] = value
    return QUBO_array, variables

def roof_duality(QUBO, strict=True):
    '''
    Given a QUBO array (with or w.o. constant shift) or a BQM, 
        calculates a valid lower bound and variables fixes based on the roof duality.
    Input:
        QUBO (ndarray | (ndarray, float) | BinaryQuadraticModel)
        strict (bool): wether only variables should be fixed, 
            which have the same value in all optimal solutions
    Outputs:
        lb (float): lower bound derived by the roof duality
        fixed_vars (dict): dictionary with fixed variables and their values
    '''
    if not isinstance(QUBO, dimod.BinaryQuadraticModel):
        if isinstance(QUBO, tuple):
            QUBO, C = QUBO
        elif isinstance(QUBO, np.ndarray):
            C = 0
        else:
            raise TypeError('Please provide a suitible input for the QUBO argument.')

        bqm = dimod.BinaryQuadraticModel.from_qubo(QUBO, C)
    else:
        bqm = QUBO

    lb, fixed_vars = dwave.preprocessing.roof_duality(bqm, strict=strict)

    return lb, fixed_vars

def replace_fixed_vars(QUBO, fixed_vars):
    '''
    Modifies the given QUBO to arrange for the fixed variables, 
        thereby reducing it's size.
    Input:
        QUBO (ndarray | (ndarray, float) | BinaryQuadraticModel)
        fixed_vars (dict): dictionary with fixed variables and their values
    Outputs:
        QUBO ((ndarray, float) | BinaryQuadraticModel): modified QUBO
    '''
    if not isinstance(QUBO, dimod.BinaryQuadraticModel):
        if isinstance(QUBO, tuple):
            QUBO, C = QUBO
        elif isinstance(QUBO, np.ndarray):
            C = 0
        else:
            raise TypeError('Please provide a suitible input for the QUBO argument.')

        new_QUBO = np.array([[QUBO[i][j] for j in range(len(QUBO)) if j not in fixed_vars] for i in range(len(QUBO)) if i not in fixed_vars])
        k = 0
        for i in range(len(QUBO)):
            if i not in fixed_vars:
                new_QUBO[k][k] += sum((QUBO[k][j]) * fixed_vars[j] for j in fixed_vars)
                k += 1
        
        new_C = C + sum((QUBO[i][j]) * fixed_vars[i] * fixed_vars[j] for i in fixed_vars for j in fixed_vars)

        return (new_QUBO, new_C)

    QUBO.fix_variables(fixed_vars)

    return QUBO

def generate_empty_sampleset(C, shots):
    """Generate an empty sampleset to add fixed variables to."""
    dtype = np.dtype([
        ('sample', '<f8', (0,)),         # 1D array of floats for samples
        ('energy', '<f8'),               # Energy as a float
        ('num_occurrences', '<i8')       # Occurrences as an integer
    ])
    record = np.recarray((1,), dtype=dtype)
    record.energy = C
    record.num_occurrences = shots

    variables = dimod.variables.Variables()

    info = {
        'timing': {'preprocessing_ns': 0, 'sampling_ns': 0, 'postprocessing_ns': 0},
        'embedding_context': {'embedding': {}}
    }

    vartype = dimod.Vartype.BINARY
    sampleset = dimod.sampleset.SampleSet(record, variables, info, vartype)

    return sampleset