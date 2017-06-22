import numpy as np
import pybrain.structure as pys
from pybrain.tools.shortcuts import buildNetwork
from pybrain.tools.functions import sigmoidPrime
from math import sin,cos,exp
import random as rnd
from multiprocessing import Pool
import functools
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

def initialize_nn():
    """Initializes a neural network using PyBrain with 6 hidden layers, 1 input layerm and 2 output layers.
    """
    net = buildNetwork(1,6,2,hiddenclass=pys.SigmoidLayer)
    return(net)

def get_weights_by_layer(net, layername):
    """Get the weights of the layer specified by layername of a PyBrain neural network net.

    Parameters
    ----------
    net : PyBrain FeedForwardNetwork
        A feed forward neural network created using PyBrain
    layername : string
        A string specifying the name of the layer that one wishes to get the weights of.

    Returns
    -------
    weights : numpy array of floats
        A Numpy array of floats containing the weights of the specified layer.
    """
    connections = net.connections[net[layername]]
    weights = np.concatenate([connection.params for connection in connections])
    return(weights)

def weights_to_binary(net):
    """Converts the weights of a PyBrain neural network to the binary encoding specified by float_to_bin.

    Parameters
    ----------
    net : PyBrain FeedForwardNetwork

    Returns
    -------
    gene : string
        A string that corresponds to the binary encoding of the neural network which corresponds to the binary encoding of the neural network weights.
    """
    weights = net.params
    binary_list = [float_to_bin(weight) for weight in weights]
    return("".join(binary_list))

def get_random_gene(gene_length):
    """Returns a random gene which corresponds to a random binary sequence of length specified by gene_length.

    Parameters
    ----------
    gene_length : int
        Length og the random gene to generate

    Returns
    -------
    gene : string
        A string representing a random gene of length gene_length
    """
    format_string =  "{:0"+str(gene_length)+"b}"
    gene = format_string.format(rnd.randint(0,2**gene_length-1))
    return(gene)

def system_to_gene(net,energy):
    """Converts a system specified by the given PyBrain neural network and energy to a binary encoding specified by float_to_bin.

    Parameters
    ----------
    net : PyBrain FeedForwardNetwork
        A PyBrain neural network that corresponds to the wavefunction guess
    energy : float
        The guessed energy that corresponds to the given neural network

    Returns
    -------
    string
        A string which specifies the binary encoding of the given system.
    """
    weights = net.params
    binary_list = [float_to_bin(weight) for weight in weights]
    binary_list.append(float_to_bin(energy))
    return("".join(binary_list))

def gene_to_system(gene):
    """Converts a gene to a system which corresponds to a set of neural network weights and an energy.

    Parameters
    ----------
    gene : string
        A string representing the binary encoding of the system

    Returns
    -------
    weights,energy : (float list,float)
        A tuple containing the weights and energy of the system.
    """
    num_length = 10
    array = np.fromstring(gene,dtype=np.dtype('S1'))
    split_array = np.split(array,27) 
    iterable = ("".join(subgene) for subgene in split_array)
    subgenes = np.fromiter(iterable,dtype=np.dtype('S10'))
    weights = map(bin_to_float,subgenes[:-1])
    energy = bin_to_float(subgenes[-1])
    return((weights,energy))

def natural_selection(genes,net,num_survivors,points,mass):
    """Applies "natural selection" to a list of genes using the tournament methodology.  
    In this methodology genes are pitted against each other in pairs and the gene with the greatest fitness wins.
    This continues until the specified amount of genes are obtained.

    Parameters
    ----------
    genes : list of strings
        A list of strings each of which corresponds to a gene.
    net : PyBrain FeedForwardNetwork
        The neural network used to approximate the wavefunction
    num_survivors : int
        The number of survivors that are desired to be left.
    points : array of floats
        A set of points one wishes to evaluate the genes fitness on. 
    mass : float
        The reduced mass used in the Schrodinger equation to evaluate fitness.

    Returns
    -------
    survivors, survivor_fitnesses : (list of strings, list of floats)
        A list of survivor genes and their associated fitnesses.
    """
    survivors = np.zeros(num_survivors,dtype=np.dtype('S270'))
    survivor_fitnesses = np.zeros(num_survivors)
    fitness_func = functools.partial(gene_fitness,net=net,points=points,mass=mass)
    with ThreadPoolExecutor(max_workers=5) as executor:
        result = executor.map(fitness_func,genes)
    fitnesses=list(result)
    for i in range(num_survivors):
        gene1,gene2 = np.random.choice(np.arange(len(genes)),replace=False,size=2)
        #fitness1,fitness2 = gene_fitness(gene1,net,points,mass), gene_fitness(gene2,net,points,mass)
        selected_gene = gene1 if fitnesses[gene1]>fitnesses[gene2] else gene2
        survivors[i] = genes[selected_gene]
        survivor_fitnesses[i]=fitnesses[selected_gene]
    return((survivors,survivor_fitnesses))

def breed_genes(gene1,gene2):
    """Mates two genes using the uniform crossover algorithm.
    
    Parameters
    ----------
    gene1 : string
        The string representing the first gene to mate
    gene2 : string
        The string representing the second gene to mate.

    Returns
    -------
    newgene : string
        The child of the two mated strings
    """
    newgene=""
    for i in range(len(gene1)):
        if rnd.uniform(0,1)<0.5:
            newgene+=gene1[i]
        else:
            newgene+=gene2[i]
    return newgene

def nn_array(net,array):
    """Apply the neural network over an array.

    Parameters
    ----------
    net : PyBrain FeedForwardNetwork
        Neural network to apply over an array.
    array : array of floats
        Array of floats to activate the neural network over

    Returns
    -------
    array of floats
        The output of the neural network for each element of the array
    """
    vector = [np.array(element) for element in array]
    return(map(net.activate,vector))

def nn_first_derivative(net,x):
    """The first derivative of a feedforward neural network with one hidden sigmoid layer.

    Parameters
    ----------
    net : PyBrain FeedForwardNetwork
        The neural network one wishes to get the first derivative of.
    x : float
        The location x where one wants the derivative

    Returns
    -------
    output1_derivative, output2_derivative : (float,float)
        The first derivative of the first and second output of the neural network.
    """
    inweights = get_weights_by_layer(net,'in')
    hiddenweights = get_weights_by_layer(net,'hidden0')
    iterable1 = (hiddenweight*sigmoidPrime(x)*inweight**2 for (inweight,hiddenweight) in zip(inweights,hiddenweights[0:6]))
    iterable2 = (hiddenweight*sigmoidPrime(x)*inweight**2 for (inweight,hiddenweight) in zip(inweights,hiddenweights[6:12]))
    output1_deriv = np.sum(np.fromiter(iterable1,dtype=float))
    output2_deriv = np.sum(np.fromiter(iterable2,dtype=float))
    return(output1_deriv,output2_deriv)


def nn_second_derivative(net,x):
    """The second derivative of a feedforward neural network with one hidden sigmoid layer.

    Parameters
    ----------
    net : PyBrain FeedForwardNetwork
        The neural network one wishes to get the first derivative of.
    x : float
        The location x where one wants the derivative

    Returns
    -------
    output1_derivative, output2_derivative : (float,float)
        The second derivative of the first and second output of the neural network.
    """
    inweights = get_weights_by_layer(net,'in')
    hiddenweights = get_weights_by_layer(net,'hidden0')
    iterable1 = (hiddenweight*sigmoidPrime(sigmoidPrime(x))*inweight**2 for (inweight,hiddenweight) in zip(inweights,hiddenweights[0:6]))
    iterable2 = (hiddenweight*sigmoidPrime(sigmoidPrime(x))*inweight**2 for (inweight,hiddenweight) in zip(inweights,hiddenweights[6:12]))
    output1_deriv = np.sum(np.fromiter(iterable1,dtype=float))
    output2_deriv = np.sum(np.fromiter(iterable2,dtype=float))
    return(output1_deriv,output2_deriv)

def eigenfunction(net,x):
    """Converts the two outputs of the neural network to the value of the wavefunction at x.

    Parameters
    ----------
    net : PyBrain FeedForwardNetwork
        The neural network that represents the wavefunction.
    x : float
        The location at which one wants to evaluate the wavefunction.

    Returns
    -------
    wavefunction : float
        The value of the wavefunction at x
    """
    (A,S)=net.activate(np.array([x]))
    return(A*sin(S))

def eigenfunction_second_deriv(net,x):
    """Calculates the second derivative of the wavefunction represented by net.

    Parameters
    ----------
    net : PyBrain FeedForwardNetwork
        The neural network representing the wavefunction.
    x : float
        The location at which one wishes to evaluate the wavefunction.
    """
    (Aprime,Sprime) = nn_first_derivative(net,x)
    (Aprime2,Sprime2) = nn_second_derivative(net,x)
    (A,S) = net.activate(np.array([x]))
    second_deriv = (Aprime2*sin(S)+Aprime*Sprime*cos(S)+
                    (Aprime*Sprime+A*Sprime2)*cos(S)-
                    A*Sprime*Sprime*sin(S))
    return(second_deriv)

def hamiltonian(net,x,mass):
    """Calculates the hamiltonian of a harmonic oscillator of reduced mass "mass" applied to the wavefunction represented by net.
    
    Parameters
    ----------
    net : PyBrain FeedForwardNetwork
        The neural network that represents the wavefunction one wishes to apply the hamiltonian to.
    x : float
        The location at which one wishes to get the value of the hamiltonian
    mass : float
        The reduced mass of the harmonic oscillator

    Returns
    -------
    float
        The value of the hamiltonian applied to the wavefunction
    """
    hamiltonian = -(1./(2.*mass))*eigenfunction_second_deriv(net,x)+0.5*(x**2)*eigenfunction(net,x)
    return(hamiltonian)

def get_random_points(xmin,xmax,M):
    """Gets n random points between xmin and xmax used for the random point evaluation method.

    Parameters
    ----------
    xmin : float
        Minimum value that the points can take.
    xmax : float
        Maximum value that the points can take.
    M : int
        The number of points to generate.
    
    Returns
    -------
    array of floats
        An array of M random points.
    """
    return(np.random.uniform(xmin,xmax,size=M))

def gene_fitness(gene,net,points,mass):
    """Calculates the fitness of the gene based on the difference between the energy and 
    the hamiltonian applied to the wavefunction squared evaluated at the points specified by points.

    Parameters
    ----------
    gene : string
        The gene with which one wishes to get the fitness of.
    net : PyBrain FeedForwardNetwork
        The neural network used to represent the wavefunction.
    points : array of floats
        An array of random points with which one wishes to evaluate the hamiltonian.
    mass : float
        The reduced mass of the harmonic oscillator system.

    Returns
    -------
    float
        The fitness of the gene.
    """
    weights,energy = gene_to_system(gene)
    net._setParameters(weights)
    energy_SSR_func = lambda x : (hamiltonian(net,x,mass)-energy*eigenfunction(net,x))**2
    normalization_func = lambda x : eigenfunction(net,x)**2
    numerator = np.sum(np.fromiter((energy_SSR_func(x) for x in points),dtype=float))
    denominator = np.sum(np.fromiter((normalization_func(x) for x in points),dtype=float))
    R = numerator/denominator
    return np.exp(-R)


def bin_to_float(binary):
    """Converts a binary number encoded with fixed point format.  Here the first digit is reserved for sign and the rest are binary places starting with 2^1 and decreasing.
    Here we use ten bits to encode a number.

    Parameters
    ----------
    binary : string
        A string of ten digits representing the number in binary fixed point format.

    Returns
    -------
    number : float
        A float that corresponds to the fixed point binary format.
    """
    number = np.sum(np.fromiter((float(digit)*2.**(place) for digit,place in zip(binary[1:],range(1,-9,-1))),dtype=float))
    number = number if int(binary[0])==0 else -number
    return number

def float_to_bin(number):
    """Converts a floating point number to a ten bit fixed point format as specified in bin_to_float.

    Parameters
    ----------
    number : float
        The floating point number to convert.

    Returns
    -------
    binary : string
        A string representing the binary encoding of the floating point number in ten bit fixed point format.
    """
    if number==0:
        return "0"*10
    wrapped_value = abs(number) % 8
    binary = "0" if number >=0 else "1"
    for i in range(1,-9,-1):
        if wrapped_value>=(2.**i):
            wrapped_value-=(2.**i)
            binary+="1"
        else:
            binary+="0"
    return binary

