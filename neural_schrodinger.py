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
    net = buildNetwork(1,6,2,hiddenclass=pys.SigmoidLayer)
    return(net)

def get_weights_by_layer(net, layername):
    connections = net.connections[net[layername]]
    weights = np.concatenate([connection.params for connection in connections])
    return(weights)

def weights_to_binary(net):
    weights = net.params
    binary_list = [float_to_bin(weight) for weight in weights]
    return("".join(binary_list))

def get_random_gene(gene_length):
    format_string =  "{:0"+str(gene_length)+"b}"
    gene = format_string.format(rnd.randint(0,2**gene_length-1))
    return(gene)

def system_to_gene(net,energy):
    weights = net.params
    binary_list = [float_to_bin(weight) for weight in weights]
    binary_list.append(float_to_bin(energy))
    return("".join(binary_list))

def gene_to_system(gene):
    num_length = 10
    array = np.fromstring(gene,dtype=np.dtype('S1'))
    split_array = np.split(array,27) 
    iterable = ("".join(subgene) for subgene in split_array)
    subgenes = np.fromiter(iterable,dtype=np.dtype('S10'))
    weights = map(bin_to_float,subgenes[:-1])
    energy = bin_to_float(subgenes[-1])
    return((weights,energy))

def natural_selection(genes,net,num_survivors,points,mass):
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
    newgene=""
    for i in range(len(gene1)):
        if rnd.uniform(0,1)<0.5:
            newgene+=gene1[i]
        else:
            newgene+=gene2[i]
    return newgene

def nn_array(net,array):
    vector = [np.array(element) for element in array]
    return(map(net.activate,vector))

def nn_first_derivative(net,x):
    inweights = get_weights_by_layer(net,'in')
    hiddenweights = get_weights_by_layer(net,'hidden0')
    iterable1 = (hiddenweight*sigmoidPrime(x)*inweight**2 for (inweight,hiddenweight) in zip(inweights,hiddenweights[0:6]))
    iterable2 = (hiddenweight*sigmoidPrime(x)*inweight**2 for (inweight,hiddenweight) in zip(inweights,hiddenweights[6:12]))
    output1_deriv = np.sum(np.fromiter(iterable1,dtype=float))
    output2_deriv = np.sum(np.fromiter(iterable2,dtype=float))
    return(output1_deriv,output2_deriv)


def nn_second_derivative(net,x):
    inweights = get_weights_by_layer(net,'in')
    hiddenweights = get_weights_by_layer(net,'hidden0')
    iterable1 = (hiddenweight*sigmoidPrime(sigmoidPrime(x))*inweight**2 for (inweight,hiddenweight) in zip(inweights,hiddenweights[0:6]))
    iterable2 = (hiddenweight*sigmoidPrime(sigmoidPrime(x))*inweight**2 for (inweight,hiddenweight) in zip(inweights,hiddenweights[6:12]))
    output1_deriv = np.sum(np.fromiter(iterable1,dtype=float))
    output2_deriv = np.sum(np.fromiter(iterable2,dtype=float))
    return(output1_deriv,output2_deriv)

def eigenfunction(net,x):
    (A,S)=net.activate(np.array([x]))
    return(A*sin(S))

def eigenfunction_second_deriv(net,x):
    (Aprime,Sprime) = nn_first_derivative(net,x)
    (Aprime2,Sprime2) = nn_second_derivative(net,x)
    (A,S) = net.activate(np.array([x]))
    second_deriv = (Aprime2*sin(S)+Aprime*Sprime*cos(S)+
                    (Aprime*Sprime+A*Sprime2)*cos(S)-
                    A*Sprime*Sprime*sin(S))
    return(second_deriv)

def hamiltonian(net,x,mass):
    hamiltonian = -(1./(2.*mass))*eigenfunction_second_deriv(net,x)+0.5*(x**2)*eigenfunction(net,x)
    return(hamiltonian)

def get_random_points(xmin,xmax,M):
    return(np.random.uniform(xmin,xmax,size=M))

def gene_fitness(gene,net,points,mass):
    weights,energy = gene_to_system(gene)
    net._setParameters(weights)
    energy_SSR_func = lambda x : (hamiltonian(net,x,mass)-energy*eigenfunction(net,x))**2
    normalization_func = lambda x : eigenfunction(net,x)**2
    numerator = np.sum(np.fromiter((energy_SSR_func(x) for x in points),dtype=float))
    denominator = np.sum(np.fromiter((normalization_func(x) for x in points),dtype=float))
    R = numerator/denominator
    return np.exp(-R)


def bin_to_float(binary):
    number = np.sum(np.fromiter((float(digit)*2.**(place) for digit,place in zip(binary,range(1,-7,-1))),dtype=float))
    number = number if int(binary[0])==0 else -number
    return number

def float_to_bin(number):
    if number==0:
        return "0"*10
    wrapped_value = number % 8
    binary = "0" if number >=0 else "1"
    for i in range(1,-7,-1):
        if wrapped_value>=(2.**i):
            wrapped_value-=(2.**i)
            binary+="1"
        else:
            binary+="0"
    return binary
    #hexvalue,sign = (float.hex(number),0) if number>0 else (float.hex(number)[1:],1)
    #mantissa, exp = int(hexvalue[4:6],16), int(hexvalue[18:])


"""
def float_to_bin(number):
    ""Modified from stackoverflow: https://stackoverflow.com/a/39378244""
    if number==0:
        return "0"*12
    hexvalue,sign = (float.hex(number),0) if number>0 else (float.hex(number)[1:],1)
    mantissa, exp = int(hexvalue[4:6],16), int(hexvalue[18:])
    if exp>3 or exp<-4:
        return "0"*12
    return "{}{:03b}{:08b}".format(sign,exp+4,mantissa)
    
def bin_to_float(binary):
    mantissa,exp = binary[4:12],int(binary[1:4],2)
    fraction = +sum([float(digit)*2.**(-(place+1)) for (place,digit) in enumerate(mantissa)])
    number = (1.+fraction)*2.**(exp-4.)
    signed_number = number if int(binary[0])==0 else -number
    return signed_number
"""
