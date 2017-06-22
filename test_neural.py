import neural_schrodinger as neural
import numpy as np
import matplotlib.pyplot as plt
from difflib import SequenceMatcher

def repopulate(survivors,numgenes):
    children = np.zeros(numgenes-len(survivors),dtype=np.dtype('S270'))
    for i in range(numgenes-len(survivors)):
        parent1,parent2 = np.random.choice(survivors,size=2,replace=False)
        children[i] = neural.breed_genes(parent1,parent2)
    return children

def check_convergence(best_gene,genes):
    match_ratios = np.empty(len(genes))
    for (i,gene) in enumerate(genes):
        match_ratios[i] = SequenceMatcher(None,best_gene,gene).ratio()
    return np.max(1-match_ratios)

def get_n_random_genes(numgenes,length):
    iterable = (neural.get_random_gene(length) for i in range(numgenes))
    return(np.fromiter(iterable,dtype=np.dtype('S270')))

numgenes=100
maxgenerations=1000
mass=1
net = neural.initialize_nn()
genes = get_n_random_genes(numgenes,270)

fitness_file = open('fitness_file.txt','a')

for i in range(maxgenerations):
    print("On generation "+str(i+1))
    points = neural.get_random_points(-6,6,100)
    survivors,survivor_fitnesses = neural.natural_selection(genes,net,50,points,mass)
    max_fitness = max(survivor_fitnesses)
    fittest_gene = survivors[np.argmax(survivor_fitnesses)]
    children = repopulate(survivors,numgenes)
    genes = np.concatenate((survivors,children))
    convergence = check_convergence(fittest_gene,genes)
    fitness_file.write(str(i+1)+'\t'+str(max_fitness)+'\t'+str(convergence)+'\n')
    fitness_file.flush()
    if convergence<0.05:
        newgenes = get_n_random_genes(numgenes,270)
        newgenes[0]=fittest_gene
        genes= newgenes
    if max_fitness>0.8:
        break

weights,energy = neural.gene_to_system(fittest_gene)
print("Energy of fittest gene is: "+str(energy))
net._setParameters(weights)
xs = np.linspace(-6,6,num=100)
phi = map(lambda x: neural.eigenfunction(net,x),xs)
plt.plot(xs,phi)
plt.show()
