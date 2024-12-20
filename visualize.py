import sys
import genotypes
import utils
from graphviz import Digraph

def plot_cell_from_alphas(alphas, filename):
	g = Digraph(
		format='png',
		edge_attr=dict(fontsize='20', fontname="times"),
		node_attr=dict(style='filled', shape='rect', align='center', fontsize='20', height='0.5', width='0.5', penwidth='2', fontname="times"),
		engine='dot')
	g.body.extend(['rankdir=LR'])
	g.node("c_{k-2}", fillcolor='darkseagreen2')
	g.node("c_{k-1}", fillcolor='darkseagreen2')

	#print("Plotting cell from alpha")
	genotype = genotypes.PRIMITIVES
	num = alphas.size()[0]
	ops = utils.derive_architecture(alphas)
	#print("\nops: {}\n\n".format(ops))

	i = 1
	while i <= num:
		tmp = ((i + 1) * (i + 2)) / 2
		#print("tmp", tmp)
		if ((tmp - 1) == num):
			steps = i
			#print("steps: {}".format(steps))
			break
		i += 1

	nodes = {}
	nodes[0] = 'c_{k-2}'
	nodes[1] = 'c_{k-1}'
	for i in range(steps):
		# For creating nodes including c_k-1 and c_k-1 for previous 2 nodes
		g.node(str(i), fillcolor='lightblue')
		nodes[i + 2] = str(i)

	#print("nodes: ", nodes)
	offset = 0
	ops_nodes = []
	for i in range(steps):
		for j in range(2 + i):
			#if ops[offset + j][0] == 'none':
			#	continue
			ops_nodes.append((ops[offset + j], j))
			#print("j: {}, i+ 2: {}, node1: {}, node2: {}".format(j, i + 2, nodes[j], nodes[i + 2]))
			g.edge(nodes[j], nodes[i + 2], label = ops[offset + j][0] + " : "+ str(ops[offset + j][1].data), fillcolor="gray")
		offset += j + 1
	#print("ops with nodes: ", ops_nodes)

	g.node("c_{k}", fillcolor='palegoldenrod')
	for i in range(steps):
		g.edge(str(i), "c_{k}", fillcolor="gray")

	g.render(filename, view=True)

def plot(genotype, filename, view = True):
	g = Digraph(
		format='png',
		edge_attr=dict(fontsize='20', fontname="times"),
		node_attr=dict(style='filled', shape='rect', align='center', fontsize='20', height='0.5', width='0.5', penwidth='2', fontname="times"),
		engine='dot')
	g.body.extend(['rankdir=LR'])

	g.node("c_{k-2}", fillcolor='darkseagreen2')
	g.node("c_{k-1}", fillcolor='darkseagreen2')
	assert len(genotype) % 2 == 0
	steps = len(genotype) // 2
	#print("len(genotype): {}, genotype = {}".format(len(genotype), genotype))
	#print("steps: {}".format(steps))

	for i in range(steps):
		# For creating nodes including c_k-1 and c_k-1 for previous 2 nodes
		g.node(str(i), fillcolor='lightblue')
	
	for i in range(steps):
		for k in [2*i, 2*i + 1]:
			op, j = genotype[k]
			if j == 0:
				u = "c_{k-2}"
			elif j == 1:
				u = "c_{k-1}"
			else:
				u = str(j-2)
			v = str(i)
			g.edge(u, v, label=op, fillcolor="gray")

	g.node("c_{k}", fillcolor='palegoldenrod')
	for i in range(steps):
		g.edge(str(i), "c_{k}", fillcolor="gray")
	
	g.render(filename, view = view)


if __name__ == '__main__':
	if len(sys.argv) != 2:
		print("usage:\n python {} ARCH_NAME".format(sys.argv[0]))
		sys.exit(1)

	genotype_name = sys.argv[1]
	try:
		genotype = eval('genotypes.{}'.format(genotype_name))
	except AttributeError:
		print("{} is not specified in genotypes.py".format(genotype_name)) 
		sys.exit(1)

	plot(genotype.normal, "normal")
	plot(genotype.reduce, "reduction")

