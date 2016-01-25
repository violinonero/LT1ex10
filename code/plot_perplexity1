import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.image as mpimg

input_neuron = []
perplexity = []

with open('ppl_output.txt') as f:
	for line in f:
		if 'Embed' in line:
			input_neuron.append(line.split()[-1])
		if 'Perplexity' in line:
			perplexity.append(line.split()[-1])

input_neuron_dev = input_neuron[0:7]
perplexity_dev = perplexity[0:7]

input_neuron_test = input_neuron[7:]
perplexity_test = perplexity[7:]

plt.plot(input_neuron_dev, perplexity_dev, color='cornflowerblue', linewidth=2, label="Development Set")
plt.plot(input_neuron_test, perplexity_test, color='indianred',  linewidth=2, label="Test Set")

red_patch = mpatches.Patch(color='cornflowerblue', label="Development Set")
blue_patch = mpatches.Patch(color='indianred', label="Test Set")

plt.legend(handles=[red_patch, blue_patch], loc=1)
plt.xlabel('Input Neurons')
plt.ylabel('Perplexity')

plt.savefig('plot_perplexity.png')



