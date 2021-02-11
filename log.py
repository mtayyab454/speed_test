import os
import numpy as np 
import matplotlib.pyplot as plt

def parse_power(power_lines, power_log = ['GPU', 'CPU', 'SOC', 'CV', 'VDDRQ', 'SYS5V']):
	values = {'GPU':[[],[]], 'CPU':[[],[]], 'SOC':[[],[]], 'CV':[[],[]], 'VDDRQ':[[],[]], 'SYS5V':[[],[]]}
	for pl in power_lines:
		sections = pl.split(' ')
		for id, sc in enumerate(sections):
			if sc in power_log:
				text_values = sections[id+1].split('/')
				values[sc][0].append(int(text_values[0]))
				values[sc][1].append(int(text_values[1]))

	return values

def parse(log_data):

	power_lines = []
	for line in log_data:
		line = line.replace('\n','')
		sections = line.split('Tboard@')
		power_lines.append(sections[1])

	values = parse_power(power_lines)

	return(values)

log_files = ['tegra_log_normal.txt']
log_data = [open(i,'r').readlines() for i in log_files]

values = parse(log_data[0])

vdd_gpu = values['GPU'][0]
vdd_cpu = values['CPU'][0]
plt.plot(range(len(vdd_gpu)), vdd_gpu, 'g',label='GPU power | MilliWatt')
plt.plot(range(len(vdd_cpu)), vdd_cpu, 'b',label='CPU power | MilliWatt')
plt.xlabel('Time')
plt.title('Power Profile')
plt.ylabel('Power Usage(MilliWatt)')
plt.legend()
plt.show()
