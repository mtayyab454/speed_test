import os
import numpy as np 
import matplotlib.pyplot as plt 



def parse(log_data, logs = ['RAM','SWAP','CPU','EMC_FREQ','GR3D_FREQ','thermal_GPU','thermal_CPU',
	                 'thermal_thermal','VDD_CPU_GPU_CV','VDD_SOC']):
	values = {i:[] for i in logs}
	for line in log_data:
		line = line.replace('\n','')
		for ix,section in enumerate(line.split(' ')):
			if section in logs:
				values[section].append(line.split(' ')[ix+1].replace('C','').replace('MB',''))
			elif section.split('@')[0] in ['GPU','CPU','thermal']:
				values['thermal_'+section.split('@')[0]].append(section.split('@')[1])
	return(values)

def convert_vals(data, logs = ['RAM','SWAP','CPU','EMC_FREQ','GR3D_FREQ','thermal_GPU','thermal_CPU',
	                 'thermal_thermal','VDD_CPU_GPU_CV','VDD_SOC']):
	numeric={i:[] for i in logs}
	for _id,val in zip(data.keys(),data.values()):
		for value in val:
			# print(value)
			if _id == 'CPU':
				numeric[_id].append([i for i in value.replace('[','').replace(']','').replace("'",'').split(',') if i!='off'])
			elif _id in ['thermal_GPU','thermal_CPU','thermal_thermal']:
				numeric[_id].append(float(value.replace('C','')))
			elif _id in ['RAM','SWAP','VDD_CPU_GPU_CV','VDD_SOC']:
				numeric[_id].append(value.split('/')[0])
	return(numeric)


log_files = ['logs/tegra_log_normal.txt', 'logs/tegra_log_normal.txt', 'logs/tegra_log_model.txt','logs/tegra_log_trt.txt']
log_data = [open(i,'r').readlines() for i in log_files]

val1 = parse(log_data[0])
value1 = convert_vals(val1)

val2 = parse(log_data[1])
value2 = convert_vals(val2)

val3 = parse(log_data[2])
value3 = convert_vals(val3)

val4 = parse(log_data[3])
value4 = convert_vals(val4)

##############################################################################################
                                  # PLOTING
##############################################################################################
# Thermal
cpu = [*value4['thermal_CPU']]
gpu = [*value4['thermal_GPU']]
total = [*value1['thermal_thermal'],*value2['thermal_thermal'],*value3['thermal_thermal']]
plt.plot(cpu, 'g', label='CPU')
plt.plot(gpu, 'b', label='GPU')
plt.plot(total, '--r', label='Total')
plt.xlabel('Time') 
plt.ylabel('Thermal(C)') 
plt.title('Thermal Profile') 
plt.legend()
plt.show()

# SWAP/Memory
ram = [*value4['RAM']]
swap = [*value4['SWAP']]
# plt.plot(range(len(swap)), [7764 for i in range(len(ram))], '--b', label='RAM_Max')
# plt.plot(range(len(ram)), [3882 for i in range(len(swap))], '--g', label='SWAP_Max')
plt.plot(range(len(swap)), swap, 'g',label='SWAP | 3882 Max')
plt.plot(range(len(ram)), ram, 'b',label='RAM | 7764 Max')

plt.xlabel('Time') 
plt.title('Memory Profile') 
plt.ylabel('Memory Usage(MB)') 
plt.legend()
plt.show()

# Power Consumption

vdd_gpu_cpu = value4['VDD_CPU_GPU_CV'] #[*value1['VDD_CPU_GPU_CV'],*value2['VDD_CPU_GPU_CV'],*value3['VDD_CPU_GPU_CV']]
vdd_soc = value4['VDD_SOC'] #[*value1['VDD_SOC'],*value2['VDD_SOC'],*value3['VDD_SOC']]
# plt.plot(range(len(vdd_soc)), [7764 for i in range(len(vdd_gpu_cpu))], '--b', label='vdd_gpu_cpu_Max')
# plt.plot(range(len(vdd_gpu_cpu)), [3882 for i in range(len(vdd_soc))], '--g', label='vdd_soc_Max')
plt.plot(range(len(vdd_soc)), vdd_soc, 'g',label='VDD_SOC | MilliWatt')
plt.plot(range(len(vdd_gpu_cpu)), vdd_gpu_cpu, 'b',label='VDD_CPU_GPU_CV | MilliWatt')
plt.xlabel('Time') 
plt.title('Power Profile') 
plt.ylabel('Power Usage(MilliWatt)') 
plt.legend()
plt.show()
















# log_files = ['tegra_log_model.txt' ,'tegra_log_normal.txt'] 
# log_data = [open(i,'r').readlines() for i in log_files]

# val1 = parse(log_data[0]) 
# value1 = convert_vals(val1) 
 
# val2 = parse(log_data[1]) 
# value2 = convert_vals(val2) 
