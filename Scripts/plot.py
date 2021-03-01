import matplotlib.pyplot as plt

x = [1, 2, 3, 4, 5, 10, 20, 30, 40, 50, 100, 200, 300, 400, 500, 1000]
y_cpu = [0.04226, 0.04858, 0.05329, 0.05984, 0.08944, 0.08957, 0.14352, 0.18972, 0.24725, 0.29268, 0.55317, 1.03589, 1.54234, 2.09681, 2.51644, 5.17637]
y_gpu = [0.64247, 0.64227, 0.64554, 0.64509, 0.63852, 0.64558, 0.67133, 0.67038, 0.69042, 0.704, 0.91162, 0.98917, 1.00544, 1.09902, 1.2394, 4.30916]
y_fpga = [0.00672, 0.00946, 0.01206, 0.01555, 0.01848, 0.03216, 0.05677, 0.08017, 0.0943, 0.11606, 0.20479, 0.40915, 0.62899, 0.82361, 1.04997, 2.10918]
y_cpu_rate = []
y_gpu_rate = []
y_fpga_rate = []
for it, t in enumerate(x):
    y_cpu_rate.append(t/y_cpu[it])
    y_gpu_rate.append(t/y_gpu[it])
    y_fpga_rate.append(t/y_fpga[it])

plt.plot(x, y_cpu, 'b^-', label="CPU")
plt.plot(x, y_gpu, 'gs-', label="GPU")
plt.plot(x, y_fpga, 'ro-', label="FPGA")

#plt.plot(x, y_cpu_rate, 'b^-', label="CPU")
#plt.plot(x, y_gpu_rate, 'gs-', label="GPU")
#plt.plot(x, y_fpga_rate, 'ro-', label="FPGA")
plt.grid()
plt.ylabel('Time [s]', fontdict={"size":15})
#plt.ylabel('Rate [Hz]', fontdict={"size":15})
plt.xlabel('Number of Images', fontdict={"size":15})
#for x, y in zip(x, y_cpu):
#    plt.annotate(format(y, ".3f"), (x, y))
# plt.xscale("log")
plt.title("Performance of Processors (Time)", fontdict={"size":20})
#plt.title("Performance of Processors (Frame Rate)", fontdict={"size":20})
plt.legend()
plt.show()
