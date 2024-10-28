import re
import matplotlib.pyplot as plt
# use latex
plt.rc('text', usetex=True)
plt.rc('font', family='serif')


def parse_memory_profiler(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
    memory = []
    for line in lines:
        if line.startswith('MEM'):
            memory.append(float(re.findall(r'\d+\.\d+', line)[0]))
    return memory


def normalize_memory(memory, max_memory):
    # Normalize memory to the maximum value.
    return [m/max_memory for m in memory]


def plot_memory_profiler(revolve, no_schedule):
    revolve = parse_memory_profiler(revolve)
    no_schedule = parse_memory_profiler(no_schedule)
    max_memory = max(max(revolve), max(no_schedule))
    max_runtime = max(len(revolve), len(no_schedule))
    revolve = normalize_memory(revolve, max_memory)
    no_schedule = normalize_memory(no_schedule, max_memory)
    max_runtime = len(revolve)
    t_revolve = [i/max_runtime for i in range(len(revolve))]
    t_no_schedule = [i/max_runtime for i in range(len(no_schedule))]
    plt.xticks(ticks=[float(i)/5 for i in range(0, 20)], fontsize=14)
    plt.yticks(ticks=[float(i)/10 for i in range(0, 11)], fontsize=14)
    plt.plot(t_revolve, revolve, label=r'Adjoint solver - master', linestyle='-', linewidth=1, color='r', marker='*')
    plt.plot(t_no_schedule, no_schedule, label=r'Adjoint solver - It is coming', linestyle='-', linewidth=3, color='k')
    plt.ylabel(r'Normalised memory', fontsize=14)
    plt.xlabel(r'Normalised runtime', fontsize=14)
    plt.title(r'Memory profiler - Nolinear Navier-Stokes', fontsize=14)
    # increase the font size
    plt.rcParams.update({'font.size': 14})
    # plt.grid()
    plt.legend()
    plt.show() 


master = 'opt_ns_master'
adj_solver = 'opt_ns_adj_solver'

plot_memory_profiler(master, adj_solver)