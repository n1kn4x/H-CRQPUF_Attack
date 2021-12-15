from matplotlib.legend_handler import HandlerLine2D, HandlerTuple
from qiskit import QuantumCircuit, assemble, Aer, transpile
from scipy.optimize import leastsq
from math import pi, sqrt
import numpy as np
from qiskit.visualization import plot_bloch_multivector, plot_histogram
from scipy.spatial.distance import hamming
import time
import concurrent.futures
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial import distance_matrix
from scipy import optimize
import sklearn.preprocessing
import sklearn.linear_model
from itertools import product
from matplotlib import cm

from qiskit import IBMQ
from qiskit.providers.aer import AerSimulator
provider = IBMQ.load_account()
plt.rcParams.update({'font.size': 14, 'font.family': 'serif', 'mathtext.fontset': 'dejavuserif'})


"""
========================= BEGIN OF IMPORTANT FUNCTIONS =========================
"""

def hpuf_circuit_multi(n_qubits, gates):
    """
    Get H-QuPUF circuit defined by gates, i.e a list of gate_chains for each qubit.
    """
    circuit = QuantumCircuit(n_qubits, n_qubits)
    # Iterate over Qubits.
    for i in range(n_qubits):
        # Perform Rotations
        challenge_chain = gates[i]
        for gate_type, param in challenge_chain:
            if gate_type == 'rx':
                circuit.rx(param, i)
            elif gate_type == 'ry':
                circuit.ry(param, i)
            elif gate_type == 'rz':
                circuit.rz(param, i)
            else:
                print("ERROR: GATE %s NOT FOUND." % gate_type)
        # Perform Hadamard gate on qubit i
        circuit.h(i)
        # Measure the qubit i
        circuit.measure(i, i)
    return circuit

def get_biases(circuit, backend, n_shots=2048):
    """
    Return the (expec.val. - 0.5) of the qubits after running circuit using
    n_shots samples.
    """
    qobj = transpile(circuit, backend=backend)
    mems = backend.run(qobj, memory=True, shots=n_shots).result().get_memory() # Do the simulation, returning the obtained measurement shots
    mems = np.array([[int(symb) for symb in res] for res in mems]) # Convert results of shots to array of 0,1 ints
    means = np.mean(mems, axis=0) # Calculate means
    biases = means - 0.5 # Calculate biases
    biases = list(biases)
    biases.reverse()
    return np.array(biases)

def get_biases_timed(circuits, backend, between_sec):
    """
    Run one circuit in circuits on backend every between_sec seconds.
    """
    results = []
    for circ in circuits:
        biases = get_biases(circ, backend)
        results.append(biases)
        time.sleep(between_sec)
    return results

def probs_to_signatures(P, n_sig_bits):
    """
    Convert an array of probabilities (n_challenges, n_qubits) to discretized
    signatures. For each probability n_sig_bits are used.
    """
    def prob_to_sig(p, n_sig_bits):
        if p > 1.0 or p < 0.0:
            raise ValueError("p must be between 0 and 1")
        n_steps = (2**n_sig_bits) - 1
        step_size = 1 / n_steps
        sig = int(p / step_size)
        sig_bin = format(sig, '#0%db'%(2+n_sig_bits))[2:7]
        return [int(char) for char in sig_bin]
    n_challenges, n_qubits = P.shape
    signatures = np.zeros((n_challenges, n_qubits*n_sig_bits))
    for l, probs in enumerate(P):
        for k, p in enumerate(probs):
            signatures[l, k*n_sig_bits:(k+1)*n_sig_bits] = prob_to_sig(p, n_sig_bits)
    return signatures

def inter_HD(A, B):
    """
    Return the hamming distance matrix of two arrays of signatures.
    """
    def compare_signatures(sig_A, sig_B):
        if sig_A.shape != sig_B.shape:
            raise ValueError("Signature shapes are not the same.")
        return hamming(sig_A, sig_B)
    num_sigs_A, _ = A.shape
    num_sigs_B, _ = B.shape
    hds = np.zeros((num_sigs_A, num_sigs_B))
    for i in range(num_sigs_A):
        for j in range(num_sigs_B):
            hds[i,j] = compare_signatures(A[i,:], B[j,:])
    return hds

def HD_distance_matrix(X, Y, num_bits=5):
    """
    Return the hamming distance matrix of two arrays of biases.
    """
    # shape of X and Y: dim1, n_qubits
    sigs_X = probs_to_signatures(X + .5, n_sig_bits=num_bits)
    sigs_Y = probs_to_signatures(Y + .5, n_sig_bits=num_bits)
    return inter_HD(sigs_X, sigs_Y)


"""
======================== BEGIN OF PLOTTING FUNCITONS =======================
"""
def plot_1d_biases(challenges, biases, all_models, fp=None, qb1=0, qb2=4):
    """
    Scatterplot of challenge response pairs and lineplot of respective models.
    """
    biases1 = biases[..., qb1]
    biases2 = biases[..., qb2]
    regr1, poly1 = all_models[qb1]
    regr2, poly2 = all_models[qb2]
    x_cont = np.linspace(0, 2*np.pi, 1000).reshape(-1,1)
    y1 = regr1.predict(poly1.fit_transform(x_cont))
    y2 = regr2.predict(poly2.fit_transform(x_cont))
    s1 = plt.scatter(x=cont_rots, y=biases1, marker='o', label="foo")
    s2 = plt.scatter(x=cont_rots, y=biases2, marker='x', label="foo")
    p1 = plt.plot(x_cont.flatten(), y1.flatten(), '-', label="foo")
    p2 = plt.plot(x_cont.flatten(), y2.flatten(), '--', label="foo")
    plt.xlabel(r"$\theta$")
    plt.ylabel(r"$R_\mathrm{out} - 0.5$")
    plt.xticks([0, np.pi, 2*np.pi], [0, r"$\pi$", r"$2\pi$"])
    leg, _ = plt.gca().get_legend_handles_labels()
    plt.legend([(leg[2],leg[0]), (leg[3],leg[1])], ["Qubit 1", "Qubit 2"] , numpoints=1,
               handler_map={tuple: HandlerTuple(ndivide=None)})
    plt.subplots_adjust(left=0.15)
    plt.grid()
    if not fp is None:
        plt.savefig(fp)

"""
============================== SETUP PHASE =================================
"""
backend_A = provider.backend.ibmq_bogota

# Uncomment if Simulator should be used a true QC
backend_A = AerSimulator.from_backend(backend_A)


# Define challenge_chain
gates = ['rx', 'ry']
#gates = ['ry']

# Query continuously the biases of QC_A
num_samples = 30
n_runs = 1
n_qubits = 5
all_circuits = []
cont_rots = np.linspace(0, 2*np.pi, num_samples)
challenges = np.array(np.meshgrid(*([cont_rots]*len(gates)))).T.reshape(-1, len(gates))

for i, rots in enumerate(challenges):
    gate_chain = [(gates[k], rots[k]) for k in range(len(gates))]
    h_qc = hpuf_circuit_multi(n_qubits, [gate_chain]*n_qubits)
    all_circuits = all_circuits + ([h_qc]*n_runs)

# Get Biases and average over n_runs
measured_biases = get_biases_timed(all_circuits, backend_A, between_sec=0)
measured_biases = np.array(measured_biases).reshape(num_samples**len(gates), n_runs, n_qubits)
measured_biases = np.mean(measured_biases, axis=1)

# Reshape
measured_biases = measured_biases.reshape(*[num_samples]*len(gates), n_qubits)


# Pick one qubit at random
#qb_idx = np.random.choice(np.arange(n_qubits))
#measured_biases = measured_biases[..., qb_idx]
# Plot results for one qubit
if len(gates) == 1:
    plt.scatter(challenges, measured_biases[...,0])
else:
    plt.imshow(measured_biases[..., 0]); plt.colorbar()


"""
============================== MODEL PHASE ===================================
"""
# For each qubit fit a polynomial model
all_models = []
for qb_idx in range(n_qubits):
    X = challenges
    Y = measured_biases[..., qb_idx].reshape(num_samples**len(gates), 1)

    poly = sklearn.preprocessing.PolynomialFeatures(degree=10)
    X_ = poly.fit_transform(X)

    regr = sklearn.linear_model.LinearRegression()
    regr.fit(X_, Y)
    all_models.append((regr, poly))

# Show model results for first qubit
regr, poly = all_models[0]

test_samples = np.linspace(0, 2*np.pi, num=100)
# 1D
if len(gates) == 1:
    test_samples = test_samples.reshape(-1,1)
else:
    # 2D
    test_samples = np.array(list(product(test_samples, test_samples)))

test_samples.shape
X_test = poly.fit_transform(test_samples)
Y_test = regr.predict(X_test)

challenges.shape
measured_biases.shape

if len(gates) == 1:
    plot_1d_biases(challenges, measured_biases, all_models, fp="/home/niklas/out.pdf", qb1=0, qb2=2)
else:
    pixels = regr.predict(X_)
    plt.imshow(pixels.reshape((num_samples,num_samples)))

"""
============================== ATTACK PHASE ===================================
"""
# Randomly sample challenges for all qubits
n_challenges = 15
# Define new gate sequence that will then be transformed to the rxry case for the attack
new_gates = ['rx', 'rx', 'rx', 'rx', 'ry', 'ry', 'ry', 'ry']
#new_gates = ['ry']
challenges = np.random.uniform(0, 2*np.pi, size=(n_challenges, len(new_gates), n_qubits))

# Predict challenges using all learned models for the qubits
# 1D case
if len(gates) == 1:
    attack_challenges = np.sum(challenges, axis=1) % (2*np.pi)
# 2D case
else:
    rx_idx = [idx for idx, elem in enumerate(new_gates) if elem == 'rx']
    ry_idx = [idx for idx, elem in enumerate(new_gates) if elem == 'ry']
    rx_challenges = challenges[:, rx_idx, :]
    ry_challenges = challenges[:, ry_idx, :]
    rx_challenges = rx_challenges.sum(axis=1) % (2*np.pi)
    ry_challenges = ry_challenges.sum(axis=1) % (2*np.pi)

    attack_challenges = np.zeros((n_challenges, 2, n_qubits)) # TODO: Generalize
    attack_challenges[:, 0, :] = rx_challenges
    attack_challenges[:, 1, :] = ry_challenges

predicted_biases = np.zeros((n_challenges, n_qubits))
for qb_idx in range(n_qubits):
    regr, poly = all_models[qb_idx]
    for i, rots in enumerate(attack_challenges[..., qb_idx]):
        # 1D case
        if len(gates) == 1:
            x = np.array([[rots]])
        # 2D case
        else:
            x = np.array([[rots[k] for k in range(len(gates))]])
        feat = poly.fit_transform(x)
        predicted_biases[i, qb_idx] = regr.predict(feat)

# Run challenges through Quantum Computer n_run times
n_runs = 5
all_circuits = []
for i, rots in enumerate(challenges):
    # Add one gate chain for each qubit
    all_gate_chains = []
    for qb_idx in range(n_qubits):
        gate_chain = [(new_gates[k], rots[k, qb_idx]) for k in range(len(new_gates))]
        all_gate_chains.append(gate_chain)
    # Add the circuit for the challenge
    h_qc = hpuf_circuit_multi(n_qubits, all_gate_chains)
    all_circuits = all_circuits + ([h_qc]*n_runs)

challenged_biases = get_biases_timed(all_circuits, backend_A, between_sec=0)
challenged_biases = np.array(challenged_biases)
challenged_biases = challenged_biases.reshape(*[n_challenges], n_runs, n_qubits)

challenged_biases.shape


# Pick the selected qubit
# challenged_biases = challenged_biases[..., qb_idx]

"""
============================== Evaluation ===================================
"""
# Plot one candle for each challenge, where the candle depicts the hamming distances
# among multiple bias measurements

distances = []
for biases_per_chal in challenged_biases:
    x = biases_per_chal.reshape(n_runs, n_qubits)
    D = HD_distance_matrix(x, x)

    # Discard symmetric values in the distance matrix and append the distances
    # to a vector.
    d = []
    for i, row in enumerate(D):
        d = d + list(row[i+1:])
    distances.append(d)



def plot():
    #plt.style.use('science')
    meanlineprops = dict(linestyle='-', color='black')
    medianprops = dict(visible=False)
    # Plot candles for different challenges
    plt.boxplot(distances, showmeans=True, meanline=True, showfliers=False,
                 meanprops=meanlineprops, medianprops=medianprops)

    # Plot mean hamming distances between challenged and predicted biases
    x = np.arange(1, n_challenges+1)
    # Average the hamming distances over the number of runs in holdout set
    y = np.zeros((n_challenges, n_challenges, n_runs))
    for i in range(n_runs):
        y[:,:, i] = HD_distance_matrix(predicted_biases, challenged_biases[:, i, :])
    y = np.mean(y, axis=-1)
    # We're interested in the HD between corresponding challenges
    y = np.diag(y)
    plt.plot(x, y, 'r.')

    plt.ylabel("Normed Hamming distance")
    plt.xlabel(r"$k$")
    plt.savefig("/home/niklas/out.pdf")
plot()


def coeff_string(coeffs):
    items = []
    for i, c in enumerate(coeffs):
        if not c:
            continue
        #if np.abs(c) < 0.05:
        #    continue
        items.append('{:10.3f}*x^{}'.format(c if c != 1 else '', i))
    result = ' + '.join(items)
    result = result.replace('x^0', '')
    result = result.replace('^1 ', ' ')
    result = result.replace('+ -', '- ')
    return result

#coeff_string(all_models[0][0].coef_.flatten())
#coeff_string(all_models[4][0].coef_.flatten())

"""
Advanced Plotting
"""
# Plot a nice 3d landscape of scattered measurements and model landscape
def plot3D():
    m_biases = measured_biases[..., 0] # select first qubit
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    # Plot measured biases
    challenges = np.array(np.meshgrid(*([cont_rots]*len(gates)))).T.reshape(-1, len(gates))
    ax.scatter3D(challenges[:,0], challenges[:,1], m_biases.reshape(-1), color='black', marker='.', zorder=0)

    # Plot the fitted curve
    X, Y = np.meshgrid(cont_rots, cont_rots)
    ax.plot_surface(Y, X, pixels.reshape(X.shape), linewidth=0, alpha=.7, cmap=cm.coolwarm, antialiased=False, zorder=1)

    ax.set_xlabel(r'Rotation $\theta_y$ around $Y$ axis')
    ax.set_ylabel(r'Rotation $\theta_x$ around $X$ axis')
    ax.set_zlabel(r'$f^{(j)}(\theta_x, \theta_y) - 0.5$')

    ax.view_init(elev=31, azim=307)
    plt.savefig("/tmp/test.pdf")


plot3D()



from mayavi import mlab
from mayavi.mlab import *

def plot3D():
    # Plot the fitted curve
    figure(bgcolor=(1,1,1), fgcolor=(0.,0.,0.))
    X, Y = np.meshgrid(cont_rots, cont_rots)
    surf(Y, X, pixels.reshape(X.shape))

    # Plot measured biases
    m_biases = measured_biases[..., 0] # select first qubit
    challenges = np.array(np.meshgrid(*([cont_rots]*len(gates)))).T.reshape(-1, len(gates))
    points3d(challenges[:,0], challenges[:,1], m_biases.reshape(-1), scale_factor=.08, color=(0, 0, 0))


    xlabel(r"$\theta_x$")
    ylabel(r"$\theta_y$")
    zlabel(r"$r_\mathrm{out} - 0.5$")

    # View it.
    mlab.show()

plot3D()
