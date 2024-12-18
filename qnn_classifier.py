import sys
import os
# # sys.path.append(os.path.join(os.path.dirname(__file__),".."))
# sys.path.append(os.path.join(os.path.abspath(''), '..'))
# import time
import pickle
import numpy as np
# import pandas as pd
# from utils.metrics import recall_precision_for_each_status_code
# from utils.load_dataset import load_data
# import tensorflow as tf
# from keras.models import Sequential, Model
# from keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Input
# import tensorflow as tf
# from tensorflow import keras
import matplotlib.pyplot as plt
# from sklearn.model_selection import RepeatedKFold
# from tqdm import tqdm
from qiskit_machine_learning.circuit.library import RawFeatureVector
from qiskit.circuit.library import EfficientSU2, ExcitationPreserving, PauliTwoDesign, RealAmplitudes
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit_algorithms.optimizers import COBYLA, L_BFGS_B
# from qiskit_algorithms.utils import algorithm_globals
from qiskit_machine_learning.algorithms.classifiers import NeuralNetworkClassifier
from qiskit_machine_learning.neural_networks import EstimatorQNN, SamplerQNN
from sklearn.model_selection import train_test_split
from IPython.display import clear_output
# from qiskit_algorithms.utils import algorithm_globals
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import clear_output
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.circuit.library import RealAmplitudes, ZZFeatureMap
from qiskit_algorithms.optimizers import COBYLA, L_BFGS_B, ADAM
from qiskit_algorithms.utils import algorithm_globals

from qiskit_machine_learning.algorithms.classifiers import NeuralNetworkClassifier, VQC
from qiskit_machine_learning.algorithms.regressors import NeuralNetworkRegressor, VQR
from qiskit_machine_learning.neural_networks import SamplerQNN, EstimatorQNN
from qiskit_machine_learning.circuit.library import QNNCircuit
import argparse
import pandas as pd
import random

objective_func_vals = []


def load_data(x_root, y_root):
    with open(x_root, 'rb') as handle:
        X_inputs = pickle.load(handle)

    with open(y_root, 'rb') as handle:
        y_outputs = pickle.load(handle) # shape: (42086, 70, 4)
    
    for i in range(len(y_outputs)):
        if y_outputs[i] == 1:
            y_outputs[i] = -1
        elif y_outputs[i] == 0:
            y_outputs[i] = 1
    return X_inputs, y_outputs

# callback function that draws a live plot when the .fit() method is called
def callback_print(weights, obj_func_eval):
    print("loss: "+str(obj_func_eval))
    objective_func_vals.append(obj_func_eval)

def loss_graph(objective_func_vals, optimizer):
    plt.title("Objective function value against iteration")
    plt.xlabel("Iteration")
    plt.ylabel("Objective function value")
    plt.plot(range(len(objective_func_vals)), objective_func_vals)
    plt.savefig("qnn_"+optimizer+".png")

def create_ansatz(type, entangle, mode, reps, num_qubits):
    gate_list = ['rx', 'ry', 'rz', 'x', 'y', 'z']
    if type == "efficientsu2":
        gate_num = random.randint(1, 3)
        su2_gates = random.sample(gate_list, gate_num)
        circuit = EfficientSU2(num_qubits = num_qubits, su2_gates=su2_gates, entanglement=entangle, reps=reps)
        return circuit
    elif type == "excitationpreserving":
        if mode=='0':
            mode = 'fsim'
        elif mode=='1':
            mode = 'iswap'
        circuit = ExcitationPreserving(num_qubits=num_qubits, reps=reps, mode=mode, entanglement=entangle)
        return circuit
    elif type == 'paulitwodesign':
        circuit = PauliTwoDesign(num_qubits=num_qubits, reps=reps)
        return circuit
    else:
        circuit = RealAmplitudes(num_qubits=num_qubits, reps=reps, entanglement=entangle)
        return circuit
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('type', type=str)
    parser.add_argument('entangle', type=str)
    parser.add_argument('mode', type=str)
    parser.add_argument('reps', type=int)
    parser.add_argument('run', type=int)
    parser.add_argument('feature_num', type=int)
    args = parser.parse_args()
    ansatz_type = args.type
    entangle = args.entangle
    mode = args.mode
    reps = args.reps
    run = args.run
    feature_num = args.feature_num

    print(ansatz_type, entangle, mode, reps)
    # feature_num = 8
    gen_number = 400
    optimizer = COBYLA(maxiter=gen_number)

    X_inputs, y_outputs = load_data('cancer/encoded_data_'+str(feature_num)+'.pkl', 'cancer/encoded_output_'+str(feature_num)+'.pkl')
    # X_train, X_test, y_train, y_test = train_test_split(X_inputs, y_outputs, test_size=0.05, train_size=0.25, random_state=1, stratify=y_outputs)
    X_train, X_test, y_train, y_test = train_test_split(X_inputs, y_outputs, test_size=500, train_size=1000, random_state=2, stratify=y_outputs)
    # construct QNN with the QNNCircuit's default ZZFeatureMap feature map and RealAmplitudes ansatz.
    feature_map = ZZFeatureMap(feature_dimension=feature_num)
    ansatz = create_ansatz(type=type, entangle=entangle, mode=mode, reps=reps, num_qubits=feature_num)
    qc = QuantumCircuit(feature_num)
    qc.compose(feature_map, inplace=True)
    qc.compose(ansatz, inplace=True)
    estimator_qnn = EstimatorQNN(circuit=qc, input_params=feature_map.parameters, weight_params=ansatz.parameters)
 
    # construct neural network classifier
    estimator_classifier = NeuralNetworkClassifier(
        estimator_qnn, optimizer=optimizer, callback=callback_print
    )

    # fit classifier to data
    estimator_classifier.fit(X_train, y_train)

    # score classifier
    print(f"Accuracy from the training data : {np.round(100 * estimator_classifier.score(X_train, y_train), 2)}%")
    y_predict = estimator_classifier.predict(X_test)
    x = np.asarray(X_test)
    y = np.asarray(y_test)

    model_path = os.path.join("zzmap_"+str(feature_num)+"_1000_seed2", str(run), "models_"+str(reps), ansatz_type)
    if mode == "0":
        model_path = os.path.join(model_path, "fsim")
    elif mode == "1":
        model_path = os.path.join(model_path, "iswap")

    if not os.path.exists(model_path):
        os.makedirs(model_path)
    estimator_classifier.save(os.path.join(model_path, entangle+".model"))

    outputs_path = os.path.join("zzmap_"+str(feature_num)+"_1000_seed2", str(run), "outputs_"+str(reps), ansatz_type)
    if mode == "0":
        outputs_path = os.path.join(outputs_path, "fsim")
    elif mode == "1":
        outputs_path = os.path.join(outputs_path, "iswap")

    if not os.path.exists(outputs_path):
        os.makedirs(outputs_path)
    
    print(f"Accuracy from the test data : {np.round(100 * estimator_classifier.score(x, y), 2)}%")

    # print(y_test)
    # print(y_predict.flatten().astype(int))
    df = pd.DataFrame({'y_test': y_test, 'y_predict': y_predict.flatten().astype(int)})
    df.to_csv(os.path.join(outputs_path, entangle+"_output.csv"), index=False)

    df = pd.DataFrame({'loss': objective_func_vals})
    df.to_csv(os.path.join(outputs_path, entangle+"_loss.csv"))
    # save model and loss
