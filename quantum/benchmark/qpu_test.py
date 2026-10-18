from ibm_cloud_sdk_core.token_managers import jwt_token_manager
from qiskit_ibm_runtime import QiskitRuntimeService

service = QiskitRuntimeService(
    channel="ibm_quantum_platform",
    instance="crn:v1:bluemix:public:quantum-computing:us-east:a/4c524564be024dddabeb021608d23a09:dd555dff-be18-4526-b785-0043d3770232::",
)
job = service.job("d9mb36vbupns73e8h6s0")
job_result = job.result()
print(job_result)
qpu_time = job.metrics()
print(qpu_time)
print(job.usage())


backend = service.backend("ibm_fez")

# Get the precise duration of a single-qubit gate (e.g., 'x' gate on qubit 0)
x_gate_properties = backend.target["x"][(0,)]
gate_latency_seconds = x_gate_properties.duration


# Since the exact latencies and duration metrics for any IBM quantum hardware machine can shift daily, because of system recalibrations
# or parameters shifts, there is no single permanent static list
# Howeverm we can query them directly on real time
print(f"Gate latency: {gate_latency_seconds * 1e9} ns")


# Precise duration of a CZ gate acting across qubit 0 and qubit 1
cz_properties = backend.target["cz"][(0, 1)]
two_qubit_latency = cz_properties.duration  # Yields time in seconds

print(f"Two-qubit gate latency: {two_qubit_latency * 1e9} ns")

# Query properties for physical qubit 0
qubit_properties = backend.target.qubit_properties[0]

t1_time = qubit_properties.t1  # Energy relaxation time in seconds
t2_time = qubit_properties.t2  # Dephasing time in seconds
# legacy_props = backend.properties()
# if legacy_props:
#     frequency = legacy_props.frequency(0) # Qubit 0 frequency in Hz
#     print(f"Qubit 0 Frequency: {frequency / 1e9:.4f} GHz")
# else:
#     print("Frequency data unavailable for this specific backend configuration.")

print(f"Qubit 0 Life/Relaxation Time (T1): {t1_time * 1e6} microseconds")


# Errors
# Get the error rate of a specific 2-qubit gate execution
two_qubit_error = backend.target["cz"][(0, 1)].error

# Get the readout assignment error for measuring a qubit
readout_error = backend.target["measure"][(0,)].error

print(f"2-Qubit Gate Error Rate: {two_qubit_error * 100:.3f}%")
print(f"Readout Error Rate: {readout_error * 100:.3f}%")

# Repetition Delay (Time between shots)
# Note that this one is configurable, lower delay means less QPU time but higher error rate
print(f"Default repetition delay: {backend.default_rep_delay * 1e6} µs")

# Total time per shot = Active Init (4us) + Circuit Gates + Readout (1 us) + Repetition Delay (250 us)

# Querying measurement time
# Get the readout duration for physical qubit 0
measure_properties = backend.target["measure"][(0,)]
readout_latency = measure_properties.duration  # Yields time in seconds

print(f"Readout Time: {readout_latency * 1e9} ns (~{readout_latency * 1e6:.1f} µs)")


# Querying Active initialization Latency
# Get the active reset state-preparation latency for qubit 0
reset_properties = backend.target["reset"][(0,)]
init_latency = reset_properties.duration  # Yields time in seconds

print(f"Active Initialization Time: {init_latency * 1e6:.2f} µs")
# This is a trick to reset faster, else you would need to wait for it to freeze out
# If the qubit is measured in 1, the controller applies a fast pi-pulse to snap it back to 0
# Latency cost = 1 Readout + 1 realtime feedforward control loop + 1 single qubit x-gate
# iF you deactivate that you should set a higher a repetition delay

# One could deactivate it
# from qiskit_ibm_runtime import EstimatorV2, EstimatorOptions

# options = EstimatorOptions()

# # True (Default): Appends the ~4 µs active reset sequence to every shot
# # False: Bypasses active reset. Qubits just thermalize over the rep_delay window
# options.execution.init_qubits = False
