import pennylane as qml
from pennylane import numpy as np

class Pennylane_solver:
    def __init__(self):
        self.dev = None
        self.ansatz = None

    def qubo_dict_to_matrix(self, qubo_dict, n_variables=None):
        """
        Convert a QUBO dictionary to a numpy matrix.
        
        Args:
            qubo_dict: Dictionary with (i, j) keys and float values
            n_variables: Number of variables (if None, will be inferred from keys)
        
        Returns:
            numpy.ndarray: QUBO matrix of shape (n_variables, n_variables)
        """
        if n_variables is None:
            # Find the maximum index from the dictionary keys
            max_idx = 0
            for (i, j) in qubo_dict.keys():
                if isinstance(i, tuple):
                    idx_i = i[0] if isinstance(i[0], int) else int(str(i[0])[1:])
                    max_idx = max(max_idx, idx_i)
                else:
                    max_idx = max(max_idx, i)
                if isinstance(j, tuple):
                    idx_j = j[0] if isinstance(j[0], int) else int(str(j[0])[1:])
                    max_idx = max(max_idx, idx_j)
                else:
                    max_idx = max(max_idx, j)
            n_variables = max_idx + 1
        
        # Create the matrix
        matrix = np.zeros((n_variables, n_variables))
        
        # Fill the matrix
        for (i, j), val in qubo_dict.items():
            # Handle both tuple keys and integer keys
            if isinstance(i, tuple):
                row_idx = i[0] if isinstance(i[0], int) else int(str(i[0])[1:])
            else:
                row_idx = i
                
            if isinstance(j, tuple):
                col_idx = j[0] if isinstance(j[0], int) else int(str(j[0])[1:])
            else:
                col_idx = j
            
            # Ensure indices are within bounds
            if 0 <= row_idx < n_variables and 0 <= col_idx < n_variables:
                matrix[row_idx, col_idx] = val
        
        return matrix
    
    # Convert binary variables x_i ∈ {0,1} to spin variables σ_i ∈ {-1,1}
    # x_i = (1 + σ_i)/2
    def qubo_to_ising(self, Q):
        n = Q.shape[0]
        coeffs = []
        observables = []

        # Symmetrize Q
        Q_sym = (Q + Q.T) / 2

        # Process all unordered pairs (i <= j)
        for i in range(n):
            for j in range(i, n): # This helps avoiding double iteration over same pairs
                coeff = Q_sym[i, j]
                if i == j:
                    # Linear term
                    coeffs.append(coeff / 2)
                    observables.append(qml.PauliZ(i))
                else:
                    # Quadratic term
                    coeffs.append(coeff / 4)
                    observables.append(qml.PauliZ(i) @ qml.PauliZ(j))

        # Constant term
        constant = np.sum(Q_sym) / 4 + np.trace(Q_sym) / 2

        H = qml.Hamiltonian(coeffs, observables)
        return H, constant
