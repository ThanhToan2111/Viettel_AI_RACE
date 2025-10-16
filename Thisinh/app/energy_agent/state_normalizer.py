import numpy as np

class StateNormalizer:
    """Handles state normalization with running statistics"""

    def __init__(self, state_dim, epsilon=1e-8, n_cells=10):
        self.state_dim = state_dim
        self.epsilon = epsilon
        self.n_cells = n_cells

        # Simulation features normalization bounds (first 17 features)
        self.simulation_bounds = {
            'totalCells': [1, 60],               # number of cells (up to 57 for rural)
            'totalUEs': [1, 300],                # number of UEs (up to 300 for dense urban)
            'simTime': [300, 3600],              # simulation time
            'timeStep': [1, 10],                 # time step
            'timeProgress': [0, 1],              # progress ratio
            'carrierFrequency': [700e6, 6e9],    # frequency Hz
            'isd': [20, 2000],                   # inter-site distance (20m for indoor to 1732m for rural)
            'minTxPower': [10, 40],              # dBm (10-35 range)
            'maxTxPower': [23, 50],              # dBm (23-49 range)
            'basePower': [50, 1500],             # watts (50-1200 range)
            'idlePower': [15, 500],              # watts (15-300 range)
            'dropCallThreshold': [0.5, 10],      # percentage
            'latencyThreshold': [10, 150],       # ms (50-100 range)
            'cpuThreshold': [70, 100],           # percentage
            'prbThreshold': [70, 100],           # percentage
            'trafficLambda': [0.1, 30],          # traffic rate (10-25 range)
            'peakHourMultiplier': [1, 5]         # multiplier
        }
        
        # Network features normalization bounds (next 12 features)
        self.network_bounds = {
            'totalEnergy': [0, 20000],           # kWh (higher for rural with many cells)
            'activeCells': [0, 60],              # number of cells
            'avgDropRate': [0, 20],              # percentage
            'avgLatency': [0, 200],              # ms
            'totalTraffic': [0, 10000],          # traffic units
            'connectedUEs': [0, 300],            # number of UEs (max 300)
            'cpuViolations': [0, 100],           # number of violations
            'prbViolations': [0, 100],           # number of violations
            'maxCpuUsage': [0, 100],             # percentage
            'maxPrbUsage': [0, 100],             # percentage
            'totalTxPower': [0, 3000],           # total power (dBm sum for all cells)
            'avgPowerRatio': [0, 1]              # ratio
        }
        
        # Cell features normalization bounds (12 features per cell)
        self.cell_bounds = {
            'cpuUsage': [0, 100],                # percentage
            'prbUsage': [0, 100],                # percentage
            'currentLoad': [0, 1000],            # load units
            'maxCapacity': [0, 1000],            # capacity units
            'numConnectedUEs': [0, 50],          # number of UEs
            'txPower': [0, 46],                  # dBm
            'energyConsumption': [0, 5000],      # watts
            'avgRSRP': [-140, -70],              # dBm
            'avgRSRQ': [-20, 0],                 # dB
            'avgSINR': [-10, 30],                # dB
            'totalTrafficDemand': [0, 500],      # traffic units
            'loadRatio': [0, 1]                  # ratio
        }
    
    def normalize(self, state_vector):
        """Normalize state vector to [0, 1] range"""
        normalized = np.zeros_like(state_vector)

        # Normalize simulation features (first 17)
        simulation_keys = list(self.simulation_bounds.keys())
        for i, key in enumerate(simulation_keys):
            if i < len(state_vector):
                min_val, max_val = self.simulation_bounds[key]
                normalized[i] = self._normalize_value(state_vector[i], min_val, max_val)

        # Normalize network features (next 12)
        network_keys = list(self.network_bounds.keys())
        for i, key in enumerate(network_keys):
            global_idx = 17 + i
            if global_idx < len(state_vector):
                min_val, max_val = self.network_bounds[key]
                normalized[global_idx] = self._normalize_value(state_vector[global_idx], min_val, max_val)

        # Dynamically determine number of cells from state vector
        # State structure: 17 simulation + 12 network + (n_cells * 12) cell features
        start_idx = 17 + 12  # After simulation and network features
        remaining_features = len(state_vector) - start_idx

        if remaining_features > 0:
            actual_n_cells = remaining_features // 12

            # Normalize cell features (remaining features in groups of 12)
            cell_keys = list(self.cell_bounds.keys())

            for cell_idx in range(actual_n_cells):
                for feat_idx, key in enumerate(cell_keys):
                    global_idx = start_idx + cell_idx * 12 + feat_idx
                    if global_idx < len(state_vector):
                        min_val, max_val = self.cell_bounds[key]
                        normalized[global_idx] = self._normalize_value(
                            state_vector[global_idx], min_val, max_val)

        return normalized
    
    def _normalize_value(self, value, min_val, max_val):
        """Normalize single value to [0, 1] range"""
        if max_val == min_val:
            return 0.5  # Default middle value
        return np.clip((value - min_val) / (max_val - min_val), 0.0, 1.0)
    
    def update_stats(self, state_vector):
        """Update running statistics (implement if using running normalization)"""
        # Optional: Update running mean/std statistics
        pass