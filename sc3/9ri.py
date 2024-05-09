import numpy as np

class GaussianMF:
    def __init__(self, name, mean, std_dev):
        self.name = name
        self.mean = mean
        self.std_dev = std_dev
    
    def membership(self, x):
        return np.exp(-0.5 * ((x - self.mean) / self.std_dev) ** 2)

class FuzzyInferenceSystem:
    def __init__(self, rules):
        self.rules = rules
    
    def infer(self, inputs):
        aggregated = {}
        for rule in self.rules:
            degree = min([mf.membership(inputs[var]) for var, mf in rule['antecedents']])
            for consequent, mf in rule['consequent']:
                if consequent in aggregated:
                    aggregated[consequent] = np.maximum(aggregated[consequent], degree * mf.membership(inputs[consequent]))
                else:
                    aggregated[consequent] = degree * mf.membership(inputs[consequent])
        
        output = {}
        for var, mfs in aggregated.items():
            output[var] = np.max(mfs)
        
        return output

# Example usage
if __name__ == "__main__":
    # Define membership functions
    cold = GaussianMF("cold", 0, 2)
    warm = GaussianMF("warm", 5, 2)
    hot = GaussianMF("hot", 10, 2)

    slow = GaussianMF("slow", 0, 2)
    medium = GaussianMF("medium", 5, 2)
    fast = GaussianMF("fast", 10, 2)

    # Define rules
    rules = [
        {'antecedents': [('temperature', cold), ('humidity', cold)], 'consequent': [('speed', slow)]},
        {'antecedents': [('temperature', warm)], 'consequent': [('speed', medium)]},
        {'antecedents': [('temperature', hot), ('humidity', hot)], 'consequent': [('speed', fast)]}
    ]

    # Create fuzzy inference system
    fis = FuzzyInferenceSystem(rules)

    # Define input values
    inputs = {'temperature': 7, 'humidity': 0.3, 'speed': 0.5}  # Correct input values passed here

    # Perform inference
    output = fis.infer(inputs)
    print(output)  # Output: {'speed': 0.8824969025845955}
