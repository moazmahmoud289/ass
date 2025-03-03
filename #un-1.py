class SimpleNeuralNetwork:
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

    # tanh activation function
    def tanh(self, x):
        return (2.718281828459045 ** x - 2.718281828459045 ** (-x)) / (2.718281828459045 ** x + 2.718281828459045 ** (-x))

    # Derivative of tanh
    def tanh_derivative(self, x):
        return 1 - x ** 2

    # Forward pass
    def forward_pass(self, inputs, weights, biases):
        self.z_hidden = inputs[0] * weights[0] + inputs[1] * weights[1] + biases[0]
        self.a_hidden = self.tanh(self.z_hidden)
        self.z_output = self.a_hidden * weights[2] + biases[1]
        self.a_output = self.tanh(self.z_output)
        return self.a_output

    # Error function
    def calculate_error(self, output, target):
        return 0.5 * (target - output) ** 2

    # Backward pass
    def backward_pass(self, target, inputs, weights, biases):
        output_error = self.a_output - target
        output_delta = output_error * self.tanh_derivative(self.a_output)

        hidden_error = output_delta * weights[2]
        hidden_delta = hidden_error * self.tanh_derivative(self.a_hidden)

        weights[2] -= self.learning_rate * self.a_hidden * output_delta
        weights[0] -= self.learning_rate * inputs[0] * hidden_delta
        weights[1] -= self.learning_rate * inputs[1] * hidden_delta
        biases[1] -= self.learning_rate * output_delta
        biases[0] -= self.learning_rate * hidden_delta

        return weights, biases

# User inputs
inputs = [float(input("Enter input 1: ")), float(input("Enter input 2: "))]
weights = [float(input("Enter weight w1: ")), float(input("Enter weight w2: ")), float(input("Enter weight w3: "))]
biases = [float(input("Enter bias for hidden neuron: ")), float(input("Enter bias for output neuron: "))]
target = float(input("Enter desired output: "))
learning_rate = float(input("Enter learning rate: "))
num_iterations = int(input("Enter the number of iterations: "))

# Initialize network
nn = SimpleNeuralNetwork(learning_rate)

# Training loop
for i in range(num_iterations):
    output = nn.forward_pass(inputs, weights, biases)
    error = nn.calculate_error(output, target)
    print(f"Iteration {i+1}: Total error = {error}")
    weights, biases = nn.backward_pass(target, inputs, weights, biases)

# Final weights
print("\nFinal updated weights:", weights)
