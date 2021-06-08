import GradientDescent
import ImplementAdagrad
import ImplementAdam
import ImplementMomentum
import ImplementMomentumNAG
import ImplementRMSProp
import MiniBatchGradientDescent
import StochasticGradientDescent
from GenerateData import generate_random_data
from Plot import run_all_functions

input_data, target_labels = generate_random_data(-2, 1)
data = GradientDescent.get_gradiant(input_data, target_labels, epochs=1000)
run_all_functions(data, input_data, target_labels)

input_data, target_labels = generate_random_data(-1, 2)
data = ImplementMomentum.get_gradiant_using_momentum(input_data, target_labels, epochs=50)
run_all_functions(data, input_data, target_labels)

input_data, target_labels = generate_random_data(-1, 2)
data = ImplementMomentumNAG.get_gradiant_using_momentum_nag(input_data, target_labels, epochs=1000)
run_all_functions(data, input_data, target_labels)

input_data, target_labels = generate_random_data(-2, 1, n=100)
data = MiniBatchGradientDescent.using_mini_batch(input_data, target_labels, epochs=1000)
run_all_functions(data, input_data, target_labels)

input_data, target_labels = generate_random_data(-2, 1)
data = StochasticGradientDescent.stochastic_GD(input_data, target_labels, 3)
run_all_functions(data, input_data, target_labels)

input_data, target_labels = generate_random_data(-1, 2)
data = ImplementAdagrad.implement_adagrad(input_data, target_labels, epochs=1000, alpha=0.1)
run_all_functions(data, input_data, target_labels)

input_data, target_labels = generate_random_data(-1, 2)
data = ImplementRMSProp.implement_rms_prop(input_data, target_labels, epochs=1000, alpha=0.1)
run_all_functions(data, input_data, target_labels)

input_data, target_labels = generate_random_data(-1, 2)
data = ImplementAdam.implement_adam(input_data, target_labels)
run_all_functions(data, input_data, target_labels)