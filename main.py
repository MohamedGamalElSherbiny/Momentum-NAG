from GradientDescent import GradientDescent
from ImplementAdagrad import ImplementAdagrad
from ImplementAdam import ImplementAdam
from ImplementMomentum import ImplementMomentum
from ImplementMomentumNAG import ImplementMomentumNAG
from ImplementRMSProp import ImplementRMSProp
from MiniBatchGradientDescent import MiniBatchGradientDescent
from StochasticGradientDescent import StochasticGradientDescent
from GenerateData import GenerateData
from Plot import Plot

generate_data = GenerateData(-2, 1)
input_data, target_labels = generate_data.generate_random_data()
gd = GradientDescent(input_data, target_labels, epochs=1000)
data = gd.get_gradiant_data()
plot_data = Plot(data, input_data, target_labels)
plot_data.run_all_functions()

generate_data = GenerateData(-2, 1)
input_data, target_labels = generate_data.generate_random_data()
stochastic_gradient_descent = StochasticGradientDescent(input_data, target_labels, 3)
data = stochastic_gradient_descent.stochastic_GD_data()
plot_data = Plot(data, input_data, target_labels)
plot_data.run_all_functions()

generate_data = GenerateData(-2, 1, n=100)
input_data, target_labels = generate_data.generate_random_data()
mini_batch_gradient_descent = MiniBatchGradientDescent(input_data, target_labels, epochs=1000)
# data = mini_batch_gradient_descent.using_mini_batch_data()
# plot_data = Plot(data, input_data, target_labels)
# plot_data.run_all_functions()

generate_data = GenerateData(-1, 2)
input_data, target_labels = generate_data.generate_random_data()
momentum_gradient_descent = ImplementMomentum(input_data, target_labels, epochs=50)
data = momentum_gradient_descent.get_momentum_data()
plot_data = Plot(data, input_data, target_labels)
plot_data.run_all_functions()

generate_data = GenerateData(-1, 2)
input_data, target_labels = generate_data.generate_random_data()
implement_adagrad = ImplementAdagrad(input_data, target_labels, epochs=1000, alpha=0.1)
data = implement_adagrad.get_adagrad_data()
plot_data = Plot(data, input_data, target_labels)
plot_data.run_all_functions()

generate_data = GenerateData(-1, 2)
input_data, target_labels = generate_data.generate_random_data()
implement_rms_prop = ImplementRMSProp(input_data, target_labels, epochs=1000, alpha=0.1)
data = implement_rms_prop.implement_rms_prop_data()
plot_data = Plot(data, input_data, target_labels)
plot_data.run_all_functions()

generate_data = GenerateData(-1, 2)
input_data, target_labels = generate_data.generate_random_data()
implement_adam = ImplementAdam(input_data, target_labels)
data = implement_adam.get_adam_data()
plot_data = Plot(data, input_data, target_labels)
plot_data.run_all_functions()

generate_data = GenerateData(-1, 2)
input_data, target_labels = generate_data.generate_random_data()
implement_momentum_nag = ImplementMomentumNAG(input_data, target_labels, epochs=1000)
data = implement_momentum_nag.get_gradiant_using_momentum_nag()
plot_data = Plot(data, input_data, target_labels)
plot_data.run_all_functions()