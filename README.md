# ANN-optimization-methods

This work has studied the hyperparameter optimization problem, and the theoretical 
aspects of these automatic hyperparameter optimization methods and how to 
implement them on a real problem with large daraset, which is the prediction of the B 
coefficient of organic and inorganic mixtures based on (Tc, Pc, Vc, Zc, µ, T) input 
variables, and this work showed that an automatic approach to hyperparameter 
optimization is very valuable, and the GUI feature can be considered a valuable tool that 
can bring the untrained user to the hyperparameters optimization world and the 
implementation of many methods is very convenient for non-experts.

One thing that has to be noticed is that the optimization methods gave us the results 
and the combinations for a 1000 epoch number, and we notice that some combinations 
may take more time to become stable and offer good results and since the 
computational cost is high we couldn’t afford to use a bigger epoch size so there is a 
high possibility that we missed better combinations, yet it is safe to say that for that 
epoch size, we found the best combination. 

Regarding this case study, Optuna and Bayesian search found the best results in a very 
small time (first few hours of the search) and even when random search found the same 
results yet we cannot depend on it because it is based on random search and we just 
got lucky specialy when we take into account the number of combination used and time 
token to find such result.

An Important consideration regarding the optimization process is that finding good 
configurations can be a slow and time consuming so the choice of an optimization 
method based on the optimization history and other factors can be very critical, and 
from the work conducted on the thesis I highly recommend Optuna as the first choice 
in this case

Details can be found here.



### Work environement 

In this project we used PyCharm and Anaconda ,Pycharm as an IDE to make it easier to 
write Python code, by providing a text editor and debugging, among other features.
And Anaconda as a Python distribution focused on data driven projects. Both tools 
popular with businesses of all sizes that use Python. The Python version used is 3.7(64 
bits) using many APIs like Scikit-learn and Keras which is running on top of tensorflow 
package.
Since the cost of computational is high, we used Vertical Scaling and Horizontal 
Scaling
Vertical Scaling: we used Google’s free cloud services via Google’s collaboratory 
application and we used CUDA to harness the power of GPUs for the parallelizable part 
of the computation
Horizontal Scaling: we used several independent machines to split the workload in 
the sense of dividing the set of testing between machines where each one of them has 
the dataset and fully independent.

### The hyperparameters that we are tuning :

the number of neurons in the hidden layers

the number of layers

batch size 

the optimizer

learning rate, momentum and decay

network weight initialization

activation functions

dropout regularization and weight constraint

### Results summary

<img src="https://user-images.githubusercontent.com/119765748/206353250-1ceae98d-6f71-439c-a699-e90aeb641484.JPG">

<img src="https://user-images.githubusercontent.com/119765748/206353254-74a33eb5-21a7-483b-9233-d12b2f06312f.JPG">

<img src="https://user-images.githubusercontent.com/119765748/206353255-9070686f-9447-4e4b-874f-6ebf75add859.JPG">

<img src="https://user-images.githubusercontent.com/119765748/206353258-c266d980-345e-4a81-9d63-df018a92fde7.JPG">

<img src="https://user-images.githubusercontent.com/119765748/206353260-a29c5c5d-cb38-4eb4-9f0d-2560a327e11c.JPG">

<img src="https://user-images.githubusercontent.com/119765748/206353242-b678cc44-1c74-4206-8e68-59fc771b891c.JPG">






