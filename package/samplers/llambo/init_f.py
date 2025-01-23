import optuna
from optuna.samplers import RandomSampler

def generate_hyperparameter_list(num_samples, objective):
    """
    Generate a list of hyperparameters using Optuna's RandomSampler.

    Args:
        num_samples (int): Number of hyperparameter sets to generate.
        objective (callable): A function that defines the search space for hyperparameters.

    Returns:
        list: A list of dictionaries, where each dictionary contains a set of hyperparameters.
    """
    # Initialize the random sampler
    sampler = RandomSampler(seed=42)

    # Create a study
    study = optuna.create_study(sampler=sampler)

    # Generate the hyperparameter list
    hyperparameter_list = []
    for _ in range(num_samples):
        trial = study.ask()  # Ask for a new trial
        params = objective(trial)  # Sample hyperparameters using the objective function
        hyperparameter_list.append(params)

    return hyperparameter_list


# Define the objective function
def objective(trial) :
    # Define the search space for each hyperparameter
    max_depth = trial.suggest_int('max_depth' , 1 , 20)
    max_features = trial.suggest_float('max_features' , 0.01 , 1.0)
    min_impurity_decrease = trial.suggest_float('min_impurity_decrease' , 0.0 , 0.5)
    min_samples_leaf = trial.suggest_float('min_samples_leaf' , 0.01 , 0.5)
    min_samples_split = trial.suggest_float('min_samples_split' , 0.01 , 0.5)
    min_weight_fraction_leaf = trial.suggest_float('min_weight_fraction_leaf' , 0.0 , 0.5)

    # Return a dictionary of the sampled hyperparameters
    return {
        'max_depth' : max_depth ,
        'max_features' : max_features ,
        'min_impurity_decrease' : min_impurity_decrease ,
        'min_samples_leaf' : min_samples_leaf ,
        'min_samples_split' : min_samples_split ,
        'min_weight_fraction_leaf' : min_weight_fraction_leaf
        }


# Generate hyperparameter list
num_samples = 5
hyperparameter_list = generate_hyperparameter_list(num_samples , objective)

# Print the generated hyperparameter list
print(hyperparameter_list)