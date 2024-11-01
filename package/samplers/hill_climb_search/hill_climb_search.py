from __future__ import annotations

from typing import Any

import numpy as np
import optuna
import optunahub

class HillClimbSearch(optunahub.samplers.SimpleBaseSampler):
    """A sampler based on the Hill Climb Local Search Algorithm dealing with discrete values.
    """

    def __init__(self,search_space: dict[str, optuna.distributions.BaseDistribution] | None = None) -> None:
        super().__init__(search_space)
        self._remaining_points = []
        self._rng = np.random.RandomState()
        
        #This is for storing the current point whose neighbors are under analysis
        self._current_point = None
        self._current_point_value = None
        self._current_state = "Not Initialized"
        
        #This is for keeping track of the best neighbor
        self._best_neighbor = None
        self._best_neighbor_value = None
        
    def _generate_random_point(self, search_space):
        """This function generates a random discrete point in the search space
        """
        params = {}
        for param_name, param_distribution in search_space.items():
            if isinstance(param_distribution, optuna.distributions.FloatDistribution):
                total_points = int((param_distribution.high - param_distribution.low) / param_distribution.step)
                params[param_name] = param_distribution.low + self._rng.randint(0, total_points)*param_distribution.step
            else:
                raise NotImplementedError
        return params
    
    def _remove_tried_points(self, neighbors, study, current_point):
        """This function removes the points that have already been tried from the list of neighbors
        """
        final_neighbors = []
        
        tried_points = [trial.params for trial in study.get_trials(deepcopy=False)]
        points_to_try = self._remaining_points
        
        invalid_points = tried_points + points_to_try + [current_point]
        
        for neighbor in neighbors:
            if neighbor not in invalid_points:
                final_neighbors.append(neighbor)
                
        return final_neighbors        
    
    def _generate_neighbors(self, current_point, search_space, study):
        """This function generates the neighbors of the current point
        """
        neighbors = []
        for param_name, param_distribution in search_space.items():
            if isinstance(param_distribution, optuna.distributions.FloatDistribution):
                current_value = current_point[param_name]
                step = param_distribution.step
                
                neighbor_low = max(param_distribution.low, current_value - step)
                neighbor_high = min(param_distribution.high, current_value + step)
                
                neighbor_low_point = current_point.copy()
                neighbor_low_point[param_name] = neighbor_low
                neighbor_high_point = current_point.copy()
                neighbor_high_point[param_name] = neighbor_high
                
                neighbors.append(neighbor_low_point)
                neighbors.append(neighbor_high_point)
            else:
                raise NotImplementedError
                
        valid_neighbors = self._remove_tried_points(neighbors, study, current_point)        
        
        return valid_neighbors
        
    def sample_relative(self, study:optuna.study.Study, trial:optuna.trial.FrozenTrial, search_space: dict[str, optuna.distributions.BaseDistribution]) -> dict[str, Any]:
        if search_space == {}:
            return {}
        
        if self._current_state == "Not Initialized":
            #Create the current point
            starting_point = self._generate_random_point(search_space)  
            self._current_point = starting_point
            
            #Add the neighbors
            neighbors = self._generate_neighbors(starting_point, search_space, study)
            self._remaining_points.extend(neighbors)
            
            #Change the state to initialized
            self._current_state = "Initialized"
            
            #Return the current point
            return starting_point
        
        elif self._current_state == "Initialized":
            #This section is only for storing the value of the current point and best neighbor point
            previous_trial = study.get_trials(deepcopy=False)[-2]
            if previous_trial.params == self._current_point:
                #Just now the current point was evaluated
                #Store the value of the current point
                self._current_point_value = previous_trial.value
            else:
                #The neighbor was evaluated
                #Store the value of the neighbor, if it improves upon the current point
                neighbor_value = previous_trial.value
                
                if neighbor_value < self._current_point_value:
                    self._best_neighbor = previous_trial.params
                    self._best_neighbor_value = neighbor_value
            
            #This section is for the next point to be evaluated
            if len(self._remaining_points) == 0:
                #This means that all the neighbors have been processed
                #Now you have to select the best neighbor
                
                if self._best_neighbor is not None:
                    #There was an improvement
                    #Select the best neighbor, make that the current point and add its neighbors
                    self._current_point = self._best_neighbor
                    self._current_point_value = self._best_neighbor_value
                    
                    self._best_neighbor = None
                    self._best_neighbor_value = None
                    self._remaining_points = [] #Happens by virtue of the condition, but just for clarity
                    
                    #Add the neighbors
                    neighbors = self._generate_neighbors(self._current_point, search_space, study)
                    self._remaining_points.extend(neighbors)
                    
                    self._current_state = "Initialized"
                    
                    return self._current_point
                
                else:
                    #If none of the neighbors are better then do a random restart
                    self._current_state = "Not Initialized"
                    restarting_point = self._generate_random_point(search_space)
                    self._current_point = restarting_point
                    
                    self._best_neighbor = None
                    self._best_neighbor_value = None
            
                    #Add the neighbors
                    neighbors = self._generate_neighbors(restarting_point, search_space, study)
                    self._remaining_points.extend(neighbors)
                    
                    #Change the state to initialized
                    self._current_state = "Initialized"
                    
                    #Return the current point
                    return self._current_point
                
            else:
                #Process as normal
                current_point = self._remaining_points.pop()
                return current_point