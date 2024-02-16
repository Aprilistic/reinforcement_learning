#!/usr/bin/env python

import numpy as np

__all__ = ['ActionValueBanditAgent']

class ActionValueBanditAgent(object):
	def __init__(self, estimators, actor):
		self.estimators = estimators
		self.actor = actor

	def was_exploring(self):
		return self.actor.explore
	
	def update(self, action, reward):
		self.estimators[action].update(reward)
		return None
	
	def get_estimates(self):
		return np.array([x.value for x in self.estimators])
	
	def get_optimal_actions(self):
		values = self.get_estimates()
		return np.where(values == values.max())[0]
	
	def action(self):
		optimal_actions = self.get_optimal_actions()
		return self.actor.action(optimal_actions)