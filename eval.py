import torch
import numpy as np
from sklearn.metrics import confusion_matrix

pixel_threshold = 0.75

def formalize_skeleton(data):
	return data > pixel_threshold


def compute_metrics(y_true, y_pred):
	y_true = y_true.numpy()
	y_pred = formalize_skeleton(y_pred).numpy()


	y_true = np.array(y_true).reshape((256**2,))
	y_pred = np.array(y_pred).reshape((256**2,))

	tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

	precision = tp / (tp + fp)
	recall = tp / (tp + fn)
	f1_score = 2*((precision * recall)/(precision + recall))

	return precision, recall, f1_score
