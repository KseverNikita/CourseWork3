from keras.models import Model
import numpy as np
import training
import preparation
import json
from matplotlib import pyplot as plt
from matplotlib.animation import ArtistAnimation

model_1 = training.model_1()
model_2 = training.model_2()

model_1 = load_weights('weights/weught_model_1.hdf5')
model_2 = load_weights('weights/weught_model_2.hdf5')
person = []

def plot_anim(arr_rig, preds):
	fig = plt.figure(figsize=(10,10))
	ax = fig.gca(projection='3d')
	min_val = min(np.min(arr_rig), np.min(preds))
	max_val = max(np.max(arr_rig), np.max(preds))
	colors = ["black", "lightpink", "r", "sienna", "olivedrab", "lightgreen", "c", "orange", "purple", "blue", "green", "mediumpurple", "salmon", "grey"] 
	rigs = [[0, 1], [1, 2], [2, 3], [3, 4], [1, 5], [5, 6], [6, 7], [1, 14], [8, 14], [8, 9], [9, 10], [11, 14], [11, 12], [12, 13]]
	for i in range(len(rigs)):
    	ax.plot([preds[rigs[i][0]], preds[rigs[i][1]]], [x_y[rigs[i][0]][0], x_y[rigs[i][1]][0]], [-x_y[rigs[i][0]][1], -x_y[rigs[i][1]][1]] ,linewidth=1.5, label = "{}-{}".format(rigs[i][0], rigs[i][1]), color = colors[i])
	ax.plot([min_val, max_val], [0, 0], [0, 0], alpha=0.0)
	ax.plot([0, 0], [min_val, max_val], [0, 0], alpha=0.0)
	ax.plot([0, 0], [0, 0], [min_val, max_val], alpha=0.0)
	plt.legend()
	plt.show()

def play_anim(x_y, preds):
	min_val = min(np.min(x_y), np.min(preds))
	max_val = max(np.max(x_y), np.max(preds))
	fig = plt.figure(figsize = (10, 10))
	ax = plt.axes(xlim=(min_val, max_val), ylim=(min_val, max_val), zlim=(min_val, max_val), projection='3d')
	ax.set_zlabel('Z')
	ax.set_ylabel('Y')
	ax.set_xlabel('X')

	new_data = []
	rigs = [[0, 1], [1, 2], [2, 3], [3, 4], [1, 5], [5, 6], [6, 7], [1, 14], [8, 14], [8, 9], [9, 10], [11, 14], [11, 12], [12, 13]]
	colors = ["black", "lightpink", "r", "sienna", "olivedrab", "lightgreen", "c", "orange", "purple", "blue", "green", "mediumpurple", "salmon", "grey"] 
	for frame in range(119):
    	new_data.append([])
    	for i in range(14):
       		new_data[frame].append(ax.plot([preds[frame][rigs[i][0]], preds[frame][rigs[i][1]]], [x_y[frame][rigs[i][0]][0], x_y[frame][rigs[i][1]][0]], [-x_y[frame][rigs[i][0]][1], -x_y[frame][rigs[i][1]][1]] ,linewidth=1.5, color = colors[i])[0])
	anim = ArtistAnimation(fig, new_data, interval=200, blit=True)
	anim.save('/animation.mp4', fps=20, extra_args=['-vcodec', 'libx264'])
	plt.show()

with open("file.json") as f:
    file_json = json.load(f)
person.append(np.array(file_json["people"][0]["pose_keypoints_2d"]).reshape((25, 3))[:15])

if (person.shape[0] == 1):
	arr_rig = np.zeros((15,2))
	for i in range(14):
    	if (i <= 7) :
        	arr_rig[i] = np.array([person[0][i][0], person[0][i][1]])
    	else:
        	arr_rig[i] = np.array([person[0][i + 1][0], person[0][i + 1][1]])
	arr_rig[14] = np.array([person[0][8][0], person[0][8][1]])
	x_y = normalize_data(arr_rig, "frame") 
	preds = model_1.predict(x_y.reshape(1, 30))[0]
	plot_anim(x_y, preds)
else:
	arr_rigs = np.zeros((person.shape[0], 15, 2))
	for j in range(person.shape[0]):
    	for i in range(14):
        	if (i <= 7) :
            	arr_rigs[j][i] = np.array([person[j][i][0], person[j][i][1]])
        	else:
            	arr_rigs[j][i] = np.array([person[j][i + 1][0], person[j][i + 1][1]])
    	arr_rigs[j][14] = np.array([person[j][8][0], person[j][8][1]])
    x_y = normalize_data(arr_rigs, "sequence")
    preds = model.predict(x_y.reshape(reshape(person.shape[0], 30)))
    play_anim(x_y, preds)




