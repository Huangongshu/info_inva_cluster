import cv2
import numpy as np

from matplotlib import pyplot as plt
import seaborn as sns
plt.style.use('seaborn-whitegrid')

palette = sns.color_palette('hls', 10)
def convex_combo(clstr, label, ax, saveto):
  plt.cla()
  ax.set_xlim([-1.3,1.3])
  ax.set_ylim([-1.3,1.3])
  def get_coord(probs, num_classes=10):
    # computes coordinate for 1 sample based on probability distribution over c
    coords_total = np.zeros(2, dtype=np.float32)
    probs_sum = np.sum(probs)
    fst_angle = 0.
    for c in range(num_classes):
      # compute x, y coordinates
      coords = np.ones(2) * 2 * np.pi * (float(c) / num_classes) + fst_angle  #10个coords：（0，0）（0.628，0.628）（1.256，1.256）
      #（1.884，1.884）（2.513，2.513）（3.141，3.141）（3.769，3.769）（4.398，4.398）（5.026，5.026）（5.654，5.654）
      coords[0] = np.sin(coords[0]) #0.0 0.58 0.95 0.95 0.58 1.2e-16 -0.58 -0.95 -0.95 -0.58
      coords[1] = np.cos(coords[1]) #1.0 0.80 0.30 -0.3 -0.8 -1.0    -0.80 -0.30 0.309 0.809
      coords_total += (probs[c] / probs_sum) * coords
    return coords_total

  coord = np.stack([get_coord(c) for c in clstr], axis=1)
  x = coord[1,:]
  y = coord[0,:]

  # sel = np.random.binomial(1,0.15,size=len(x)) #选择部分进行显示
  sel = np.random.binomial(1,1,size=len(x))
  x     =     np.squeeze(x[np.argwhere(sel)])
  y     =     np.squeeze(y[np.argwhere(sel)])
  label = np.squeeze(label[np.argwhere(sel)])
  # x = np.squeeze(x)
  # y = np.squeeze(x)
  # label = np.squeeze(label)
  x = x + np.random.normal(0, 0.05, size=len(x))
  y = y + np.random.normal(0, 0.05, size=len(y))

  for k in range(10):
    ix = np.squeeze(label == k)
    # print('\t{}:{}'.format(k, ix.sum()))
    ax.scatter(x[ix], y[ix], s=1, alpha=0.5, c=[palette[k]] * ix.sum(), label='{}'.format(k))

  plt.legend(bbox_to_anchor= (1.2, 1.2))
  plt.savefig(saveto, bbox_inches='tight')
  
