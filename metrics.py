import matplotlib
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
import numpy as np
import json
import csv
import sys
import operator
import matplotlib.image as mpimg

output_folder = './metrics_output/'
models_present = ['basic','resnet','resnet_he_uniform','mobilenet']

def load_model_metrics(sheet_name):
  model = {
      'name': sheet_name,
      'train_imgs': [],
      'test_imgs': []
  }
  classes = []
  i = 2

  with open('./csv_output/%s.csv' % sheet_name) as f:
    reader = csv.reader(f, delimiter=',')
    first = 0
    for row in reader:
      if first == 0:
        first += 1
      else:
        if not ('activation' in model):
          model['activation'] = row[1]
        if not ('pretrained' in model):
          model['pretrained'] = row[2]

        qry = 'train_imgs' if row[3] == 'train' else 'test_imgs'
        ls = model[qry]
        ls.append({'image_path': row[5], 'ground_truth': row[6], 'prediction': row[7]})
        if not (row[6] in classes):
          classes.append(row[6])
        model[qry] = ls
  return model, classes

def display_error_samples(model_name, is_test, samples):
  imgs = list(map(lambda x: mpimg.imread('./classes/' + x['image_path']), samples))
  fig = plt.figure(figsize=(15,15))
  cols = 3
  rows = round(len(imgs) / cols) + 1
  axes = []
  for i in range(len(imgs)):
    axes.append(fig.add_subplot(rows, cols, i + 1))
    axes[-1].set_title("Ground Truth: %s\nPrediction: %s" % (samples[i]['ground_truth'], samples[i]['prediction']))
    plt.imshow(imgs[i])
    plt.axis('off')

  fig.tight_layout()    
  # plt.show()
  plt.savefig("%s%s_%s_errorsamples.png" % (output_folder, model_name, 'testing' if is_test else 'training'))

def display_model_conf_matrix(model, classes, is_test):
  qry = 'test_imgs' if is_test else 'train_imgs'
  cats = []
  samples = []

  for cl in classes:
    ls = []
    for cll in classes:
      ls.append(0)
    cats.append(ls)

  for each in model[qry]:
    gt = each['ground_truth']
    pred = each['prediction']

    cats[classes.index(gt)][classes.index(pred)] += 1
    if (classes.index(gt) != classes.index(pred) and cats[classes.index(gt)][classes.index(pred)] == 1):
      samples.append(each)

  acc = sum(cats[i][i] for i in range(len(cats))) / len(model[qry]) * 100
  accuracy = "{:.2f}%".format(acc)
  precisions = []
  f1 = []

  for i in range(len(cats)):
    prec = "{:.2f}%".format(cats[i][i] / sum(cats[j][i] for j in range(len(cats))) * 100)
    f = "{:.2f}%".format((2 * cats[i][i]) / (sum(cats[j][i] for j in range(len(cats))) + sum(cats[i][j] for j in range(len(cats)))) * 100)
    precisions.append(prec)
    f1.append(f)
  formatted_precisions = "\n".join(["%s - %s" % (classes[i], val) for i, val in enumerate(precisions)])
  formatted_f1 = "\n".join(["%s - %s" % (classes[i], val) for i, val in enumerate(f1)])

  title = "%s model using %s set" % (model['name'], 'testing' if is_test else 'training')
  df_cm = pd.DataFrame(cats, range(6), range(6))
  df_cm.index.name = 'Actual'
  df_cm.columns.name = 'Predicted\n\nAccuracy: %s\nPrecision:\n%s\nF1 Score:\n%s' % (accuracy, formatted_precisions, formatted_f1)
  plt.figure(figsize=(10,7))
  plt.title(title)
  sn.set(font_scale=1.4) # for label size
  sn.heatmap(df_cm, annot=True, annot_kws={"size": 16}, xticklabels=classes, yticklabels=classes) # font size

  # ytable = plt.table(cellText=[precisions, f1], rowLabels=['Precision', 'F1 Score'], colLabels=classes, loc="bottom", position=(0, 10))
  # plt.axis("off")
  # plt.grid(False)
  # plt.show()
  plt.savefig("%s%s_%s_confmatrix.png" % (output_folder, model['name'], 'testing' if is_test else 'training'))

  return acc, samples

def display_all_models(model_names, accs):
  accs = list(zip(*accs))
  x = np.arange(len(model_names))  # the label locations
  width = 0.2  # the width of the bars

  fig, ax = plt.subplots()
  rects1 = ax.bar(x - width / 2, accs[0], width, label='Test')
  rects2 = ax.bar(x + width / 2, accs[1], width, label='Train')

  # Add some text for labels, title and custom x-axis tick labels, etc.
  ax.set_ylabel('Accuracy')
  ax.set_title('Accuracy of each model')
  ax.set_xticks(x)
  ax.set_xticklabels(model_names)
  ax.legend()


  def autolabel(rects):
      """Attach a text label above each bar in *rects*, displaying its height."""
      for rect in rects:
          height = rect.get_height()
          ax.annotate('{:.2f}'.format(height),
                      xy=(rect.get_x() + rect.get_width() / 2, height),
                      xytext=(0, 3),  # 3 points vertical offset
                      textcoords="offset points",
                      ha='center', va='bottom')


  autolabel(rects1)
  autolabel(rects2)

  fig.tight_layout()
  fig_size = plt.gcf().get_size_inches()
  sizefactor = 1.5
  plt.gcf().set_size_inches(fig_size * sizefactor) 

  # plt.show()
  plt.savefig("%sgeneral_acc_compare.png" % output_folder)

def display_model_result(name):
  model_results, classes = load_model_metrics(name)
  test_acc, test_samples = display_model_conf_matrix(model_results, classes, True)
  display_error_samples(name, True, test_samples)
  train_acc, train_samples = display_model_conf_matrix(model_results, classes, False)
  display_error_samples(name, False, train_samples)

  return test_acc, train_acc


if len(sys.argv) < 2:
  print('Please indicate at least 1 model')
else:
  models = sys.argv[1:]
  filtered = [each for each in models if not each in models_present]
  if len(filtered) > 0:
    print('The following models were not found')
  else:
    accs = list(map(display_model_result, models))
    display_all_models(models, accs)