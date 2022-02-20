from models_mod import *
from load_data import *
import numpy as np
import sys, getopt
from tqdm import tqdm as tqdm
import pandas as pd

out_folder = "models"
cuda1 = torch.device('cuda:0')

# Command line options
out_folder = "models"

try:
    opts, args = getopt.getopt(sys.argv[1:], "ho:", ["out_folder="])
except getopt.GetoptError:
    print('HERE : Commad to run : python train_featurelossnet.py -o <out_folder>', flush=True)
    sys.exit(2)

for opt, arg in opts:
    if opt == '-h':
    
        print('Commad to run : python train_featurelossnet.py -o <out_folder>', flush=True)
        sys.exit()

    elif opt in ("-o", "--out_folder"):
        out_folder = arg

print('Folder to save the model ---- ' + out_folder, flush=True)


# Feature Loss Network Parameters
FEATURE_LOSS_LAYERS = 14
BASE_CHANNELS = 32
BASE_CHANNELS_INCREMENT = 5

# Network Setup
no_of_tasks = 0
no_of_classes = []
error_type = []
task_labels = []
layers_task = []
pred_output_task = []
loss_layer_task = []
optimizer_task = []

# Data Loading
file_names = []
labels = []
datasets = []
labels_lists = []
sets = ['train', 'val']


# Acoustic Scene Classification Task
no_of_tasks += 1

ase_labels, ase_names, ase_datasets, ase_labels_lists = loadASCData("dataset1/asc")
print(ase_labels_lists)
no_of_classes.append(len(ase_labels_lists))
error_type.append(1)

file_names.append({})
labels.append({})
datasets.append({})

# print(len(ase_names['train']))

for x in sets:
    file_names[no_of_tasks-1][x] = ase_names[x]
    labels[no_of_tasks-1][x] = ase_labels[x]
    datasets[no_of_tasks-1][x] = ase_datasets[x]

labels_lists.append(ase_labels_lists)

# Domestic Audio Tagging Task
no_of_tasks += 1

dat_labels, dat_names, dat_datasets, dat_labels_lists = loadDATData("dataset1/dat")
print(dat_labels_lists)
no_of_classes.append(len(dat_labels_lists))
error_type.append(2)

file_names.append({})
labels.append({})
datasets.append({})

for x in sets:
    file_names[no_of_tasks-1][x] = dat_names[x]
    labels[no_of_tasks-1][x] = dat_labels[x]
    datasets[no_of_tasks-1][x] = dat_datasets[x]

labels_lists.append(dat_labels_lists)

datasets = [{k:torch.tensor(i[k], device = cuda1) for k in sets} for i in datasets]

print("Data loading completed !!! ", flush=True)
# Epoch Initialization

train_error = []
test_error = []
threshold_error = []

for task in range(no_of_tasks):
    train_error.append(np.zeros(len(file_names[task]['train'])))
    test_error.append(np.zeros(len(file_names[task]['val'])))
    threshold_error.append(0.5 * np.ones(len(labels_lists[task])))

MAX_NUM_FILE = 0

training_prediction_label = []
training_true_label = []
test_prediction_label = []
test_true_label = []

for task in range(no_of_tasks):
    MAX_NUM_FILE = np.maximum(MAX_NUM_FILE, len(file_names[task]['train']))
    
    # Training Data
    training_prediction_label.append([])
    training_true_label.append([])
    
    for file in range(len(file_names[task]['train'])):
        training_prediction_label[task].append(np.zeros((0, len(labels_lists[task]))))
        training_true_label[task].append(np.zeros((0, len(labels_lists[task]))))
    
    # Testing Data
    test_prediction_label.append([])
    test_true_label.append([])
    
    for file in range(len(file_names[task]['val'])):
        test_prediction_label[task].append(np.zeros((0, len(labels_lists[task]))))
        test_true_label[task].append(np.zeros((0, len(labels_lists[task]))))

        
def init_weights(m):
	if type(m) == nn.Conv2d:
		torch.nn.init.xavier_uniform_(m.weight)
		m.bias.data.fill_(0.0)
             
print(MAX_NUM_FILE)
# Epoch Loop
Epochs = 2500
feature_loss_model = FeatureLossNet().cuda(cuda1)
learning_rate = 1e-4
optimizer = []
output_conv = []
loss = [nn.CrossEntropyLoss(), nn.BCEWithLogitsLoss()]
load_saved_model = False

for task in range(no_of_tasks):
    optimizer.append(torch.optim.Adam(feature_loss_model.parameters(), lr=learning_rate))

if load_saved_model:
    model_state = torch.load(out_folder + "/loss_model_without_reg.pth")
    feature_loss_model.load_state_dict(model_state)
    print('old model loaded........................................')
    optimizer_state = torch.load('./optimizer/' + "/optimizer_without_reg.pth")
    optimizer[0].load_state_dict(optimizer_state['task1_optimizer'])
    optimizer[1].load_state_dict(optimizer_state['task2_optimizer'])
    print('old optimizer loaded........................................')
else:
    feature_loss_model.apply(init_weights)

for epoch in tqdm(range(1, Epochs+1)):
    
    feature_loss_model.train()
    print("\n", flush=True)
    print("################################################## Epoch " + str(epoch) + " started ########################################################", flush=True)
    ids = []

    for task in range(no_of_tasks):
        ids.append(np.random.permutation(len(file_names[task]['train'])))

    for idx in range(MAX_NUM_FILE*no_of_tasks):
        task = idx % no_of_tasks
        if task == 0:
            token = idx//2
            file_id = ids[task][token%len(ids[task])]
        else:
            token = (idx-1)//2
            file_id = ids[task][token%len(ids[task])]
            
        input_data = datasets[task]['train'][file_id]
        data_size = torch.prod(torch.tensor(input_data.shape))

        #input_data = torch.tensor(input_data).cuda(cuda1)
        #data_shape = np.shape(input_data)
        
        min_width = 2**(FEATURE_LOSS_LAYERS + 1) - 1
        top_width = 1.*data_size
        max_width = top_width
        
        exponent = np.random.uniform(np.log10(min_width - 0.5), np.log10(max_width + 0.5))
        width = int(np.round(10. ** exponent))
        start_point = np.random.randint(0, data_size - width + 1)
        input_data = input_data[:, :, :, start_point:start_point + width]
        input_label = torch.tensor(np.reshape(1. * np.array([(l in labels[task]['train'][file_id]) for l in labels_lists[task]]), (1,-1)), dtype=torch.long).cuda(cuda1)
        
        prediction, outputs = feature_loss_model(input_data, task)
        if task == 0:
            cross_entropy_loss = loss[task](outputs[-1], torch.max(input_label,1)[1])
        else:
            cross_entropy_loss = loss[task](outputs[-1], input_label.type_as(outputs[-1]))
        
        optimizer[task].zero_grad()
        cross_entropy_loss.backward()
        optimizer[task].step()
        train_error[task][file_id] = cross_entropy_loss.item()
        prediction = prediction.cpu().detach().numpy()
        training_prediction_label[task][file_id] = np.reshape(prediction, [1,-1])
        training_true_label[task][file_id] = input_label.cpu().detach().numpy()

    ################################### ALL TRAINING ERROS COMPUTATIONS ##################################### 
    # print("\n")
    to_print = "TRAINING ERRORS : \n\n"

    for task in range(no_of_tasks):
        to_print += "Training Error for task " + str(task) + " = "
        to_print += "%.6f " % (np.mean(train_error[task][np.where(train_error[task])]))
        to_print += "\n"

        if error_type[task] == 1:
            # Mean classification error
            to_print += "Mean Classification accuracy for task " + str(task) + " = "
            to_print += "%.6f " % (np.mean(1.0 * (np.argmax(np.vstack(training_prediction_label[task]),axis=1) == np.argmax(np.vstack(training_true_label[task]),axis=1))))
            to_print += "\n"

        elif error_type[task] == 2:
            to_print += "Mean Equal error for task " + str(task) + " = "
            eq_error_rate = 0.

            for n1, label in enumerate(labels_lists[task]):
                thres = np.array([0.,1.,0.])
                fp = 1
                fn = 0

                while abs(fp-fn) > 1e-4 and abs(np.diff(thres[:-1])) > 1e-10:
                    thres[-1] = np.mean(thres[:-1])
                    fp = (np.sum((np.vstack(training_prediction_label[task])[:,n1] > thres[-1]) * (1.-np.vstack(training_true_label[task])[:,n1]))) / (1e-15 + np.sum(1.-np.vstack(training_true_label[task])[:,n1]))
                    fn = (np.sum((np.vstack(training_prediction_label[task])[:,n1] <= thres[-1]) * np.vstack(training_true_label[task])[:,n1])) / (1e-15 + np.sum(np.vstack(training_true_label[task])[:,n1]))

                    if fp < fn:
                        thres[1] = thres[-1]
                    else:
                        thres[0] = thres[-1]

                threshold_error[task][n1] = thres[-1]
                eq_error_rate += (fp+fn)/2

            eq_error_rate /= len(labels_lists[task])
            to_print += "%.6f " % eq_error_rate
            to_print += "\n"


    print(to_print, flush=True)

    #### SAVE MODEL HERE
    if epoch%5==0:
        print("Saving the model .....", flush=True)
        torch.save(feature_loss_model.state_dict(), out_folder + "/loss_model_without_reg.pth")
        optimizer_state = {'task1_optimizer': optimizer[0].state_dict(), 'task2_optimizer': optimizer[1].state_dict()}
        torch.save(optimizer_state, './optimizer/' + "/optimizer_without_reg.pth")
        print("Model saving done", flush=True)
        print("\n", flush=True)
    
    
    if not epoch%25==0:
        print("################################################## Epoch " + str(epoch) + " ended ########################################################", flush=True)
        print("\n", flush=True)
        continue

    ################################################ VALIDATION LOOP ######################################################
    print("------------------------ Validation loop started ------------------------", flush=True)
    print("\n", flush=True)
    feature_loss_model.eval()

    for task in range(no_of_tasks):

        for idx in range(len(file_names[task]['val'])):
            file_id = idx
            input_data = datasets[task]['val'][file_id]
            input_label = torch.tensor(np.reshape(1. * np.array([(l in labels[task]['val'][file_id]) for l in labels_lists[task]]), (1,-1)), dtype=torch.long).cuda(cuda1)

            prediction, outputs = feature_loss_model(input_data, task)

            if task == 0:
                cross_entropy_loss = loss[task](outputs[-1], torch.max(input_label,1)[1])
            else:
                cross_entropy_loss = loss[task](outputs[-1], input_label.type_as(outputs[-1]))

            prediction = prediction.cpu().detach().numpy()
            test_error[task][file_id] = cross_entropy_loss.item()
            test_prediction_label[task][file_id] = prediction
            test_true_label[task][file_id] = input_label.cpu().detach().numpy()

    ################################### ALL VALIDATION ERROS COMPUTATIONS ##################################### 
    to_print = "\n"
    to_print += "VALIDATION ERROS : \n"

    for task in range(no_of_tasks):
        to_print += "Validation Error for task " + str(task) + " = "
        to_print += "%.6f " % (np.mean(test_error[task][np.where(test_error[task])]))
        to_print += "\n"

        if error_type[task] == 1:
            to_print += "Mean Classification accuracy for task " + str(task) + " = "
            to_print += "%.6f " % (np.mean(1.0 * (np.argmax(np.vstack(test_prediction_label[task]), axis=1) == np.argmax(np.vstack(test_true_label[task]), axis=1))))
            to_print += "\n"

        elif error_type[task] == 2:
            to_print += "Mean Equal error for task " + str(task) + " = "
            eq_error_rate = 0

            for n1, label in enumerate(labels_lists[task]):
                thres = np.array([0.,1.,.0])
                fp = 1
                fn = 0

                while abs(fp-fn) > 1e-4 and abs(np.diff(thres[:-1])) > 1e-10:
                    thres[-1] = np.mean(thres[:-1])
                    fp = (np.sum((np.vstack(test_prediction_label[task])[:,n1] > thres[-1]) * (1.-np.vstack(test_true_label[task])[:,n1]))) / (1e-15 + np.sum(1.-np.vstack(test_true_label[task])[:,n1]))
                    fn = (np.sum((np.vstack(test_prediction_label[task])[:,n1] <= thres[-1]) * np.vstack(test_true_label[task])[:,n1])) / (1e-15 + np.sum(np.vstack(test_true_label[task])[:,n1]))

                    if fp < fn:
                        thres[1] = thres[-1]
                    else:
                        thres[0] = thres[-1]

                eq_error_rate += (fp+fn)/2

            eq_error_rate /= len(labels_lists[task])
            to_print += "%.6f " % (eq_error_rate)
            to_print += "\n"


    print(to_print, flush=True)

    print("################################################## Epoch " + str(epoch) + " ended ########################################################", flush=True)
    print("\n", flush=True)
