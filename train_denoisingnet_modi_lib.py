from models_mod import *
from load_data_lib import *
from tqdm import tqdm as tqdm
from losses import *
import pandas as pd
import sys, getopt

cuda = torch.device('cuda:0')
# Denoising Network Parameters
DN_LAYERS = 13
DN_CHANNELS = 64
DN_LOSS_LAYERS = 6
DN_NORM_TYPE = "ADN" # Adaptive Batch Norm
DN_LOSS_TYPE = "FL" # Feature Loss

# Feature Loss Network
LOSS_LAYERS = 14
LOSS_BASE_CHANNELS = 32
LOSS_BASE_CHANNELS_INCREMENT = 5
LOSS_NORM_TYPE = "SBN" # Stochastic Batch Norm

SET_WEIGHT_EPOCH = 10
SAVE_MODEL_EPOCH = 10

print('This model does not use featureloss model in evaluation mode and use 6 layers for loss cal................')

# Command line options
data_folder = "/home/sauravpathak/data/NSDTSEA/"
loss_model_path = "./models/loss_model_without_reg_mod_data.pth"
output_folder = "./models/"

print('Dataset folder is "' + data_folder + '/"', flush=True)
print('Loss model path is "' + loss_model_path, flush=True)
print('Folder to save model is "' + output_folder + '/"', flush=True)

# Loading of data
training_set, validation_set = loadEntireDataList(dataFolder = data_folder)
training_set, validation_set = loadEntireData(training_set, validation_set)

#training_set['inaudio'] = [torch.tensor(i, device = cuda) for i in training_set['inaudio']]
#training_set['outaudio'] = [torch.tensor(i, device = cuda) for i in training_set['outaudio']]
#validation_set['inaudio'] = [torch.tensor(i, device = cuda) for i in validation_set['inaudio']]
#validation_set['outaudio'] = [torch.tensor(i, device = cuda) for i in validation_set['outaudio']]

if DN_LOSS_TYPE == "FL":
	loss_weights = np.ones(DN_LOSS_LAYERS)
else:
	loss_weights = []

if DN_LOSS_TYPE == "FL":
	training_loss = np.zeros((len(training_set["innames"]), DN_LOSS_LAYERS+2))
	validation_loss = np.zeros((len(validation_set["innames"]), DN_LOSS_LAYERS+2))
else:
	training_loss = np.zeros((len(training_set["innames"]), 1))
	validation_loss = np.zeros((len(validation_set["innames"]), 1))



#################################################### MODELS INITIALIZATION #################################################################
total_epochs = 320

def init_weights(m):
    if type(m) == nn.Conv2d:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.0)

denoising_model = DenoisingNet().cuda(cuda)
denoising_model.apply(init_weights)

feature_loss_model = FeatureLossNet().cuda(cuda)
feature_loss_model.load_state_dict(torch.load(loss_model_path))

learning_rate = 1e-4
optimizer = torch.optim.Adam(denoising_model.parameters(), lr=learning_rate)

l1_loss = nn.L1Loss()
l2_loss = nn.MSELoss()

######################################################## Spectrogram Extraction #########################################################

def get_spectrogram_db(raw_audio, sr = 16000, n_fft=2048, hop_length=512):
	raw_audio = raw_audio.cpu().detach().numpy()
	wav = np.squeeze(raw_audio)
	S = librosa.stft(wav, n_fft=n_fft)
	S = np.abs(S)
	S = librosa.amplitude_to_db(S, ref=np.max)
	return S

def spec_to_image(spec, eps=1e-6):
	mean = spec.mean()
	std = spec.std()
	spec_norm = (spec - mean) / (std + eps)
	spec_min, spec_max = spec_norm.min(), spec_norm.max()
	spec_scaled = 1 * (spec_norm - spec_min) / (spec_max - spec_min)
	return spec_scaled

def input_to_eff_net(raw_audio):
	S = spec_to_image(get_spectrogram_db(raw_audio))[np.newaxis,...]
	S = torch.as_tensor(S, device = cuda)
	S = torch.unsqueeze(S, 0)
	return S

######################################################## LOSS CALCULATION ##############################################################
def featureLoss(actualOutput, modelOutput, lossWeights):

	width_1 = actualOutput.shape[3]
	width_2 = modelOutput.shape[3]

	if width_1 > width_2:
		modelOutput = F.pad(modelOutput, (0, width_1 - width_2, 0, 0, 0, 0, 0, 0))

	else:
		actualOutput = F.pad(actualOutput, (0, width_2 - width_1, 0, 0, 0, 0, 0, 0))

	_ , model_output_vectors = feature_loss_model(modelOutput, 0)
	_ , actual_output_vectors = feature_loss_model(actualOutput, 0)

	loss_vectors = [0]

	for i in range(DN_LOSS_LAYERS):
		temp = l1_loss(model_output_vectors[i], actual_output_vectors[i]) / lossWeights[i]       
		loss_vectors.append(temp)
	temp = si_sdr_loss(actualOutput, modelOutput) / lossWeights[-1]
	loss_vectors.append(temp)

	for i in range(1,DN_LOSS_LAYERS+2): # DN_LOSS_LAYERS+2 if we include si_sdr loss
		loss_vectors[0] += loss_vectors[i]

	return loss_vectors

load_saved_model = False

if load_saved_model:
	model_state = torch.load(output_folder + "/denoising_model_modi_lib_R6.pth")
	denoising_model.load_state_dict(model_state)
	print('old model loaded........................................')
	optimizer_state = torch.load('./optimizer/' + "/denoising_optimizer_modi_lib_R6.pth")
	optimizer.load_state_dict(optimizer_state)
	print('old optimizer loaded........................................')


for epoch in tqdm(range(1, total_epochs+1)):
	tsi_sdr_losse = 0
	vsi_sdr_losse = 0
	tsdr_losse = 0
	vsdr_losse = 0
	tl2_losse = 0
	vl2_losse = 0
	tstsa_mse_losse = 0
	vstsa_mse_losse = 0

	print("\n", flush=True)
	print("################################################## Epoch " + str(epoch) + " started ########################################################", flush=True)

	training_ids = np.random.permutation(len(training_set["innames"]))

	############################################### Training Epoch ########################################################

	for id in range(0, len(training_ids)):

		index = training_ids[id]
		input_data = training_set["inaudio"][index]
		output_data = training_set["outaudio"][index]

		input_data = torch.tensor(input_data, device = cuda)
		output_data = torch.tensor(output_data, device = cuda)

		enhanced_data = denoising_model(input_data)

		loss = []

		if DN_LOSS_TYPE == "L1":
			loss = l1_loss(output_data, enhanced_data) + si_sdr_loss(output_data, enhanced_data)

		elif DN_LOSS_TYPE == "L2":
			loss = l2_loss(output_data, enhanced_data) + si_sdr_loss(output_data, enhanced_data)

		else:            
			loss = featureLoss(output_data, enhanced_data, loss_weights)
            
		width_1 = output_data.shape[3]
		width_2 = enhanced_data.shape[3]

		if width_1 > width_2:
			modelOutput = F.pad(enhanced_data.detach(), (0, width_1 - width_2, 0, 0, 0, 0, 0, 0))
			actualOutput = output_data.detach()        

		else:
			actualOutput = F.pad(output_data.detach(), (0, width_2 - width_1, 0, 0, 0, 0, 0, 0))
			modelOutput = enhanced_data.detach()  
 
		tsi_sdr_losse += si_sdr_loss(actualOutput, modelOutput).detach()
		tl2_losse += l2_loss(actualOutput, modelOutput).detach()
		tsdr_losse += sdr_loss(actualOutput, modelOutput).detach()
		tstsa_mse_losse += stsa_mse_loss(input_to_eff_net(actualOutput), input_to_eff_net(modelOutput)).detach()                           
        

		optimizer.zero_grad()
		loss[0].backward()      
		optimizer.step()

		training_loss[id][0] = loss[0].detach().item()

		if DN_LOSS_TYPE == "FL":
			for j in range(DN_LOSS_LAYERS+1): # DN_LOSS_LAYERS+1 if we include si_sdr loss
				training_loss[id][j+1] = loss[j+1] 

                
	####################################### Printing Training Errors ################################################

	to_print = "TRAINING ERRORS : \n"

	if DN_LOSS_TYPE == "FL":
		for j in range(DN_LOSS_LAYERS + 2): # DN_LOSS_LAYERS+2 if we include si_sdr loss
			to_print += "\n%10.6e" % (np.mean(training_loss, axis=0)[j])

	else:
		to_print += "\n%10.6e" % (np.mean(training_loss, axis=0)[0])
        
	to_print += "\nsi_sdr_loss: %10.6e" % (tsi_sdr_losse/len(training_ids))
	to_print += "\nl2_loss: %10.6e" % (tl2_losse/len(training_ids))       
	to_print += "\nsdr_loss: %10.6e" % (tsdr_losse/len(training_ids))
	to_print += "\nstsa_mse_loss: %10.6e" % (tstsa_mse_losse/len(training_ids))

	to_print += "\n"
	print(to_print, flush=True)


	###################################### Change loss weights #############################################

	if DN_LOSS_TYPE == "FL" and epoch == SET_WEIGHT_EPOCH:
		print("\nSetting loss weights for the loss calculation ....\n", flush=True)
		loss_weights = np.absolute(np.mean(training_loss, axis = 0)[1:])
		print("Weights has been set\n")
    

	###################################### Validation Epoch ###############################################
	if not epoch%10==0:
		print("************************************************ Epoch " + str(epoch) + " ended *********************************************", flush=True)
		print("\n", flush=True)
		continue

	print("------------------------ Validation loop started ------------------------", flush=True)
	print("\n", flush=True)
	with torch.no_grad():
		for id in range(0, len(validation_set['innames'])):

			index = id
			input_data = validation_set['inaudio'][index]
			output_data = validation_set['outaudio'][index]

			input_data = torch.tensor(input_data, device = cuda)
			output_data = torch.tensor(output_data, device = cuda)

			enhanced_data = denoising_model(input_data)

			loss = []

			if DN_LOSS_TYPE == "L1":
				loss = l1_loss(output_data, enhanced_data) + si_sdr_loss(output_data, enhanced_data)

			elif DN_LOSS_TYPE == "L2":
				loss = l2_loss(output_data, enhanced_data) + si_sdr_loss(output_data, enhanced_data)

			else:           
				loss = featureLoss(output_data, enhanced_data, loss_weights)                 
            
			width_1 = output_data.shape[3]
			width_2 = enhanced_data.shape[3]

			if width_1 > width_2:
				modelOutput = F.pad(enhanced_data.detach(), (0, width_1 - width_2, 0, 0, 0, 0, 0, 0))
				actualOutput = output_data.detach()

			else:
				actualOutput = F.pad(output_data.detach(), (0, width_2 - width_1, 0, 0, 0, 0, 0, 0))
				modelOutput = enhanced_data.detach()

			vsi_sdr_losse += si_sdr_loss(actualOutput, modelOutput).detach()
			vsdr_losse += sdr_loss(actualOutput, modelOutput).detach()
			vl2_losse += l2_loss(actualOutput, modelOutput).detach()                 
			vstsa_mse_losse += stsa_mse_loss(input_to_eff_net(actualOutput), input_to_eff_net(modelOutput)).detach()
            
			validation_loss[id][0] = loss[0].detach().item()


			if DN_LOSS_TYPE == "FL":
				for j in range(DN_LOSS_LAYERS+1): # DN_LOSS_LAYERS+1 if we include si_sdr loss
					validation_loss[id][j+1] = loss[j+1]


	####################################### Printing Validation Errors ################################################

		# to_print = "\n"
		to_print = "VALIDATION ERROS : \n"

		if DN_LOSS_TYPE == "FL":
			for j in range(DN_LOSS_LAYERS + 2): # DN_LOSS_LAYERS+2 if we include si_sdr loss
				to_print += "\n%10.6e" % (np.mean(validation_loss, axis=0)[j])

		else:
			to_print += "\n%10.6e" % (np.mean(validation_loss, axis=0)[0])

	to_print += "\nsi_sdr_loss: %10.6e" % (vsi_sdr_losse/len(validation_set['innames']))
	to_print += "\nl2_loss: %10.6e" % (vl2_losse/len(validation_set['innames']))       
	to_print += "\nsdr_loss: %10.6e" % (vsdr_losse/len(validation_set['innames']))  
	to_print += "\nstsa_mse_loss: %10.6e" % (vstsa_mse_losse/len(validation_set['innames']))

	to_print += "\n"
	print(to_print, flush=True)

	print("Saving the model and losses .....", flush=True)

	model_state = denoising_model.state_dict()
	optimizer_state = optimizer.state_dict()
	torch.save(model_state, output_folder + "/denoising_model_modi_lib_R6.pth")
	torch.save(optimizer_state, './optimizer/' + "/denoising_optimizer_modi_lib_R6.pth") 
    
	print("Model and Loss saving done", flush=True)
	print("\n", flush=True)


	print("************************************************ Epoch " + str(epoch) + " ended *********************************************", flush=True)
	print("\n", flush=True)