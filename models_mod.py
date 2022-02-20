import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class FeatureLossNet(nn.Module):

	def __init__(self, numLayers=14, baseChannels=32, baseChannelsIncrement=5, inputChannels=1, kernelSize=3):

		super(FeatureLossNet, self).__init__()

		outputChannels = baseChannels # 32
        
		self.conv1 = nn.Conv2d(inputChannels, outputChannels, kernel_size=(1,3), stride=(1,2), padding=(0,1)) # in_ch=1, out_ch=32
		self.conv1_bn = nn.BatchNorm2d(outputChannels,eps=1e-03, momentum=0.001)
		self.conv2 = nn.Conv2d(outputChannels, outputChannels, kernel_size=(1,3), stride=(1,2), padding=(0,1)) # in_ch=32, out_ch=32
		self.conv2_bn = nn.BatchNorm2d(outputChannels,eps=1e-03, momentum=0.001)
		self.conv3 = nn.Conv2d(outputChannels, outputChannels, kernel_size=(1,3), stride=(1,2), padding=(0,1)) # in_ch=32, out_ch=32
		self.conv3_bn = nn.BatchNorm2d(outputChannels,eps=1e-03, momentum=0.001)
		self.conv4 = nn.Conv2d(outputChannels, outputChannels, kernel_size=(1,3), stride=(1,2), padding=(0,1)) # in_ch=32, out_ch=32
		self.conv4_bn = nn.BatchNorm2d(outputChannels,eps=1e-03, momentum=0.001)
		self.conv5 = nn.Conv2d(outputChannels, outputChannels, kernel_size=(1,3), stride=(1,2), padding=(0,1)) # in_ch=32, out_ch=32
		self.conv5_bn = nn.BatchNorm2d(outputChannels,eps=1e-03, momentum=0.001)
         # https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html

		inputChannels = outputChannels # 32
		outputChannels = outputChannels * 2 # 64

		self.conv6 = nn.Conv2d(inputChannels, outputChannels, kernel_size=(1,3), stride=(1,2), padding=(0,1)) # in_ch=32, out_ch=64
		self.conv6_bn = nn.BatchNorm2d(outputChannels,eps=1e-03, momentum=0.001)
		self.conv7 = nn.Conv2d(outputChannels, outputChannels, kernel_size=(1,3), stride=(1,2), padding=(0,1)) # in_ch=64, out_ch=64
		self.conv7_bn = nn.BatchNorm2d(outputChannels,eps=1e-03, momentum=0.001)
		self.conv8 = nn.Conv2d(outputChannels, outputChannels, kernel_size=(1,3), stride=(1,2), padding=(0,1)) # in_ch=64, out_ch=64
		self.conv8_bn = nn.BatchNorm2d(outputChannels,eps=1e-03, momentum=0.001)
		self.conv9 = nn.Conv2d(outputChannels, outputChannels, kernel_size=(1,3), stride=(1,2), padding=(0,1)) # in_ch=64, out_ch=64
		self.conv9_bn = nn.BatchNorm2d(outputChannels,eps=1e-03, momentum=0.001)
		self.conv10 = nn.Conv2d(outputChannels, outputChannels, kernel_size=(1,3), stride=(1,2), padding=(0,1)) # in_ch=64, out_ch=64
		self.conv10_bn = nn.BatchNorm2d(outputChannels,eps=1e-03, momentum=0.001)

		inputChannels = outputChannels # 64
		outputChannels = outputChannels * 2 # 128

		self.conv11 = nn.Conv2d(inputChannels, outputChannels, kernel_size=(1,3), stride=(1,2), padding=(0,1)) # in_ch=64, out_ch=128
		self.conv11_bn = nn.BatchNorm2d(outputChannels,eps=1e-03, momentum=0.001)
		self.conv12 = nn.Conv2d(outputChannels, outputChannels, kernel_size=(1,3), stride=(1,2), padding=(0,1)) # in_ch=128, out_ch=128
		self.conv12_bn = nn.BatchNorm2d(outputChannels,eps=1e-03, momentum=0.001)
		self.conv13 = nn.Conv2d(outputChannels, outputChannels, kernel_size=(1,3), stride=(1,2), padding=(0,1)) # in_ch=128, out_ch=128
		self.conv13_bn = nn.BatchNorm2d(outputChannels,eps=1e-03, momentum=0.001)
		self.conv14 = nn.Conv2d(outputChannels, outputChannels, kernel_size=(1,3), padding=(0,1))  # in_ch=128, out_ch=128; stride=1
		self.conv14_bn = nn.BatchNorm2d(outputChannels,eps=1e-03, momentum=0.001)

		self.conv15 = nn.Conv2d(outputChannels, 15, kernel_size=(1,1), padding=0) # in_ch=128, out_ch=15; stride=1

		self.conv16 = nn.Conv2d(outputChannels, 7, kernel_size=(1,1), padding=0) # in_ch=128, out_ch=7; stride=1

		self.softmax = nn.Softmax(dim = 1) # channel=7, Apply softmax along dimension 1 
		self.sigmoid = nn.Sigmoid()

	def forward(self, myinput, tasknum): # -->[1,1,1,N]
		allOutputs = []
		output = F.leaky_relu(self.conv1_bn(self.conv1(myinput)),negative_slope=0.2) # [1,1,1,N] --> [1,32,1,N/2]; ceiling in N/2
		allOutputs.append(output)
		output = F.leaky_relu(self.conv2_bn(self.conv2(output)),negative_slope=0.2) # [1,32,1,N/2] --> [1,32,1,N/4]; ceiling
		allOutputs.append(output)
		output = F.leaky_relu(self.conv3_bn(self.conv3(output)),negative_slope=0.2) # [1,32,1,N/4] --> [1,32,1,N/8]
		allOutputs.append(output)
		output = F.leaky_relu(self.conv4_bn(self.conv4(output)),negative_slope=0.2) # [1,32,1,N/8] --> [1,32,1,N/16]
		allOutputs.append(output)
		output = F.leaky_relu(self.conv5_bn(self.conv5(output)),negative_slope=0.2) # [1,32,1,N/16] --> [1,32,1,N/32]
		allOutputs.append(output)
		output = F.leaky_relu(self.conv6_bn(self.conv6(output)),negative_slope=0.2) # [1,32,1,N/32] --> [1,64,1,N/64]
		allOutputs.append(output)
		output = F.leaky_relu(self.conv7_bn(self.conv7(output)),negative_slope=0.2) # [1,64,1,N/64] --> [1,64,1,N/128]
		allOutputs.append(output)
		output = F.leaky_relu(self.conv8_bn(self.conv8(output)),negative_slope=0.2) # [1,64,1,N/128] --> [1,64,1,N/256]
		allOutputs.append(output)
		output = F.leaky_relu(self.conv9_bn(self.conv9(output)),negative_slope=0.2) # [1,64,1,N/256] --> [1,64,1,N/512]
		allOutputs.append(output)
		output = F.leaky_relu(self.conv10_bn(self.conv10(output)),negative_slope=0.2) # [1,64,1,N/512] --> [1,64,1,N/1024]
		allOutputs.append(output)
		output = F.leaky_relu(self.conv11_bn(self.conv11(output)),negative_slope=0.2) # [1,64,1,N/1024] --> [1,128,1,N/2048]
		allOutputs.append(output)
		output = F.leaky_relu(self.conv12_bn(self.conv12(output)),negative_slope=0.2) # [1,128,1,N/2048] --> [1,128,1,N/4096]
		allOutputs.append(output)
		output = F.leaky_relu(self.conv13_bn(self.conv13(output)),negative_slope=0.2) # [1,128,1,N/4096] --> [1,128,1,N/8192]
		allOutputs.append(output)
		output = F.leaky_relu(self.conv14_bn(self.conv14(output)),negative_slope=0.2) # [1,128,1,N/8192] --> [1,128,1,N/8192]
                                                                                   # because stride is (1,1) instead of (1,2)
		allOutputs.append(output)
        
		avg_features = torch.mean(output, 3, True) # take mean of activations along dim 3; # [1,128,1,N/8192] --> [1,128,1,1]
		if tasknum == 0:
			output = self.conv15(avg_features) # [1,128,1,1] --> [1,15,1,1]
			output = torch.reshape(output, [1,15]) # [1,15,1,1] --> [1,15]
			allOutputs.append(output)
			prediction = self.softmax(output)
			return prediction, allOutputs

		elif tasknum == 1:
			output = self.conv16(avg_features) # [1,128,1,1] --> [1,7,1,1]
			output = torch.reshape(output, [1,7])  # [1,7,1,1] --> [1,7]
			allOutputs.append(output)
			prediction = self.sigmoid(output)
			return prediction, allOutputs

class DenoisingNet(nn.Module):

	def __init__(self, numLayers=13, baseChannels=32, inputChannels=1, kernelSize=3):
		super(DenoisingNet, self).__init__()
        

		self.conv1 = nn.Conv2d(inputChannels, baseChannels, kernel_size=(1,3), padding=(0,1)) # in_ch=1, out_ch=32
		self.conv1_bn = nn.BatchNorm2d(baseChannels,eps=1e-03, momentum=0.001)
		self.an1 = nn.Parameter(torch.rand(2,1)) # not initialized at alpha = 1 and beta = 0
		self.conv2 = nn.Conv2d(baseChannels, baseChannels, kernel_size=(1,3), padding=(0,1)) # in_ch=32, out_ch=32
		self.conv2_bn = nn.BatchNorm2d(baseChannels,eps=1e-03, momentum=0.001)
		self.an2 = nn.Parameter(torch.rand(2,1))
		self.conv3 = nn.Conv2d(baseChannels, baseChannels, kernel_size=(1,3), padding=(0,1)) # in_ch=32, out_ch=32
		self.conv3_bn = nn.BatchNorm2d(baseChannels,eps=1e-03, momentum=0.001)
		self.an3 = nn.Parameter(torch.rand(2,1))
		self.conv4 = nn.Conv2d(baseChannels, baseChannels, kernel_size=(1,3), padding=(0,1)) # in_ch=32, out_ch=32
		self.conv4_bn = nn.BatchNorm2d(baseChannels,eps=1e-03, momentum=0.001)
		self.an4 = nn.Parameter(torch.rand(2,1))
		self.conv5 = nn.Conv2d(baseChannels, baseChannels, kernel_size=(1,3), padding=(0,1)) # in_ch=32, out_ch=32
		self.conv5_bn = nn.BatchNorm2d(baseChannels,eps=1e-03, momentum=0.001)
		self.an5 = nn.Parameter(torch.rand(2,1))
		self.conv6 = nn.Conv2d(baseChannels, baseChannels, kernel_size=(1,3), padding=(0,1)) # in_ch=32, out_ch=32
		self.conv6_bn = nn.BatchNorm2d(baseChannels,eps=1e-03, momentum=0.001)
		self.an6 = nn.Parameter(torch.rand(2,1))
		self.conv7 = nn.Conv2d(baseChannels, baseChannels, kernel_size=(1,3), padding=(0,1)) # in_ch=32, out_ch=32
		self.conv7_bn = nn.BatchNorm2d(baseChannels,eps=1e-03, momentum=0.001)
		self.an7 = nn.Parameter(torch.rand(2,1))
		self.conv8 = nn.Conv2d(baseChannels, baseChannels, kernel_size=(1,3), padding=(0,1)) # in_ch=32, out_ch=32
		self.conv8_bn = nn.BatchNorm2d(baseChannels,eps=1e-03, momentum=0.001)
		self.an8 = nn.Parameter(torch.rand(2,1))
		self.conv9 = nn.Conv2d(baseChannels, baseChannels, kernel_size=(1,3), padding=(0,1)) # in_ch=32, out_ch=32
		self.conv9_bn = nn.BatchNorm2d(baseChannels,eps=1e-03, momentum=0.001)
		self.an9 = nn.Parameter(torch.rand(2,1))
		self.conv10 = nn.Conv2d(baseChannels, baseChannels, kernel_size=(1,3), padding=(0,1)) # in_ch=32, out_ch=32
		self.conv10_bn = nn.BatchNorm2d(baseChannels,eps=1e-03, momentum=0.001)
		self.an10 = nn.Parameter(torch.rand(2,1))
		self.conv11 = nn.Conv2d(baseChannels, baseChannels, kernel_size=(1,3), padding=(0,1)) # in_ch=32, out_ch=32
		self.conv11_bn = nn.BatchNorm2d(baseChannels,eps=1e-03, momentum=0.001)
		self.an11 = nn.Parameter(torch.rand(2,1))
		self.conv12 = nn.Conv2d(baseChannels, baseChannels, kernel_size=(1,3), padding=(0,1)) # in_ch=32, out_ch=32
		self.conv12_bn = nn.BatchNorm2d(baseChannels,eps=1e-03, momentum=0.001)
		self.an12 = nn.Parameter(torch.rand(2,1))
		self.conv13 = nn.Conv2d(baseChannels, baseChannels, kernel_size=(1,3), padding=(0,1)) # in_ch=32, out_ch=32
		self.conv13_bn = nn.BatchNorm2d(baseChannels,eps=1e-03, momentum=0.001)
		self.an13 = nn.Parameter(torch.rand(2,1))
		self.conv14 = nn.Conv2d(baseChannels, baseChannels, kernel_size=(1,3), padding=(0,1)) # in_ch=32, out_ch=32
		self.conv14_bn = nn.BatchNorm2d(baseChannels,eps=1e-03, momentum=0.001)
		self.an14 = nn.Parameter(torch.rand(2,1))
		self.conv15 = nn.Conv2d(baseChannels, 1, kernel_size=(1,1), padding=(0,1)) # in_ch=32, out_ch=1




	def signalDilation(self, signal, channels, dilation):
		signal_shape = signal.shape
		num_elements_to_pad = dilation - 1 - (signal_shape[3] + dilation - 1) % dilation
		dilated_signal = F.pad(signal, (0, num_elements_to_pad, 0, 0, 0, 0, 0, 0))
		dilated_signal = torch.reshape(dilated_signal, (signal_shape[0], channels, -1, dilation))
		return torch.transpose(dilated_signal, 2, 3), num_elements_to_pad



	def inverseSignalDilation(self, dilated_signal, channels, toPad):
		signal_shape = dilated_signal.shape
		dilated_signal = torch.transpose(dilated_signal, 2, 3)
		dilated_signal = torch.reshape(dilated_signal, (signal_shape[0], channels, 1, -1))
		return dilated_signal[:,:,:,:signal_shape[2] * signal_shape[3] - toPad]



	def forward(self, myinput):
		baseChannels = 32
		output = F.leaky_relu(self.an1[0]*myinput + self.an1[1]*self.conv1_bn(self.conv1(myinput)),negative_slope=0.2)
# 		print(torch.tensor(output).size()) # [1,1,1,N] --> [1,32,1,N]
        
		dilation_depth = 2
		dilated_input, padding = self.signalDilation(output, channels=baseChannels, dilation=dilation_depth)
# 		print(torch.tensor(dilated_input).size(),padding) # [1,32,1,N] --> [1,32,2,N/2]
		dilated_output = F.leaky_relu(self.an2[0]*dilated_input + self.an2[1]*self.conv2_bn(self.conv2(dilated_input)),negative_slope=0.2)
# 		print(torch.tensor(dilated_output).size()) # [1,32,2,N/2] --> [1,32,2,N/2]
		output = self.inverseSignalDilation(dilated_output, channels=baseChannels, toPad=padding)
# 		print(torch.tensor(output).size()) # [1,32,2,N/2] --> [1,32,1,N]

		dilation_depth *= 2 # 4
		dilated_input, padding = self.signalDilation(output, channels=baseChannels, dilation=dilation_depth)
# 		print(torch.tensor(dilated_input).size(),padding) # [1,32,1,N] --> [1,32,4,N/4]
		dilated_output = F.leaky_relu(self.an3[0]*dilated_input + self.an3[1]*self.conv3_bn(self.conv3(dilated_input)),negative_slope=0.2)
# 		print(torch.tensor(dilated_output).size()) # [1,32,4,N/4] --> [1,32,4,N/4]
		output = self.inverseSignalDilation(dilated_output, channels=baseChannels, toPad=padding)
# 		print(torch.tensor(output).size()) # [1,32,4,N/4] --> [1,32,1,N]
        
		dilation_depth *= 2 # 8
		dilated_input, padding = self.signalDilation(output, channels=baseChannels, dilation=dilation_depth)
# 		print(torch.tensor(dilated_input).size(),padding) # [1,32,1,N] --> [1,32,8,N/8]
		dilated_output = F.leaky_relu(self.an4[0]*dilated_input + self.an4[1]*self.conv4_bn(self.conv4(dilated_input)),negative_slope=0.2)
# 		print(torch.tensor(dilated_output).size()) # [1,32,8,N/8] --> [1,32,8,N/8]
		output = self.inverseSignalDilation(dilated_output, channels=baseChannels, toPad=padding)
# 		print(torch.tensor(output).size()) # [1,32,8,N/8] --> [1,32,1,N]
        
		dilation_depth *= 2 # 16
		dilated_input, padding = self.signalDilation(output, channels=baseChannels, dilation=dilation_depth)
# 		print(torch.tensor(dilated_input).size(),padding) # [1,32,1,N] --> [1,32,16,N/16]
		dilated_output = F.leaky_relu(self.an5[0]*dilated_input + self.an5[1]*self.conv5_bn(self.conv5(dilated_input)),negative_slope=0.2)
# 		print(torch.tensor(dilated_output).size()) # [1,32,16,N/16] --> [1,32,16,N/16]
		output = self.inverseSignalDilation(dilated_output, channels=baseChannels, toPad=padding)
# 		print(torch.tensor(output).size()) # [1,32,16,N/16] --> [1,32,1,N]
        
		dilation_depth *= 2 # 32
		dilated_input, padding = self.signalDilation(output, channels=baseChannels, dilation=dilation_depth)
# 		print(torch.tensor(dilated_input).size(),padding) # [1,32,1,N] --> [1,32,32,N/32]
		dilated_output = F.leaky_relu(self.an6[0]*dilated_input + self.an6[1]*self.conv6_bn(self.conv6(dilated_input)),negative_slope=0.2)
# 		print(torch.tensor(dilated_output).size()) # [1,32,32,N/32] --> [1,32,32,N/32]
		output = self.inverseSignalDilation(dilated_output, channels=baseChannels, toPad=padding)
# 		print(torch.tensor(output).size()) # [1,32,32,N/32] --> [1,32,1,N]

		dilation_depth *= 2 # 64
		dilated_input, padding = self.signalDilation(output, channels=baseChannels, dilation=dilation_depth)
# 		print(torch.tensor(dilated_input).size(),padding) # [1,32,1,N] --> [1,32,64,N/64]
		dilated_output = F.leaky_relu(self.an7[0]*dilated_input + self.an7[1]*self.conv7_bn(self.conv7(dilated_input)),negative_slope=0.2)
# 		print(torch.tensor(dilated_output).size()) # [1,32,64,N/64] --> [1,32,64,N/64]
		output = self.inverseSignalDilation(dilated_output, channels=baseChannels, toPad=padding)
# 		print(torch.tensor(output).size()) # [1,32,64,N/64] --> [1,32,1,N]

		dilation_depth *= 2 # 128
		dilated_input, padding = self.signalDilation(output, channels=baseChannels, dilation=dilation_depth)
# 		print(torch.tensor(dilated_input).size(),padding) # [1,32,1,N] --> [1,32,128,N/128]
		dilated_output = F.leaky_relu(self.an8[0]*dilated_input + self.an8[1]*self.conv8_bn(self.conv8(dilated_input)),negative_slope=0.2)
# 		print(torch.tensor(dilated_output).size()) # [1,32,128,N/128] --> [1,32,128,N/128]
		output = self.inverseSignalDilation(dilated_output, channels=baseChannels, toPad=padding)
# 		print(torch.tensor(output).size()) # [1,32,128,N/128] --> [1,32,1,N]

		dilation_depth *= 2 # 256
		dilated_input, padding = self.signalDilation(output, channels=baseChannels, dilation=dilation_depth)
# 		print(torch.tensor(dilated_input).size(),padding) # [1,32,1,N] --> [1,32,256,N/256]
		dilated_output = F.leaky_relu(self.an9[0]*dilated_input + self.an9[1]*self.conv9_bn(self.conv9(dilated_input)),negative_slope=0.2)
# 		print(torch.tensor(dilated_output).size()) # [1,32,256,N/256] --> [1,32,256,N/256]
		output = self.inverseSignalDilation(dilated_output, channels=baseChannels, toPad=padding)
# 		print(torch.tensor(output).size()) # [1,32,256,N/256] --> [1,32,1,N]

		dilation_depth *= 2 # 512
		dilated_input, padding = self.signalDilation(output, channels=baseChannels, dilation=dilation_depth)
# 		print(torch.tensor(dilated_input).size(),padding) # [1,32,1,N] --> [1,32,512,N/512]
		dilated_output = F.leaky_relu(self.an10[0]*dilated_input + self.an10[1]*self.conv10_bn(self.conv10(dilated_input)),negative_slope=0.2)
# 		print(torch.tensor(dilated_output).size()) # [1,32,512,N/512] --> [1,32,512,N/512]
		output = self.inverseSignalDilation(dilated_output, channels=baseChannels, toPad=padding)
# 		print(torch.tensor(output).size()) # [1,32,512,N/512] --> [1,32,1,N]

		dilation_depth *= 2 # 1024
		dilated_input, padding = self.signalDilation(output, channels=baseChannels, dilation=dilation_depth)
# 		print(torch.tensor(dilated_input).size(),padding) # [1,32,1,N] --> [1,32,1024,N/1024]
		dilated_output = F.leaky_relu(self.an11[0]*dilated_input + self.an11[1]*self.conv11_bn(self.conv11(dilated_input)),negative_slope=0.2)
# 		print(torch.tensor(dilated_output).size()) # [1,32,1024,N/1024] --> [1,32,1024,N/1024]
		output = self.inverseSignalDilation(dilated_output, channels=baseChannels, toPad=padding)
# 		print(torch.tensor(output).size()) # [1,32,1024,N/1024] --> [1,32,1,N]
        
		dilation_depth *= 2 # 2048
		dilated_input, padding = self.signalDilation(output, channels=baseChannels, dilation=dilation_depth)
# 		print(torch.tensor(dilated_input).size(),padding) # [1,32,1,N] --> [1,32,2048,N/2048]
		dilated_output = F.leaky_relu(self.an12[0]*dilated_input + self.an12[1]*self.conv12_bn(self.conv12(dilated_input)),negative_slope=0.2)
# 		print(torch.tensor(dilated_output).size()) # [1,32,2048,N/2048] --> [1,32,2048,N/2048]
		output = self.inverseSignalDilation(dilated_output, channels=baseChannels, toPad=padding)
# 		print(torch.tensor(output).size()) # [1,32,2048,N/2048] --> [1,32,1,N]

		dilation_depth *= 2 # 4096
		dilated_input, padding = self.signalDilation(output, channels=baseChannels, dilation=dilation_depth)
# 		print(torch.tensor(dilated_input).size(),padding) # [1,32,1,N] --> [1,32,4096,N/4096]
		dilated_output = F.leaky_relu(self.an13[0]*dilated_input + self.an13[1]*self.conv13_bn(self.conv13(dilated_input)),negative_slope=0.2)
# 		print(torch.tensor(dilated_output).size()) # [1,32,4096,N/4096] --> [1,32,4096,N/4096]
		output = self.inverseSignalDilation(dilated_output, channels=baseChannels, toPad=padding)
# 		print(torch.tensor(output).size()) # [1,32,4096,N/4096] --> [1,32,1,N]

                
		output = F.leaky_relu(self.an14[0]*output + self.an14[1]*self.conv14_bn(self.conv14(output)),negative_slope=0.2)
# 		print(torch.tensor(output).size()) # [1,32,1,N] --> [1,32,1,N]
        
		output = self.conv15(output)
# 		print(torch.tensor(output).size()) # [1,32,1,N] --> [1,32,1,N]
        
		return output