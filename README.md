# semi-blind-trace-deep-learning
This demo is a TensorFlow implementation of the algorithm explained in the paper “Semi-blind-trace algorithm for self-supervised attenuation of trace-wise noise” and “Multi-blind-trace deep learning with a hybrid loss for attenuation of trace-wise noise” by Mohammad Mahdi Abedi, David Pardo, Tariq Alkhalifah.
The accompanying data includes a small part of the 2004 BP velocity estimation benchmark model (with tripled trace interval) contaminated with trace-wise noise. The data and the code are provided without warranties. Please read the Terms of Use of the original data here: https://wiki.seg.org/wiki/2004_BP_velocity_estimation_benchmark_model
Please unzip the provided dataset, and run the code.  
The semi_blind_trace.py code loads the data, defines the architecture, loss function, and custom training steps, trains the proposed model with the proposed algorithm, and plots the results. 
The “Results” folder includes a pretrained model, obtained using the provided code and data set.  

![image](https://github.com/mahdiabedi/semi-blind-trace-deep-learning/assets/134224333/bac672ee-6bd5-42b3-bc4e-734f6189c15f)


