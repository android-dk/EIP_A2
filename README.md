# EIP_A2
Train on 60000 samples, validate on 10000 samples
Epoch 1/20

Epoch 00001: LearningRateScheduler setting learning rate to 0.003.
60000/60000 [==============================] - 13s 214us/step - loss: 0.0240 - acc: 0.9921 - val_loss: 0.0313 - val_acc: 0.9902
Epoch 2/20

Epoch 00002: LearningRateScheduler setting learning rate to 0.0022744503.
60000/60000 [==============================] - 10s 175us/step - loss: 0.0231 - acc: 0.9925 - val_loss: 0.0254 - val_acc: 0.9923
Epoch 3/20

Epoch 00003: LearningRateScheduler setting learning rate to 0.0018315018.
60000/60000 [==============================] - 10s 173us/step - loss: 0.0191 - acc: 0.9938 - val_loss: 0.0255 - val_acc: 0.9923
Epoch 4/20

Epoch 00004: LearningRateScheduler setting learning rate to 0.0015329586.
60000/60000 [==============================] - 10s 172us/step - loss: 0.0182 - acc: 0.9943 - val_loss: 0.0194 - val_acc: 0.9938
Epoch 5/20

Epoch 00005: LearningRateScheduler setting learning rate to 0.0013181019.
60000/60000 [==============================] - 10s 171us/step - loss: 0.0171 - acc: 0.9942 - val_loss: 0.0224 - val_acc: 0.9929
Epoch 6/20

Epoch 00006: LearningRateScheduler setting learning rate to 0.0011560694.
60000/60000 [==============================] - 10s 173us/step - loss: 0.0159 - acc: 0.9945 - val_loss: 0.0192 - val_acc: 0.9942
Epoch 7/20

Epoch 00007: LearningRateScheduler setting learning rate to 0.0010295127.
60000/60000 [==============================] - 10s 171us/step - loss: 0.0150 - acc: 0.9954 - val_loss: 0.0195 - val_acc: 0.9934
Epoch 8/20

Epoch 00008: LearningRateScheduler setting learning rate to 0.0009279307.
60000/60000 [==============================] - 10s 173us/step - loss: 0.0158 - acc: 0.9946 - val_loss: 0.0230 - val_acc: 0.9928
Epoch 9/20

Epoch 00009: LearningRateScheduler setting learning rate to 0.0008445946.
60000/60000 [==============================] - 10s 172us/step - loss: 0.0154 - acc: 0.9948 - val_loss: 0.0211 - val_acc: 0.9931
Epoch 10/20

Epoch 00010: LearningRateScheduler setting learning rate to 0.0007749935.
60000/60000 [==============================] - 10s 172us/step - loss: 0.0152 - acc: 0.9948 - val_loss: 0.0206 - val_acc: 0.9939
Epoch 11/20

Epoch 00011: LearningRateScheduler setting learning rate to 0.0007159905.
60000/60000 [==============================] - 10s 170us/step - loss: 0.0141 - acc: 0.9953 - val_loss: 0.0213 - val_acc: 0.9934
Epoch 12/20

Epoch 00012: LearningRateScheduler setting learning rate to 0.000665336.
60000/60000 [==============================] - 10s 170us/step - loss: 0.0140 - acc: 0.9952 - val_loss: 0.0204 - val_acc: 0.9935
Epoch 13/20

Epoch 00013: LearningRateScheduler setting learning rate to 0.0006213753.
60000/60000 [==============================] - 10s 172us/step - loss: 0.0137 - acc: 0.9957 - val_loss: 0.0192 - val_acc: 0.9940
Epoch 14/20

Epoch 00014: LearningRateScheduler setting learning rate to 0.0005828638.
60000/60000 [==============================] - 10s 170us/step - loss: 0.0128 - acc: 0.9957 - val_loss: 0.0201 - val_acc: 0.9936
Epoch 15/20

Epoch 00015: LearningRateScheduler setting learning rate to 0.0005488474.
60000/60000 [==============================] - 10s 170us/step - loss: 0.0120 - acc: 0.9958 - val_loss: 0.0197 - val_acc: 0.9943
Epoch 16/20

Epoch 00016: LearningRateScheduler setting learning rate to 0.0005185825.
60000/60000 [==============================] - 10s 169us/step - loss: 0.0119 - acc: 0.9963 - val_loss: 0.0212 - val_acc: 0.9935
Epoch 17/20

Epoch 00017: LearningRateScheduler setting learning rate to 0.000491481.
60000/60000 [==============================] - 10s 169us/step - loss: 0.0125 - acc: 0.9957 - val_loss: 0.0215 - val_acc: 0.9929
Epoch 18/20

Epoch 00018: LearningRateScheduler setting learning rate to 0.0004670715.
60000/60000 [==============================] - 10s 166us/step - loss: 0.0128 - acc: 0.9955 - val_loss: 0.0205 - val_acc: 0.9932
Epoch 19/20

Epoch 00019: LearningRateScheduler setting learning rate to 0.0004449718.
60000/60000 [==============================] - 10s 169us/step - loss: 0.0119 - acc: 0.9960 - val_loss: 0.0208 - val_acc: 0.9936
Epoch 20/20

Epoch 00020: LearningRateScheduler setting learning rate to 0.000424869.
60000/60000 [==============================] - 10s 172us/step - loss: 0.0117 - acc: 0.9957 - val_loss: 0.0202 - val_acc: 0.9940

Output of model.evaluate(X_test, Y_test) [0.020208156953587603, 0.994]

Strategy Overview :
I have used 32, 64, 128 kernels in my first DNN (98k params). For this assignment, I have used 4, 8, 16 kernels (doubling the kernels on each convolution step) in my first Convolution Block. I've then used MaxPooling(2,2) in the Transition Block. In the second Convolution Block I've used 8, 16, 32 kernels (doubling on each step). Finally I've used Global Average Pooling which helped me achieve an accuracy of 99.4% using 9138 params.

Architecture Strategy :
Since the image size was 28 x 28, I thought it was reasonable to use no more than 32 (3 x 3) kernels since 512 kernels were used for a 400 x 400 image.

Fewer number of Kernels : To employ fewer parameters without sacrificing accuracy I used (4, 8, 16) number of kernels in the first Convolution block and (8, 16, 32) number of kernels in the second Convolution block.
Note : Reversing the order of these Convolution blocks resulted in lower accuracies. Trying to find out why.

Use of Batch Normalization : Batch Normalization was used at the end of each Convolution layer to ensure higher learning rates at each level.

Use of Global Average Pooling (GAP) : GAP layer ensures that spatial information is retained. It also helps reduce the number of parameters considerably. Further, GAP is more native to Convolution structure than the traditional fully connected layer (which I have used in my last model).

Effect of Dropout : Employing Dropout at each Convolution layer ensure there is no overfitting since Dropout is a regulariser.

