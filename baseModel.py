from keras.applications.vgg19 import VGG19

model = VGG19()

model.summary()

model.save('vgg19.h5')