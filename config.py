from os.path import join

# +
training_dir = join("..","..","datasets","AT&T Database of Faces","faces","training")
testing_dir = join("..","..","datasets","AT&T Database of Faces","faces","testing")

siamese_training_csv = join("..","..","datasets","AT&T Database of Faces","faces","training","siamese_train_data.csv")
siamese_testing_csv = join("..","..","datasets","AT&T Database of Faces","faces","testing","siamese_test_data.csv")

compare_siamese_csv = join("..","..","datasets","AT&T Database of Faces","faces","training","compare_siamese.csv")
compare_cnn_csv = join("..","..","datasets","AT&T Database of Faces","faces","training","compare_cnn.csv")
compare_test_csv = join("..","..","datasets","AT&T Database of Faces","faces","training","compare_test.csv")


train_batch_size = 128
train_number_epochs = 60
learning_rate = 0.0005
weight_decay = 0.0005
step_size=15
gamma=0.5
img_height = 100
img_width = 100

# Loss function parameter
margin = 0.25
