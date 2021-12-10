from os.path import join

# +
training_dir = join("..","..","datasets","AT&T Database of Faces","faces","training")
testing_dir = join("..","..","datasets","AT&T Database of Faces","faces","testing")

siamese_training_csv = join("..","..","datasets","AT&T Database of Faces","faces","training","siamese_train_data.csv")
siamese_testing_csv = join("..","..","datasets","AT&T Database of Faces","faces","testing","siamese_test_data.csv")

train_batch_size = 40
train_number_epochs = 50
learning_rate = 0.0005
weight_decay = 0.0005
step_size=10
gamma=0.2
img_height = 100
img_width = 100

# Loss function parameter
margin = 1.0
# -

# testing_dir = join("..","..","datasets", "tutorial","test")
# testing_csv = "test_data.csv"
