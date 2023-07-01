import joblib
import yaml
import matplotlib.pyplot as plt
from pedestrian_dataset_loader import PedestrianDatasetLoader
from sklearn.neural_network import MLPClassifier

DATASET_PATH = 'dataset'
DATASET_TEST_SIZE = 0.2
MODEL_PATH = 'pedestrian_classifier.joblib'
CFG_FILE_PATH = 'model_cfg.yaml'
CONFIG_FILE = 'classifier_cfg.yaml'

# Function to save the model configuration to a YAML file


def save_model_cfg(cfg_path, mean_pedestrian_height, mean_pedestrian_width):
    model_cfg = {'MEAN PEDESTRIAN HEIGHT': mean_pedestrian_height,
                 'MEAN PEDESTRIAN WIDTH': mean_pedestrian_width}

    with open(cfg_path, 'w') as file:
        yaml.dump(model_cfg, file)


# Create an instance of the PedestrianDatasetLoader
pedestrian_dataset_loader = PedestrianDatasetLoader(DATASET_PATH)
# Load and split the pedestrian dataset into training and testing sets
X_train, X_test, y_train, y_test = pedestrian_dataset_loader.load_pedestrian_dataset(
    DATASET_TEST_SIZE)
# Print information about the dataset
pedestrian_dataset_loader.print_dataset_info()

# Load the classifier configuration from the YAML file
with open(CONFIG_FILE, "r") as file:
    cfg_dict = yaml.safe_load(file)


print('Training classifier...')
# Create an MLP classifier with the specified configuration and fit it to the training data
classifier = MLPClassifier(random_state=1,
                           hidden_layer_sizes=(cfg_dict['HIDDEN LAYER1 SIZE'],
                                               cfg_dict['HIDDEN LAYER2 SIZE']),
                           solver='sgd',
                           verbose=True,
                           max_iter=cfg_dict['MAX ITER'],
                           batch_size=cfg_dict['BATCH SIZE'],
                           early_stopping=True,
                           validation_fraction=cfg_dict['VALIDATION FRACTION'],
                           learning_rate_init=cfg_dict['LEARNING_RATE_INIT']).fit(X_train, y_train)

print('Done training!!!')

loss = classifier.loss_curve_
validation = classifier.validation_scores_

# Evaluate the classifier on the testing data and calculate the classification score
classification_score = classifier.score(X_test, y_test)
print('Classification score: {}'.format(classification_score))

# Make predictions on the testing data
y_pred = classifier.predict(X_test)

# Plot the loss and validation scores
plt.figure(figsize=(8, 6))
plt.plot(range(len(loss)), loss, color='blue', label='Loss score')
plt.plot(range(len(validation)), validation,
         color='red', label='Validation score')
plt.xlabel('Iterations')
plt.ylabel('Loss/Accuracy')
plt.title('LOSS SCORE vs VALIDATION SCORE\n'
          f'Max iterations: {cfg_dict["MAX ITER"]}\n'
          f'Batch size: {cfg_dict["BATCH SIZE"]}\n'
          f'Learning rate: {cfg_dict["LEARNING_RATE_INIT"]}\n'
          f'Hidden layer sizes: ({cfg_dict["HIDDEN LAYER1 SIZE"]}, {cfg_dict["HIDDEN LAYER2 SIZE"]})\n'
          f'Classification score: {classification_score}')

plt.legend()
plt.show()

plt.savefig('first_plot.png')

# Save the trained classifier and model configuration
joblib.dump(classifier, MODEL_PATH)
save_model_cfg(CFG_FILE_PATH, pedestrian_dataset_loader.get_mean_pedestrian_height(),
               pedestrian_dataset_loader.get_mean_pedestrian_width())

print('Model and model cfg saved')
