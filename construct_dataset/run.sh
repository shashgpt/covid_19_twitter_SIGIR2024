#!/bin/bash

asset_name="without_rule_label_consistency_check_fixed_error_in_counters"
file_no_start=481
for i in {0..55}
do  
    file_no=$(expr $file_no_start + $i)
    screen -S "corona_tweets"$file_no -d -m taskset --cpu-list $i bash config.sh $asset_name $file_no
done

# asset_name="without_rule_label_consistency_check"
# process_no=1
# start=1
# stop=1
# screen -S "screen"$process_no -d -m taskset --cpu-list $process_no bash config.sh $asset_name $start $stop
# # bash config.sh $asset_name $start $stop

# def build_model(hp):
#     model = keras.Sequential()
#     model.add(keras.layers.Dense(hp.Choice('units', [8, 16, 32]), activation='relu'))
#     model.add(keras.layers.Dense(1, activation='relu'))
#     model.compile(loss='mse')
#     return model

# tuner = keras_tuner.RandomSearch(build_model,
#                                 objective='val_loss',
#                                 max_trials=5)

# tuner.search(x_train, y_train, epochs=5, validation_data=(x_val, y_val))
# best_model = tuner.get_best_models()[0]


# def build_model(hp):
#     model = keras.Sequential()
#     model.add(layers.Flatten())
#     model.add(
#         layers.Dense(
#             # Define the hyperparameter.
#             units=hp.Int("units", min_value=32, max_value=512, step=32),
#             activation="relu",
#         )
#     )
#     model.add(layers.Dense(10, activation="softmax"))
#     model.compile(
#         optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"],
#     )
#     return model

def call_existing_code(units, activation, dropout, lr):
    model = keras.Sequential()
    model.add(layers.Flatten())
    model.add(layers.Dense(units=units, activation=activation))
    if dropout:
        model.add(layers.Dropout(rate=0.25))
    model.add(layers.Dense(10, activation="softmax"))
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_model(hp):
    units = hp.Int("units", min_value=32, max_value=512, step=32)
    activation = hp.Choice("activation", ["relu", "tanh"])
    dropout = hp.Boolean("dropout")
    lr = hp.Float("lr", min_value=1e-4, max_value=1e-2, sampling="log")
    # call existing model-building code with the hyperparameter values.
    model = call_existing_code(
        units=units, activation=activation, dropout=dropout, lr=lr
    )
    return model


build_model(keras_tuner.HyperParameters())

tuner = keras_tuner.RandomSearch(
    hypermodel=build_model,
    objective="val_accuracy",
    max_trials=3,
    executions_per_trial=2,
    overwrite=True,
    directory="my_dir",
    project_name="helloworld",
)

tuner.results_summary()

# Get the top 2 hyperparameters.
best_hps = tuner.get_best_hyperparameters(5)
# Build the model with the best hp.
model = build_model(best_hps[0])
# Fit with the entire dataset.
x_all = np.concatenate((x_train, x_val))
y_all = np.concatenate((y_train, y_val))
model.fit(x=x_all, y=y_all, epochs=1)