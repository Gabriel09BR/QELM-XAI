import re
from pathlib import Path
from time import time

import pandas as pd

from IPython.display import clear_output

from sklearn.metrics import f1_score, accuracy_score, confusion_matrix

from pycaret.classification import setup, create_model, predict_model, pull, compare_models



from pathlib import Path

output_file = (
    Path(__file__).resolve().parents[2]
    / "data" / "results" / "ALLmodelsPickandPlace.xlsx"
)

output_file.parent.mkdir(parents=True, exist_ok=True)


# Comparing all models: ELMs, QELMs, 'rf', 'lightgbm', 'gbc', 'et'

classifiers = [
    model1, model2, model3, model4, model5, model6,
    model7, model8, model9, model10, model11, model12,
    model13, model14, model15, model16, model17, model18,
    'rf', 'lightgbm', 'gbc', 'et'
]


def base_model_name(cl):
    return re.split(r'\s*\(', str(cl), 1)[0]


min_instances = 20  # 20–70
max_instances = 25


results = pd.DataFrame(columns=[
    'random_state', 'classifier', 'n',
    'unseen_all_cm', 'unseen_all_f1', 'unseen_all_acc',
    'test_all_cm', 'test_all_f1', 'test_all_acc',
    'train_time_sec', 'test_time'
])

row = 0
for rs in range(0, 29):  # different random_states → experiments run 30 times (0–29)

    clear_output(wait=True)
    print("Running rs =", rs)

    for cl in classifiers:  # different classifiers → all models listed in classifiers
        for n in range(min_instances, max_instances, 5):  # different training set sizes


            # df1: dataset simulating the data already available (past)
            # df2: dataset simulating unseen data (future)

            # limiting the dataset to n fault instances, since faults are rare events
            df1 = dataset[
                ((dataset['item'] <= n) & (dataset['class'] == 0))
                |
                ((dataset['item'] <= n) & (
                    (dataset['class'] == 10)
                    |
                    (dataset['class'] == 11)
                    |
                    (dataset['class'] == 12)
                    |
                    (dataset['class'] == 20)
                    |
                    (dataset['class'] == 21)
                    |
                    (dataset['class'] == 22)
                ))
            ].copy()


            classes = df1['class'].unique()  # Define classes here

            for class_number in classes:
                print(str(class_number), ':', len(df1[df1['class'] == class_number]))


            df2 = dataset.drop(df1.index)  # unseen / reserved data (not used for training or testing)

            for class_number in classes:
                print(str(class_number), ':', len(df2[df2['class'] == class_number]))


            setup(
                df1,
                target='class',
                train_size=0.7,
                session_id=rs,
                normalize=True,
                # silent=True,
                # numeric_features=num_features,
                remove_outliers=False,
                feature_selection=False,
                imputation_type='simple',
                # fix_imbalance=True
            )

            # Measure training time
            classifier = create_model(cl)

            all_metrics = pull()  # extracting classifier metrics

            compare_models([cl])
            train_time_sec = pull()['TT (Sec)'][0]


            # Metrics on unseen / reserved / future-like data
            start = time()
            unseen_predictions = predict_model(classifier, data=df2, raw_score=True)
            end = time()
            test_time = end - start

            y_true_unseen = unseen_predictions['class'].astype(int)
            y_pred_unseen = unseen_predictions['prediction_label'].astype(int)

            unseen_all_f1  = f1_score(y_true_unseen, y_pred_unseen, average='weighted')
            unseen_all_acc = accuracy_score(y_true_unseen, y_pred_unseen)
            unseen_all_cm  = confusion_matrix(y_true_unseen, y_pred_unseen)


            # Metrics on available test data
            test_predictions = predict_model(classifier, raw_score=True)

            y_true = test_predictions['class'].astype(int)
            y_pred = test_predictions['prediction_label']

            test_all_f1  = all_metrics['F1'][0]
            test_all_acc = all_metrics['Accuracy'][0]
            test_all_cm  = confusion_matrix(y_true, y_pred)


            # Saving results
            results.loc[row] = [
                rs, base_model_name(cl), n,
                unseen_all_cm, unseen_all_f1, unseen_all_acc,
                test_all_cm, test_all_f1, test_all_acc,
                train_time_sec, test_time
            ]

            row += 1
            print('rs:', rs, ' cl:', cl, ' row:', row)


    results.to_excel(output_file, index=False)
        

df = pd.read_excel(output_file)
df.insert(0, "Column 0", range(len(df)))
df.to_excel(output_file, index=False)



