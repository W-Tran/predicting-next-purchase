import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, classification_report
from feature_engineering import _get_train_test_data
import scikitplot as skplt


def train_models(ModelClass, invoices, observation_end_dates, **kwargs):
    train_results = dict(
        models=[],
        observation_end_dates=observation_end_dates,
        X_train=[],
        y_train=[],
        X_test=[],
        y_test=[],
    )

    for observation_end_date in observation_end_dates:
        X_train, y_train, X_test, y_test = _get_train_test_data(invoices, observation_end_date)
        model = ModelClass(**kwargs)
        model.fit(X_train, y_train)
        train_results['models'].append(model)
        train_results['X_train'].append(X_train)
        train_results['y_train'].append(y_train)
        train_results['X_test'].append(X_test)
        train_results['y_test'].append(y_test)

    return train_results


def evaluate_models(train_results, naive_labels, metric='accuracy', average_all_months=False):
    evaluation_method = dict(accuracy=accuracy_score, precision=precision_score)
    train_evals, test_evals = [], []

    for model, X_train, y_train, X_test, y_test, observation_end_date, y_naive_train, y_naive_test in zip(
            train_results['models'],
            train_results['X_train'],
            train_results['y_train'],
            train_results['X_test'],
            train_results['y_test'],
            train_results['observation_end_dates'],
            naive_labels['y_train'],
            naive_labels['y_test'],
    ):

        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        train_eval = round(evaluation_method[metric](y_train, y_pred_train), 3)
        test_eval = round(evaluation_method[metric](y_test, y_pred_test), 3)
        naive_train_eval = round(evaluation_method[metric](y_train, y_naive_train), 3)
        naive_test_eval = round(evaluation_method[metric](y_test, y_naive_test), 3)
        train_evals.append(train_eval)
        test_evals.append(test_eval)
        pred_num_pos = y_pred_test.sum()
        naive_num_pos = y_naive_test.sum()

        if not average_all_months:
            print(f"observation period end: {observation_end_date}\n"
                  f"MODEL - train {metric}: {train_eval}, test {metric}: {test_eval}, # +Predictions: {pred_num_pos}\n"
                  f"NAIVE - train {metric}: {naive_train_eval}, test {metric}: {naive_test_eval}, # +Predictions: {naive_num_pos}")
            print("---")

    if average_all_months:
        mean_train_eval, mean_test_eval = round(np.mean(train_evals), 3), round(np.mean(test_evals), 3)
        print(f"average train {metric}: {mean_train_eval}, average test {metric}: {mean_test_eval}")


def evaluate_models_by_plotting(train_results, naive_labels, metric='accuracy'):
    evaluation_method = dict(accuracy=accuracy_score, precision=precision_score)
    train_metrics, test_metrics, naive_test_metrics = [], [], []
    observation_end_dates = train_results['observation_end_dates']

    for model, X_test, y_test, observation_end_date, y_naive_test in zip(
            train_results['models'],
            train_results['X_test'],
            train_results['y_test'],
            train_results['observation_end_dates'],
            naive_labels['y_test'],
    ):
        y_pred_test = model.predict(X_test)
        test_eval = round(evaluation_method[metric](y_test, y_pred_test), 3)
        naive_test_eval = round(evaluation_method[metric](y_test, y_naive_test), 3)
        test_metrics.append(test_eval)
        naive_test_metrics.append(naive_test_eval)
        # baseline.append(y_test.value_counts().max()/len(y_test))

    fig = plt.figure(figsize=(9, 6))
    ax1 = fig.add_subplot(111)
    ax1.set_xlabel('Observation Period End')
    color = 'tab:red'
    ax1.set_ylabel('Test {}'.format(metric), color=color)
    ax1.plot(observation_end_dates, test_metrics, color=color, label='Test accuracy')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(alpha=0.3)
    ax1.set_ylim(0.4, 1.0)
    ax1.set_title('{} of predictions'.format(metric))

    color = 'tab:blue'
    ax2 = ax1.twinx()
    ax2.set_ylabel('Test {} of naive model'.format(metric), color=color)
    ax2.plot(observation_end_dates, naive_test_metrics, color=color, label='Majority label classifier')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylim(0.4, 1.0)


def single_month_performance(train_results, observation_end_date):
    results_index = train_results['observation_end_dates'].index(observation_end_date)
    y_test = train_results['y_test'][results_index]
    model = train_results['models'][results_index]
    X_test = train_results['X_test'][results_index]
    y_pred_test = model.predict(X_test)
    y_pred_probas = model.predict_proba(X_test)

    print(classification_report(y_test, y_pred_test))

    fig = plt.figure(figsize=(18, 5))
    ax = fig.add_subplot(131)
    skplt.metrics.plot_cumulative_gain(y_test, y_pred_probas, ax=ax)
    ax = fig.add_subplot(132)
    skplt.metrics.plot_lift_curve(y_test, y_pred_probas, ax=ax)
    ax = fig.add_subplot(133)
    skplt.metrics.plot_confusion_matrix(y_test, y_pred_test, ax=ax);
