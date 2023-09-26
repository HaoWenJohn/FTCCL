import draw_comfusion_matrix
from . import _eval_protocols as eval_protocols

from sklearn.metrics import classification_report,accuracy_score,cohen_kappa_score,confusion_matrix
from sklearn.model_selection import train_test_split


def eval_classification(model, test_data, test_labels,fine_tune_size,draw=True):
    test_repr = model.encode(test_data, encoding_window='full_series')
    fit_clf = eval_protocols.fit_svm
    finetune_train_X, test_X, finetune_train_labels, test_labels = train_test_split(test_repr, test_labels,
                                                                                     test_size=1-fine_tune_size, random_state=42)

    clf = fit_clf(finetune_train_X, finetune_train_labels)
    predict_labels = clf.predict(test_X)

    return classification_report(test_labels, predict_labels, digits=4, output_dict=True),\
        accuracy_score(test_labels,predict_labels),\
        cohen_kappa_score(test_labels,predict_labels), \
        confusion_matrix(test_labels,predict_labels,labels=[x for x in range(10)])

