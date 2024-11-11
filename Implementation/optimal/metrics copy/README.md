The metrics files store a tuple in this format: 

```
metrics = (conf_matrix, class_report, scores)
```

where scores is a dictionary in this format:

```
scores = {}
scores['accuracy'] = accuracy
scores['f1'] = f1
scores['precision'] = precision
scores['recall'] = recall
scores['auc_roc'] = auc_roc
```
