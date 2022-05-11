# Final-Prototype-Analysis

Data analysis and engineering for final prototype, built using python/pytorch

# RUN command

Run the random forest classifier with following command:

```bash
python train_rf.py --question-file question.json --user-data '{"living_situation": 3}'
```

where the `question-file` argument expects the path to the json containing the
question definition for each user feature and `user-data` a json encoded string containing all the collected user information up until now. the keys for the user data are the same as in the question file, the values are the indices of the answers selected by the user. So in the example, the user answered the question after his or her living situation with the answer with index 3 which would be "Anderer Beziehungsstatus".

Call `python train_rf.py --help` for a list of arguments.
