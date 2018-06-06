
# coding: utf-8

# # Build the Classifier
# This notebook is dedicated to creating classifiers and run classification analyses of interest on neuroimaging data.
# 
# Can we accurately classify:
# - adults vs. children across conditions
# - adults versus children within each condition
# 
# Can we predict:
# - child age based on multivariate patterns of brain activation
# - child age based on brain activation in each condition



def run_svm(analysis):
    from pandas import DataFrame, Series, read_csv

    # Study specific variables
    study_home = '/home/camachocm2/Analysis/KidVid_MVPA'
    standard_mask = study_home + '/template/MNI152_T1_2mm_brain_mask_KV.nii.gz'
    gm_mask = study_home + '/template/gm_2mm_mask.nii.gz'
    template = study_home + '/template/MNI152_T1_1mm_brain.nii.gz'
    sub_data_file = study_home + '/doc/subjectinfo.csv'
    preproc_dir = study_home + '/analysis/preproc/betas'
    output_dir = study_home + '/analysis/classifier'

    condition_data = read_csv(study_home + '/doc/conditionslist.csv')
    subject_info = read_csv(sub_data_file)

    ## Create a conditions list for the feature set
    condition_labels = condition_data['labels'].tolist()
    subjects_list = subject_info['subjID'].tolist()
    age_group_list = subject_info['group'].tolist()
    ages_mos_list = subject_info['age_mos'].tolist()

    conditions = condition_data
    conditions['subject'] = Series(subjects_list[0], index=conditions.index)
    conditions['ageGroup'] = Series(age_group_list[0], index=conditions.index)
    conditions['age'] = Series(ages_mos_list[0], index=conditions.index)

    for a in range(1,len(subjects_list)):
        temp=DataFrame()
        temp['labels'] = Series(condition_labels)
        temp['subject'] = Series(subjects_list[a], index=temp.index)
        temp['ageGroup'] = Series(age_group_list[a], index=temp.index)
        temp['age'] = Series(ages_mos_list[a], index=temp.index)


        conditions = conditions.append(temp, ignore_index=True)

    #conditions.to_csv(output_dir + '/featureset_key.csv')

    ## Temporally concatenate all the parameter estimates from preproc to create a feature set
    from glob import glob
    from nipype.interfaces.fsl.utils import Merge
    files = glob(preproc_dir + '/*/betas.nii.gz')
    files = sorted(files)

    bold_feature_data = output_dir + '/featureset.nii.gz'

    merge = Merge()
    merge.inputs.in_files = files
    merge.inputs.dimension = 't'
    merge.inputs.merged_file = bold_feature_data
    #merge.run()
    
    if analysis == 'all_conditions':
        mask = conditions['labels'].isin(['negative','positive','neutral'])
        labels = conditions['labels']
        type_svm = 'binary'
    elif analysis == 'allConds_predAge':
        mask = conditions['labels'].isin(['negative','positive','neutral'])
        labels = conditions['ageGroup']
        type_svm = 'binary'
    elif analysis == 'negative':
        mask = conditions['labels'].isin(['negative'])
        labels = conditions['ageGroup']
        type_svm = 'binary'
    elif analysis == 'positive':
        mask = conditions['labels'].isin(['positive'])
        labels = conditions['ageGroup']
        type_svm = 'binary'
    elif analysis == 'neutral':
        mask = conditions['labels'].isin(['neutral'])
        labels = conditions['ageGroup']
        type_svm = 'binary'
    elif analysis=='age':
        mask = (conditions['ageGroup']=='child')
        labels = conditions['age']
        type_svm = 'nonbinary'
    elif analysis == 'age_neg':
        mask = (conditions['ageGroup']=='child') & (conditions['labels']=='negative')
        labels = conditions['age']
        type_svm = 'nonbinary'
    elif analysis == 'age_pos':
        mask = (conditions['ageGroup']=='child') & (conditions['labels']=='positive')
        labels = conditions['age']
        type_svm = 'nonbinary'
    elif analysis == 'age_neu':
        mask = (conditions['ageGroup']=='child') & (conditions['labels']=='neutral')
        labels = conditions['age']
        type_svm = 'nonbinary'

    conditions[mask].describe()
    results_file = open(output_dir + '/results_' + analysis + '.txt','w')


    # ## Non-binary Classification
    # 
    # The below cells performs non-binary classifiacation based on age.

    # In[ ]:


    if type_svm == 'nonbinary':
        # Perform the support vector classification
        from nilearn.input_data import NiftiMasker
        from sklearn.feature_selection import f_regression, SelectPercentile, SelectFdr
        from sklearn.svm import SVR
        from sklearn.pipeline import Pipeline

        # Set up the regression
        svr = SVR(kernel='linear', C=1)
        masker = NiftiMasker(mask_img=gm_mask,standardize=True, 
                             memory='nilearn_cache', memory_level=1)

        feature_selection = SelectFdr(f_regression, alpha=0.00000021) #0.05/228453 voxels = 0.00000021  #0.01 = 0.00000004
        fs_svr = Pipeline([('feat_select', feature_selection), ('svr', svr)])

        # Run the regression
        X = masker.fit_transform(bold_feature_data)
        X = X[mask]
        maskedlabels=labels[mask]
        fs_svr.fit(X, maskedlabels)

        from sklearn.model_selection import cross_val_predict, LeaveOneGroupOut

        loso = LeaveOneGroupOut()
        y_pred = cross_val_predict(fs_svr, X, y=maskedlabels, 
                                   groups=conditions['subject'][mask],cv=loso)

        from scipy.stats import linregress
        slope, intercept, r_val, p_val, stderr = linregress(maskedlabels, y_pred) 

        print("prediction accuracy: %.4f / p-value: %f" % (r_val, p_val))

        results_file.write("prediction accuracy r-value: %.4f / p-value: %f \n" % (r_val, p_val))
        results_file.write('predicted: ' + str(y_pred) + '\n')
        results_file.write('actual: ' + str(maskedlabels) + '\n')

        # save SVR weights
        coef = svr.coef_
        coef = feature_selection.inverse_transform(coef)
        coef_image = masker.inverse_transform(coef)
        coef_image.to_filename(output_dir + '/svrweights_' + analysis + '.nii.gz')

        import matplotlib.pyplot as plt

        plt.scatter(maskedlabels, y_pred, color='b')
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.savefig(output_dir + '/scatter_pred_actual_' + analysis + '.png', transparent=True)
        plt.close()

    elif type_svm == 'binary':
        # Perform the support vector classification
        from nilearn.input_data import NiftiMasker
        from sklearn.svm import SVC
        from sklearn.feature_selection import f_classif, SelectPercentile, SelectFdr
        from sklearn.pipeline import Pipeline

        # Set up the support vector classifier
        svc = SVC(kernel='linear')
        masker = NiftiMasker(mask_img=gm_mask,standardize=True, 
                             memory='nilearn_cache', memory_level=1)

        # Select the top 5 percent features contributing to the model
        feature_selection = SelectFdr(f_classif, alpha=0.00000021) #0.05/228453 voxels = 0.00000021 #0.01 = 0.00000004
        fs_svc = Pipeline([('feat_select', feature_selection), ('svc', svc)])

        # Run the classifier
        X = masker.fit_transform(bold_feature_data)
        X = X[mask]
        maskedlabels=labels[mask]
        fs_svc.fit(X, maskedlabels)
        # Obtain prediction values via cross validation
        from sklearn.model_selection import cross_validate, LeaveOneGroupOut, cross_val_predict

        loso = LeaveOneGroupOut()
        cv_scores = cross_validate(fs_svc, X, y=maskedlabels, n_jobs=10, return_train_score=True,
                                   groups=conditions['subject'][mask], cv=loso, scoring='accuracy')
        y_pred = cross_val_predict(fs_svc, X, y=maskedlabels, n_jobs=10,
                                   groups=conditions['subject'][mask], cv=loso)

        classification_accuracy = cv_scores['test_score'].mean()

        print("Classification accuracy: %.4f / Chance level: %f" % 
              (classification_accuracy, 1. / len(labels.unique())))

        results_file.write("Classification accuracy: %.4f / Chance level: %f \n" % 
        (classification_accuracy, 1. / len(labels.unique())))

        ## Save the SVM weights to a nifti
        coef = svc.coef_
        coef = feature_selection.inverse_transform(coef)
        weight_img = masker.inverse_transform(coef)
        weight_img.to_filename(output_dir + '/svmweights_'+ analysis +'.nii.gz')

        # compute and display a confusion matrix
        from sklearn.metrics import confusion_matrix
        from numpy import set_printoptions
        import itertools
        import matplotlib.pyplot as plt

        cnf_matrix = confusion_matrix(maskedlabels, y_pred)
        set_printoptions(precision=2)
        classes = maskedlabels.unique()

        def plot_confusion_matrix(cm, classes):
            from numpy import arange
            plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
            plt.title('Confusion matrix')
            plt.colorbar()
            tick_marks = arange(len(classes))
            plt.xticks(tick_marks, classes, rotation=45)
            plt.yticks(tick_marks, classes)

            thresh = cm.max() / 2.
            for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
                plt.text(j, i, format(cm[i, j],  'd'),
                         horizontalalignment='center',
                         color='white' if cm[i, j] > thresh else 'black')

            plt.tight_layout()
            plt.ylabel('True label')
            plt.xlabel('Predicted label')

        plot_confusion_matrix(cnf_matrix, classes)
        plt.savefig(output_dir + '/confusion_matrix_' + analysis + '.png', transparent=True)
        plt.close()

    results_file.close()    


for analysis in ['all_conditions','allConds_predAge','negative',
                 'neutral','positive','age','age_neg','age_pos','age_neu']:
    run_svm(analysis)