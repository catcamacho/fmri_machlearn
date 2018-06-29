
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

# In[1]:


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
subject_info = read_csv(sub_data_file, index_col=0)
subject_info.describe()


# In[2]:


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
conditions.describe()


# In[3]:


## Concatenate all the parameter estimates from preproc to create a feature set
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


# In[23]:


# determine which analysis to run
for analysis in ['all_conditions','allConds_predAge','negative','positive','neutral','age','age_neg','age_pos','age_neu']:

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
        mask = conditions['ageGroup']=='child'
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

    results_file = open(output_dir + '/results_' + analysis + '_final.txt','w')
    conditions[mask].describe()


    # ## Non-binary Classification
    # 
    # The below cells performs non-binary classification (support vector regression) based on age.

    # In[24]:


    if type_svm == 'nonbinary':
        # Perform the support vector classification
        from nilearn.input_data import NiftiMasker
        from sklearn.feature_selection import f_regression, SelectPercentile
        from sklearn.svm import SVR
        from sklearn.pipeline import Pipeline

        # Set up the regression
        svr = SVR(kernel='linear', C=1)
        masker = NiftiMasker(mask_img=gm_mask,standardize=True, 
                             memory='nilearn_cache', memory_level=1)

        feature_selection = SelectPercentile(f_regression, percentile=5)
        fs_svr = Pipeline([('feat_select', feature_selection), ('svr', svr)])

        # Run the regression
        X = masker.fit_transform(bold_feature_data)
        X = X[mask]
        maskedlabels=labels[mask]
        fs_svr.fit(X, maskedlabels)

        from sklearn.model_selection import cross_val_predict, LeaveOneGroupOut

        loso = LeaveOneGroupOut()
        y_pred = cross_val_predict(fs_svr, X, y=maskedlabels, n_jobs=6,
                                   groups=conditions['subject'][mask],cv=loso)
        # save weights
        coef = svr.coef_
        coef = feature_selection.inverse_transform(coef)
        coef_image = masker.inverse_transform(coef)
        coef_image.to_filename(output_dir + '/svrweights_' + analysis + '.nii.gz')

        ages_df = conditions[mask]
        ages_df['pred_age'] = Series(y_pred, index=ages_df.index)
        ages_df.head()

        actual_age = ages_df.groupby(['subject'])['age'].mean()
        actual_age = actual_age.tolist()
        pred_age = ages_df.groupby(['subject'])['pred_age'].mean()
        pred_age = pred_age.tolist()

        from scipy.stats import linregress
        slope, intercept, r_val, p_val, stderr = linregress(actual_age, pred_age) 

        from sklearn.metrics import mean_squared_error
        mse = mean_squared_error(actual_age, pred_age)

        from scipy.stats import spearmanr
        spear_r, spear_p = spearmanr(actual_age, pred_age)

        print("prediction accuracy: %.4f / p-value: %f / MSE: %f // Spearman: %f / p-value: %f" % (r_val, p_val, mse, spear_r, spear_p))

        # plot the predicted versus actual values
        import matplotlib.pyplot as plt
        plt.scatter(actual_age, pred_age, color='b')
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.savefig(output_dir + '/scatter_pred_actual_mean_' + analysis + '_final.svg', transparent=True)
        plt.show()
        plt.close()

        results_file.write("MEAN prediction accuracy r-value: %.4f / p-value: %f / MSE: %f // Spearman: %f / p-value: %f \n" % (r_val, p_val, mse, spear_r, spear_p))
        results_file.write('predicted: ' + str(pred_age) + '\n')
        results_file.write('actual: ' + str(actual_age) + '\n')


        actual_age = ages_df.groupby(['subject'])['age'].median()
        actual_age = actual_age.tolist()
        pred_age = ages_df.groupby(['subject'])['pred_age'].median()
        pred_age = pred_age.tolist()

        from scipy.stats import linregress
        slope, intercept, r_val, p_val, stderr = linregress(actual_age, pred_age) 

        from sklearn.metrics import mean_squared_error
        mse = mean_squared_error(actual_age, pred_age)

        from scipy.stats import spearmanr
        spear_r, spear_p = spearmanr(actual_age, pred_age)

        print("prediction accuracy: %.4f / p-value: %f / MSE: %f // Spearman: %f / p-value: %f" % (r_val, p_val, mse, spear_r, spear_p))

        # plot the predicted versus actual values
        import matplotlib.pyplot as plt
        plt.scatter(actual_age, pred_age, color='b')
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.savefig(output_dir + '/scatter_pred_actual_median_' + analysis + '_final.svg', transparent=True)
        plt.close()

        results_file.write("MEDIAN prediction accuracy r-value: %.4f / p-value: %f / MSE: %f // Spearman: %f / p-value: %f \n" % (r_val, p_val, mse, spear_r, spear_p))
        results_file.write('predicted: ' + str(pred_age) + '\n')
        results_file.write('actual: ' + str(actual_age) + '\n')

        results_file.close()


    # ## Perform binary support vector classification

    # In[16]:


    if type_svm == 'binary':
        # Perform the support vector classification
        from nilearn.input_data import NiftiMasker
        from sklearn.svm import SVC
        from sklearn.feature_selection import f_classif, SelectPercentile
        from sklearn.pipeline import Pipeline

        # Set up the support vector classifier
        svc = SVC(kernel='linear')
        masker = NiftiMasker(mask_img=gm_mask,standardize=True, 
                             memory='nilearn_cache', memory_level=1)

        # Select the features contributing to the model
        feature_selection = SelectPercentile(f_classif, percentile=5) #0.05/228453 voxels
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

        ## Save the SVM weights to a nifti
        coef = svc.coef_
        coef = feature_selection.inverse_transform(coef)
        weight_img = masker.inverse_transform(coef)
        weight_img.to_filename(output_dir + '/svmweights_'+ analysis +'.nii.gz')

        ## Calculate performance metrics
        from sklearn.metrics import recall_score, precision_score

        classification_accuracy = cv_scores['test_score'].mean()
        chance = 1. / len(labels.unique())
        print("Classification accuracy: %.4f / Chance level: %f" % 
              (classification_accuracy, chance))

        for label in labels.unique():
            sensitivity = recall_score(maskedlabels,y_pred,labels=[label],average='weighted')
            precision = precision_score(maskedlabels,y_pred,labels=[label],average='weighted')

            results_file.write("%s: classification accuracy: %.4f / chance level: %f / sensitivity: %f / precision: %f \n" % 
            (label, classification_accuracy, chance, sensitivity, precision))

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
        plt.savefig(output_dir + '/confusion_matrix_' + analysis + '.svg', transparent=True)

        results_file.close()


    # ## Perform permutation testing to get a p-value for the classifier

    # In[ ]:


    from sklearn.model_selection import permutation_test_score
    import matplotlib.pyplot as plt
    from numpy import savetxt

    results_file = open(output_dir + '/permut_results_' + analysis + '_final.txt','w')

    if type_svm == 'binary':
        # Perform permutation testing to get a p-value
        score, permutation_scores, pvalue = permutation_test_score(fs_svc, X, maskedlabels, scoring='accuracy', 
                                                                   cv=loso, n_permutations=500, n_jobs=10, 
                                                                   groups=conditions['subject'][mask])
        savetxt(output_dir + '/permutation_scores_' + analysis + '.txt', permutation_scores)

        print("Classification score %s (pvalue : %s)" % (score, pvalue))
        # Save a figure of the permutation scores
        plt.hist(permutation_scores, 20, label='Permutation scores',
                 edgecolor='black')
        ylim = plt.ylim()
        plt.plot(2 * [score], ylim, '--g', linewidth=3,
                 label='Classification Score (pvalue %f)' % pvalue)
        plt.plot(2 * [1. / len(labels.unique())], ylim, '--k', linewidth=3, label='Luck')
        plt.ylim(ylim)
        plt.legend()
        plt.xlabel('Score')
        plt.savefig(output_dir + '/permutation_plot_' + analysis + '.svg', transparent=True)
        plt.close()

        # save final pval/classifier score
        results_file.write("Classification score %s (pvalue : %s)" % (score, pvalue))
        results_file.close()

    elif type_svm == 'nonbinary':
        ## Perform permutation testing to get a p-value for MSE
        score, permutation_scores, pvalue = permutation_test_score(fs_svr, X, maskedlabels, scoring='neg_mean_squared_error', 
                                                                   cv=loso, n_permutations=500, n_jobs=10, 
                                                                   groups=conditions['subject'][mask])
        savetxt(output_dir + '/permutation_scores_mse_' + analysis + '.txt', permutation_scores)

        # Save a figure of the permutation scores
        plt.hist(permutation_scores, 20, label='Permutation scores',
                 edgecolor='black')
        ylim = plt.ylim()
        plt.plot(2 * [score], ylim, '--g', linewidth=3,
                 label='Mean Squared Error (pvalue %f)' % pvalue)
        plt.ylim(ylim)
        plt.legend()
        plt.xlabel('Score')
        plt.savefig(output_dir + '/permutation_plot_mse_' + analysis + '.svg', transparent=True)
        plt.close()

        # save final pval/classifier score
        results_file.write('MSE score %s (pvalue : %s) \n' % (score, pvalue))

        ## Perform permutation testing to get a p-value for r-squared
        score, permutation_scores, pvalue = permutation_test_score(fs_svr, X, maskedlabels, scoring='r2', 
                                                                   cv=loso, n_permutations=500, n_jobs=10, 
                                                                   groups=conditions['subject'][mask])
        savetxt(output_dir + '/permutation_scores_r2_' + analysis + '.txt', permutation_scores)

        # Save a figure of the permutation scores
        plt.hist(permutation_scores, 20, label='Permutation scores',
                 edgecolor='black')
        ylim = plt.ylim()
        plt.plot(2 * [score], ylim, '--g', linewidth=3,
                 label='R-squared (pvalue %f)' % pvalue)
        plt.ylim(ylim)
        plt.legend()
        plt.xlabel('Score')
        plt.savefig(output_dir + '/permutation_plot_r2_' + analysis + '.svg', transparent=True)
        plt.close()

        # save final pval/classifier score
        results_file.write('R square: %s (pvalue : %s) \n' % (score, pvalue))
        results_file.close()    

