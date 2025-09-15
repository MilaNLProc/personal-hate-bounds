import pandas as pd
import numpy as np
from custom_logger import get_logger


def generate_aggregated_labels_dataset(
        dataset_name,
        dataset_path_train,
        dataset_path_test,
        subsample_majority_class=False
    ):
    """
    Loads the selected dataset (only Kumar et al. is implemented right now),
    with pre-made training and test splits. Label binarization (if used) is
    inherited from the loaded datasets.

    If `subsample_majority_class` is True, all classes are subsampled to a
    number of datapoints equal to the size of the minority class.
    Note: this may break other contraints on the data, e.g. that every
          annotator is present both in the training and in the test data.
    """
    logger = get_logger('generate_aggregated_labels_dataset')

    logger.info(
        f'Reading {dataset_name} training data from: {dataset_path_train}'
        f' | Reading {dataset_name} test data from: {dataset_path_test}'
    )
    
    if dataset_name.lower() == 'popquorn':
        # OUTDATED
        # data_df = pd.read_csv(dataset_path)

        # data_df = pd.merge(
        #     left=data_df[['instance_id', 'text']].drop_duplicates(subset='instance_id'),
        #     right=data_df.groupby('instance_id').apply(
        #         lambda group: group['offensiveness'].value_counts().sort_values(ascending=False).index[0]
        #     ).reset_index().rename(columns={0: 'offensiveness'}),
        #     on='instance_id',
        #     how='left'
        # )

        # data_df['offensiveness'] = data_df['offensiveness'].astype(int)

        # data_df['label'] = (data_df['offensiveness'] - 1).astype(int)
        raise NotImplementedError(
            'The original implementation for the popquorn dataset is outdated'
            ' and no new implementation exists'
        )
    elif dataset_name.lower() == 'kumar':
        training_data = pd.read_csv(dataset_path_train)[
            ['comment', 'text_id', 'toxic_score']
        ]
        test_data = pd.read_csv(dataset_path_test)[
            ['comment', 'text_id', 'toxic_score']
        ]

        # Aggregate by majority vote.
        training_data = training_data.groupby('text_id').agg(
            text=pd.NamedAgg('comment', 'first'),
            label=pd.NamedAgg(
                'toxic_score',
                lambda group: group.value_counts(ascending=False).index[0]
            )
        ).reset_index()

        test_data = test_data.groupby('text_id').agg(
            text=pd.NamedAgg('comment', 'first'),
            label=pd.NamedAgg(
                'toxic_score',
                lambda group: group.value_counts(ascending=False).index[0]
            )
        ).reset_index()

        if subsample_majority_class:
            logger.info('Subsampling the majority class')

            # Compute class counts.
            class_counts_train = (
                training_data['label']
                .value_counts()
                .sort_values(ascending=False)
            )
            class_counts_test = (
                test_data['label']
                .value_counts()
                .sort_values(ascending=False)
            )

            # For each label, randomly select a number of samples equal to the
            # minimal class counts.
            training_data = pd.concat([
                (
                    training_data[training_data['label'] == label]
                    .sample(frac=1)
                    .iloc[:class_counts_train.min()]
                )
                for label in class_counts_train.index
            ]).sample(frac=1).reset_index(drop=True)
            test_data = pd.concat([
                (
                    test_data[test_data['label'] == label]
                    .sample(frac=1)
                    .iloc[:class_counts_test.min()]
                )
                for label in class_counts_test.index
            ]).sample(frac=1).reset_index(drop=True)

    else:
        raise NotImplementedError(f'Dataset {dataset_name} not supported')

    return training_data, test_data


def subsample_dataset(
        training_data,
        test_data,
        optimal_n_training_datapoints,
        annotators_data_path
    ):
    """
    Subsamples the training and test dataset by
      1. Selecting the K top annotators by number of training datapoints so
         that the resulting number of datapoints is as close as possible to
         `optimal_n_training_datapoints`.
      2. Identifying the extreme annotators.
      3. Subsampling the training and test data to include all samples that
           * were annotated by an annotator in the selected top K
         OR
           * were annotated by an extreme annotator.
    """
    logger = get_logger('subsample_dataset')

    # Load annotators data.
    annotators_data = pd.read_csv(annotators_data_path)

    # Compute the number of annotations IN THE TRAINING DATA for each
    # annotator.
    annotations_per_annotator = (
        training_data
        .groupby('annotator_id')['text_id']
        .count()
        .sort_values(ascending=False)
        .reset_index()
        .rename(columns={'text_id': 'n_annotations'})
    )

    annotations_per_annotator['n_annotators_cumulative_sum'] = (
        range(1, len(annotations_per_annotator) + 1)
    )
    annotations_per_annotator['n_annotations_cumulative_sum'] = (
        annotations_per_annotator['n_annotations'].cumsum()
    )

    # Add annotators data.
    annotations_per_annotator = pd.merge(
        left=annotations_per_annotator,
        right=annotators_data[['annotator_id', 'extreme_annotator']],
        on='annotator_id',
        how='left'
    )

    # Compute the corresponding number of annotators to include.
    optimal_n_annotators = annotations_per_annotator.iloc[
        np.argmin(np.abs(
            annotations_per_annotator['n_annotations_cumulative_sum'].values
            - optimal_n_training_datapoints
        ))
    ]['n_annotators_cumulative_sum']

    logger.info(
        f'Optimal N datapoints: {optimal_n_training_datapoints}'
        f' | Optimal N annotators: {optimal_n_annotators}'
    )

    # Get the IDs of the selected annotators.
    selected_annotator_ids = annotations_per_annotator[
        annotations_per_annotator['n_annotators_cumulative_sum'] <= optimal_n_annotators
    ]['annotator_id'].tolist()

    # Subsample the training and test data to include the annotations
    # from the selected annotators AND those from the extreme annotators.
    logger.info('Subsampling the data (manually including all extreme annotators)')

    training_data_subsampled = training_data[
        (training_data['annotator_id'].isin(selected_annotator_ids))
        | (training_data['annotator_id'].isin(annotators_data[annotators_data['extreme_annotator']]['annotator_id']))
    ].reset_index(drop=True)

    test_data_subsampled = test_data[
        test_data['annotator_id'].isin(selected_annotator_ids)
        | (test_data['annotator_id'].isin(annotators_data[annotators_data['extreme_annotator']]['annotator_id']))
    ].reset_index(drop=True)

    logger.info(
        f'N training datapoints: {training_data_subsampled.shape[0]}'
        f' | N annotators: {test_data_subsampled["annotator_id"].unique().shape[0]}'
    )

    return training_data_subsampled, test_data_subsampled

