from csv import reader, writer
from custom.spe import get_analysis_results
import numpy as np
from numpy.random import default_rng
from pathlib import Path
import pandas as pd

def classify_function_analysis(pred_L, pred_R):
    """Use classifer prediction to update analysis results.
    
    Args
        pred_L: List of string. Predictions on left-side clouds.
        pred_R: List of string. Predictions on right-side clouds.
    
    Returns: Numpy array.
    """
    
    results = np.array(get_analysis_results())

    num_left = np.array([get_classify_num(pred) for pred in pred_L])
    num_right = np.array([get_classify_num(pred) for pred in pred_R])

    classify_results = np.ones(results.shape)
    classify_results[:, 0:2] = results[:, 0:2]
    classify_results[:, 2] = np.multiply(num_left, results[:, 2])
    classify_results[:, 3] = np.multiply(num_right, results[:, 3])
    classify_results[:, 4] = np.sum(classify_results[:, 1:4], axis=1)
    classify_results[:, 5] = np.sum(classify_results[:, 2:4], axis=1)

    return classify_results

def create_directory_structure(path_main):
    """Creates the directory structure for the model data.
    
    Args
        path_main: Path object. Destination for model files.
    """

    if not path_main.exists():
        path_main.mkdir(parents=True)

def display_results(results, metrics):
    """Display sorted results for each metric.
    
    Args
        metrics: Dict. Contains strings as keys corresponding to
            metric function as values.
    """

    param_combs = results['params']
    param_keys = param_combs[0].keys()
    param_dict = {k: [comb[k] for comb in param_combs]
                  for k in param_keys}

    df_metric_result = {k: pd.DataFrame(
        {'Metric mean': results[f'mean_test_{k}'],
        'Std. dev.': results[f'std_test_{k}'],
        **param_dict})
        for k in metrics.keys()}

#    metrics_results = ([m, s, results['params']]
#                       for m, s in zip(means, stds))
#
#    results_bymetric = {
#        k:v for k, v in zip(metrics.keys(), metrics_results)}

    for k, v in df_metric_result.items():
        print(f'Results for {k} metric:')
        print()
        print(v.sort_values(by=['Metric mean'], ascending=False))
        print()
#    for k, v in results_bymetric.items():
#        sorted_results = sort_results(v)
#        print(f'Results for {k} metric:')
#        print()
#        for _, params, std, mean in sorted_results:
#            print(f'{mean:0.03f} +/- {std:0.03f} for {params!r}')
#        print()

def get_classify_num(pred):
    """Multiplicative factor to determined atom number.
    
    Args
        pred: String. Classifier prediction.
    
    Returns: Int. Atom number 0 for empty class. Otheriwse, atom
        number is unchanged.
    """

    if pred == 'empty':
        return 0
    else:
        return 1

def get_path_image(path_data, label, filename):
    """Get a Path object for an image file.
    
    Args
        pata_data: Path object. Parent directory for alls
            training images.
        label: String. Class label for image, specifies subfolder the
            image resides in..
        filename: String. Filename for image.
    
    Returns: Path object. Full path for image file."""

    return path_data.joinpath(f'label_{label}', filename)

def getstats_fromimage(path_data, label, filename):
    """Calculate stats of interest from the image.
    
    Args
        path_data: Path object. Parent directory for all
            training images.
        label: String. Class label for image, specifies subfolder the
            image resides in.
        filename: String. Filename for image.
    
    Returns: 4 element tuple.
    """
    path_image = get_path_image(path_data, label, filename)
    image = np.fromfile(path_image, np.float64)

    max_ = np.amax(image)
    min_ = np.amin(image)
    mean = np.mean(image)
    std = np.std(image)

    return max_, min_, mean, std

def getstats_fromstream(path_model_id, path_data):
    """Find parameters for scaling data by streaming images.
    
    Needed when dataset is too large to fit in memory.
    
    Args
        path_model_id: Path object. Directory of CSV file containing
            list of images in the training set.
        path_data: Path object. Parent directory for all
           training images.
    
    Returns: Two element tuple. Scaling parameters for data.
    """

    path_dataset_file = path_model_id.joinpath('training_set.csv')

    with path_dataset_file.open(mode='r', newline='') as f:
        csv_reader = reader(f, delimiter=',')
        rows = list(csv_reader)

    num_pixels = np.fromfile(get_path_image(path_data, rows[0][1], rows[0][0]),
                             np.float64).shape[0]

    stats_byimage = np.array([getstats_fromimage(path_data, row[1], row[0])
                     for row in rows])

    max_ = np.amax(stats_byimage[:,0])
    min_ = np.amin(stats_byimage[:, 1])

    return num_pixels, max_, min_

def load_datasets(path_sets, path_images):
    """Load images and labels for all sets (training, test, etc.)
    
    Args
        path_sets: Path object. Directory to sets where images to use
            for each class are specified in a CSV file.
        path_images: Path object. Directory to all images for each class.
        
    Returns: List. First half of the entries are lists containing the 
        datasets. Second half of the entries are lists containing the
        corresponding labels.
    """
    dataset_files = tuple(path_set_file.name 
        for path_set_file in path_sets.glob('*.csv'))

    set_names = [dataset_file[: dataset_file.find('_')]
                 for dataset_file in dataset_files]
    
    if len(dataset_files) == 3:
        name_order = ['training', 'validation', 'test']
        set_order = tuple(dataset_files.index(f'{name}_set.csv')
                          for name in name_order)
        num_sets = 3
    else:
        training_index = dataset_files.index('training_set.csv')
        set_order = (training_index, 1 - training_index)
        num_sets = 2

    images_and_labels = [None] * num_sets * 2
    
    for k in range(num_sets):
        path_dataset_file = path_sets.joinpath(dataset_files[set_order[k]])

        with path_dataset_file.open(mode='r', newline='') as f:
            csv_reader = reader(f, delimiter=',')
            dataset = list(csv_reader)

        path_dataset_images = [path_images.joinpath(f'label_{row[1]}', row[0])
                               for row in dataset]

        images_and_labels[k] = np.array([np.fromfile(path_image, np.float64)
                                         for path_image
                                         in path_dataset_images])

        images_and_labels[k+num_sets] = [row[1] for row in dataset]

    return images_and_labels

def make_datasets(class_names, dataset_dict, path_source, path_dest, seed):
    """Prepares training, test, validation sets.
    
    Args
        class_names: E.g., dog, cat, fish.
        dataset_dict: Dictionary containing number of examples of each
            class.
        path_source: Path object. Location of all example images.
        path_dest: Path object. Destination for model files.
        seed: Set random seed for randomly choosing data.
    """
    
    create_directory_structure(path_dest)

    path_alldata = [path_source.joinpath(f'label_{class_}')
                    for class_ in class_names]

    path_imagefiles = [class_path.glob('*.bin')
                       for class_path in path_alldata]

    size = sum([v for k, v in dataset_dict.items()])
    rng = default_rng(seed)

    datasets_by_class = np.array([rng.choice([image_file.name
                                  for image_file in image_filelist],
                                  size=size, replace=False)
                                  for image_filelist in path_imagefiles])

    dataset_labels = np.array([np.full(size, class_)
                               for class_ in class_names])

    if not path_dest.exists():
        path_dest.mkdir(parents=True)

    start=0
    for set_name, num_ex in dataset_dict.items():
        stop = start + num_ex

        filename = f'{set_name}_set.csv'
        path_file = path_dest.joinpath(filename)
        
        images = datasets_by_class[:,start:stop].flatten()
        labels = dataset_labels[:,start:stop].flatten()
        rows = np.transpose(np.vstack((images, labels))).tolist()

        with path_file.open(mode='w', newline='') as f:
            csv_writer = writer(f)
            csv_writer.writerows(rows)

        start = num_ex

def sort_results(metric_results):
    """Sort metric performance results by the mean.
    
    Args
        metric_results: List of lists. First two lists are mean and
            standard deviation of metric (accuracy, ...) from cross
            validation while tuning model parameters. Last list is
            a dictionary of the corresponding values of the model
            hyper-parameters being tuned.
    
    Returns: A numpy array of shape (n_param_combinations, 3)
        where n_param_combinations is the number of hyper-parameter
        combinations tested while tuning the model."""

    means, stds, params_list = metric_results
    dtype = [('index', int), ('params_list', object), ('std', float), ('mean', float)]

    #Sort will fail when attempting to rank based on the
    #dictionary 'params_list' when encountering identical mean and
    #standard deviations. To avoid this, use a list of distinct
    #integers to break the tie.
    values = zip(range(len(means)), params_list, stds, means)

    a = np.sort(np.array(list(values), dtype=dtype),
                kind='mergesort', order=['mean', 'std', 'index'])
    return np.flip(a, axis=-1)

