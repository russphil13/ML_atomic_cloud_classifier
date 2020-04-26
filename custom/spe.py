from csv import reader
import numpy as np
from os.path import sep
from custom.pgsql import close_connection, insert_image_data, insert_exp_params
from custom.pgsql import insert_analysis_data, find_image_id
from custom.pgsql import get_table
from custom.pgsql import get_image_ids, get_row_by_id, open_connection

def analyze_and_write(image_files, params_file):
    """Write analysis data to the database.
    
    Args
        images_files: List of Path objects. Full-path filenames
            of images to analyze.
        params_file: Path object. CSV file containing parameters
            needed for image analysis.

    Returns: Numpy array.
    """
    conn = open_connection()

    image_ids = get_image_ids(
        conn, 'experiment_parameters', schema='atomic_cloud_images')

    results = [analyze_by_id(conn, img_id, params_file)
               for img_id in image_ids]

    values = [(img_id,) + res
              for img_id, res in zip(image_ids, results)]

    insert_analysis_data(conn, values, schema='atomic_cloud_images')

    close_connection(conn)

def analyze_by_id(conn, image_id, params_file, frame_num=1):
    """Analyze the image with the given id.
    
    Args
        conn: Database connection.
        image_id: Int. Image id (primary key column).
        params_file: Path object. CSV file containing parameters
            needed for analysis.
        frame_num: Int. Frame number to analyze.
        
    Returns: A tuple.
    """
    frame = get_image_by_id(conn, image_id, frame_num=frame_num)

    results = do_analysis(frame, params_file)

    return results

def convert_frames(frames_bytes, datatype):
    """Convert frames binary data to a numpy array.
    
    Args
        frames_bytes: Bytes or MemoryView object. Frames binary data.
        frame_count: Int. Number of frames.
        n_rows: Int. Number of rows.
        n_cols: Int. Number of columns.
        datatype: Int. Represents the data type of the pixel values.
            
    Returns: Numpy array.
    """
    
    dtype_dict = {1: np.int32,
                  2: np.int16,
                  3: np.uint16,
                  5: np.float64}

    frames = np.frombuffer(frames_bytes, dtype=dtype_dict[datatype])

    return frames

def do_background_subtraction(image, border):
    """Subtract background from image.

    The background is determined by averaging the pixels values in a
    perimeter of 'border' pixels around the region of interest.

    Args
        image: Numpy array of the pixel values.
        border: Int. Width of background border in pixels.

    Returns: Numpy array. Image with the background subtracted out.
    """

    atoms_region = get_region(image, border-1, -border, border-1, -border)

    image_region_size = image.shape
    atoms_region_size = atoms_region.shape

    atoms_region_area = atoms_region_size[0] * atoms_region_size[1]
    border_region_area = (image_region_size[0] * image_region_size[1]
                          - atoms_region_area)

    image_sum = np.sum(image)
    atoms_sum = np.sum(atoms_region)
    border_sum = image_sum - atoms_sum
    
    bkgnd = border_sum / border_region_area * np.ones(image_region_size)

    return image - bkgnd

def do_analysis(image_frame, params_file):
    """Find atom number for each cloud in all images.

    Args
        image_files: Path object. Full-path image file.
        params_file: Path object. CSV file containing ROI.
    
    Returns: Tuple.
    """

    num_results = get_cloud_nums(image_frame, 8, params_file)
    num_total = sum(num_results)
    num_double = num_results[1] + num_results[2]

    results = tuple(num_results) + (num_total, num_double)

    return results

def extract_filename_params(image_file):
    """Retrieves experiment parameters from image filename.

    Experiment parameters
        params[0]: Acquisition date.
        params[1]: Time-of-flight (ms).
        params[2]: Disorder strength (E_R).
        params[3]: Lattice depth (E_R).
        params[4]: Spin-exchange hold time (ms).
        params[5]: Initial dipole laser power (mW).
        params[6]: The count for duplicate images with identical
            values for the above parameters. Ranges from 0 to N,
            where params[6] = k is the (k+1)-th image with 
            identical parameteters.

    Args
        image_file: String. Filename of image.

    Returns: A list. 
    """
    param_strings = ['201','tof', 'delta', 'ldepth', 'holdtime', 'dipolehigh', '(']
    params_count = len(param_strings)
    param_pos = [image_file.find(param) for param in param_strings]

    params_start = [param_pos[n] + len(param_strings[n]) + 1 for n in range(params_count)]
    params_start[0] = param_pos[0]
    params_start[-1] = param_pos[-1] + 1

    params_end = [image_file.find(' ', pos) for pos in param_pos] 
    params_end[0] = params_start[0] + 10
    params_end[-1] = len(image_file) - 5

    params = [None] * 7
    
    params[0] = image_file[params_start[0]:params_end[0]].replace(sep, '-')
    params[1] = int(image_file[params_start[1] : params_end[1]])
    params[2] = float(image_file[params_start[2] : params_end[2]])
    params[3] = int(image_file[params_start[3] : params_end[3]])
    params[4] = float(image_file[params_start[4] : params_end[4]])
    params[5] = int(image_file[params_start[5] : params_end[5]])

    if param_pos[6] == -1:
        params[6] = 0
    else:
        params[6] = int(image_file[params_start[6] : params_end[6]])

    return params

def get_all_images():
    """Retrieve all images from the database.
    
    Also returns the associated id.
    
    Args
        conn: Database connection.
        schema: String. Schema containing the images table.
        
    Returns: A list-of-tuples: (image_id, frames as a numpy array)
    """
    conn = open_connection()
    rows = get_table(conn, 'image_data', schema='atomic_cloud_images')
    close_connection(conn)

    ids = [row[0] for row in rows]

    images = [convert_frames(row[5], row[4]).reshape(tuple(row[1:4]))
              for row in rows]

    return images, ids

def get_analysis_results():
    """Retrieve the image analysis results from the database.
    
    Args
        conn: Database connection.
        schema: String. Schema containing the image analysis table.
        
    Returns: A list-of-tuples: (image_id, num_center, num_left, num_right,
        num_total, num_double).
    """

    conn = open_connection()
    rows = get_table(conn, 'image_analysis', schema='atomic_cloud_images')
    close_connection(conn)

    return rows

def get_clouds(roi, frame_num=1):
    """Retrieve the cloud in the specified ROI.
    
    Args
        roi: List/tuple: (col1, col2, row1, row2). Region of interest
            where cloud is in the image.
        frame_num: Int. Frame number of the image to return.
        
    Returns: A tuple of (clouds, ids). clouds: Numpy array with shape
        (n_images, n_rows, n_cols). ids: list of length n_images.
    """

    images, ids = get_all_images()

    clouds = np.array([get_region(image[frame_num - 1], *roi)
                      for image in images])

    return clouds, ids

def get_cloud_nums(image, border, params_file):
    """Calculates the total atom number.
    
    Args
        image: Numpy array. Image or region of image
            containing a single atomic cloud.
        border: Int. Size of border in pixels to use for
            background substraction.
        params_file: Path object. CSV file containing parameters
            needed for image analysis.

    Returns: Tuple: (num_center, num_left, num_right)
    """
    image_no_bkgnd = do_background_subtraction(image, border)

    nums = get_ROI_nums(image_no_bkgnd, params_file)

    return nums

def get_frames(spefile, frame_count, n_rows, n_cols, datatype=None):
    """Retrieves each frame from the image file.
    
    Args
        spefile: File object. The .spe image.
        frame_count: Int. Number of frames in the image.
        n_rows: Int. Number of rows in each frame.
        n_cols: Int. Number of columns in each frame.
        datatype: Int. Represents the data type of the
            pixel values. If not given then find it.
        
    Returns: Numpy array containing each frame.
    """
    
    num_pixels = n_rows * n_cols
    shape = (frame_count, n_rows, n_cols)

    if datatype is None:
        datatype = read_spe_file(spefile, 108, 1)
    
    frames_bytes = read_spe_file(
        spefile, 4100, 8 * frame_count * num_pixels, convert_to_int=False)
    
    frames = convert_frames(frames_bytes, datatype).reshape(shape)

    return frames

def get_frame_info(spefile):
    """Gets properties of the frame..

    Properties are the number of frames, columns, rows and datatype of
    the pixel values.

    Args
        spefile: File object. The .spe image.
    
    Returns
        frame_count: Int. Number of frames in the image.
        n_rows: Int. Number of rows in each frame.
        n_cols: Int Number of columns in each frame.
        dataype: Int. Represents the data type of the
            pixel values.
    """
    read_params = ([1446, 1], [42, 2], [656, 2], [108, 1])
    
    [frame_count, n_cols, n_rows, datatype] = [read_spe_file(spefile, *params)
                                               for params in read_params]
    
    return frame_count, n_rows, n_cols, datatype

def get_image_by_id(conn, image_id, frame_num=1):
    """Retrieve the image with the given id from the database.
    
    Args
        conn: Database connection.
        image_id: Int. Image id (primary key column).
        frame_num: Int. Frame number to analyze.
        
    Returns: Numpy array.
    """

    row = get_row_by_id(
        conn, image_id, 'image_data', schema='atomic_cloud_images')
    
    shape = (row[1], row[2], row[3])
    frames = convert_frames(row[5], row[4]).reshape(shape)

    return frames[frame_num - 1]

def get_image_list(path_search, pattern):
    """Gets files matching a specifed pattern in a directory.

    Args
        path_main: List of Path objects. Parent directory containing
            folders with image files.
        dir_search: List of strings. Sub-directories to search.
        pattern: Pattern string. Specify relevant images by filename.

    Result: List of Path objects for full-path filenames.
    """

    filelist = [path_image_file for path_images in path_search
                for path_image_file in path_images.glob(pattern)]

    return filelist

def get_region(image, x1, x2, y1, y2):
    """Grab a region from an image.
    
    The coordinates are pixel numbers along the rows and columns.
    
    Args
        image: A numpy array of the pixel values.
        x1: Int. Start column.
        x2: Int. End column.
        y1: Int. Start row.
        y2: Int. End row.
        
    Returns: A numpy array.
    """

    return image[y1-1 : y2, x1-1 : x2]

def get_ROI_nums(image, params_file):
    """Get the atom number in the ROI for each cloud.
    
    The ROI is given in params_file.
    
    Args
        image: Numpy array. Pixel values.
        params_file: Path object. CSV file containing the parameters
            needed for image analysis.
        
    Returns: List. Atom number in each ROI.
    """

    image_ROIs = get_ROI_params(params_file)

    roi_nums = [3.05**2 * np.sum(get_region(image, *roi)) / 291
                for roi in image_ROIs]

    return roi_nums

def get_ROI_params(params_file):
    """Get the ROI for each atomic cloud.
    
    Args
        params_file: Path object. CSV file containing the parameters
            needed for image analysis..

    Returns: Tuple. ROI for each cloud.
    """

    with open(params_file) as f:
        clouds_params = list(reader(f))

    roi_center = [int(item) for item in clouds_params[0][0:4]]
    roi_left = [int(item) for item in clouds_params[0][4:8]]
    roi_right = [int(item) for item in clouds_params[0][8:12]]

    roi_center[1], roi_center[2] = roi_center[2], roi_center[1]
    roi_left[1], roi_left[2] = roi_left[2], roi_left[1]
    roi_right[1], roi_right[2] = roi_right[2], roi_right[1]

    return roi_center, roi_left, roi_right

def loadspe(image_file, frame_num=1):
    """Loads pixels from .spe image file.
    
    Returns the first frame unless specified.
    
    Args
        image_file: Path object. Full path of the .spe image file.
        frame_num: Int. Frame number to return.
        
    Returns: Numpy array. Image frame.
    """
    
    with open(image_file, mode='rb') as spefile:
        
        frame_count, n_rows, n_cols, datatype = get_frame_info(spefile)

        shape = (frame_count, n_rows, n_cols)
        frames = get_frames(
            spefile, frame_count, n_rows, n_cols, datatype=datatype)
            
    return frames[frame_num - 1]

def read_image_data(spefile):
    """Reads important image data.

    The data is the number of frames, columns, rows, datatype of
    the pixel values and the pixel values as bytes.
    
    Args
        spefile: File object. The .spe image.

    Returns: A list of
        frame_count: Int. Number of frames in the image.
        n_rows: Int. Number of rows in each frame.
        n_cols: Int. Number of columns in each frame.
        datatype: Int. Represents the data type of the
            pixel values.
        pixels_bin: Bytes object. Pixel values.
    """
    read_params = ([1446, 1], [42, 2], [656, 2], [108, 1])
    
    frame_count, n_cols, n_rows, datatype = [read_spe_file(spefile, *params)
                                             for params in read_params]

    num_pixels = frame_count * n_rows * n_cols

    pixels_bin = read_spe_file(
        spefile, 4100, 8 * num_pixels, convert_to_int=False)

    return [frame_count, n_rows, n_cols, datatype, pixels_bin]

def read_spe_file(spefile, position, num_bytes_read,
        byte_order='little', convert_to_int=True):
    """Reads the number of bytes at the specified position.
    
    Args
        file: File object. The .spe image.
        position: Int. Number of offset bytes from the
            beginning of the file.
        num_bytes_read: Int. The number of bytes to read starting at
            'position'.
        byte_order: String. Most significant byte at the end is 'little'
            (default). Most significant byte at the beginning is 'big'.
        convert_to_int: Boolean. Whether to convert the bytes
            to an integer.
    
    Returns: The data as a bytes object or integer.
    """
    
    spefile.seek(position)
    byte_result = spefile.read(num_bytes_read)
    
    if convert_to_int:
        return int.from_bytes(byte_result, byteorder=byte_order, signed=False)
    else:
        return byte_result

def write_experiment_parameters(image_files):
    """Write the experiment parameters to the database.
    
    Args
        image_files: List of Path objects. Full path of image files.
    """

    conn = open_connection()

    exp_params = [tuple(extract_filename_params(image_file))
                  for image_file in image_file]

    insert_exp_params(conn, exp_params, schema='atomic_cloud_images')

    close_connection(conn)

def write_image_data(image_files):
    """Write image data to the database.
    
    Args
        image_files: List of Path objects. Full path of image files.
    """

    image_count = len(image_files)

    exp_params = [extract_filename_params(imgf) for imgf in image_files]

    image_data = [[None] * 6 for n in range(len(image_files))]
    
    for n in range(image_count):
        with open(image_files[n], 'rb') as spefile:
            image_data[n] = read_image_data(spefile)

    conn = open_connection()

    table_name = 'experiment_parameters'
    schema = 'atomic_cloud_images'

    image_ids = [find_image_id(conn, table_name, ep, schema=schema)
                 for ep in exp_params]

    values = [(img_id,) + tuple(img_data)
              for img_id, img_data in zip(image_ids, image_data)]

    insert_image_data(conn, values, schema='atomic_cloud_images')

    close_connection(conn)

