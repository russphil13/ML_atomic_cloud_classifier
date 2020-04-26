from config import config
import psycopg2
from psycopg2.sql import Identifier, SQL
from custom.pgsql import add_table_constraint, create_schema, close_connection
from custom.pgsql import get_schemas, get_tables, new_table, open_connection


schema = 'atomic_cloud_images'
exp_params_table = 'experiment_parameters'
img_data_table = 'image_data'
img_analysis_table = 'image_analysis'


exp_params_cols = ['image_id', 'acquisition_date', 'time_of_flight',
    'disorder_strength', 'lattice_depth', 'hold_time', 'initial_dipole_power',
    'duplicate_count'
    ]
exp_params_dtypes = ['serial', 'text', 'integer', 'real', 'integer', 'real',
    'integer', 'integer'
    ]
exp_params_col_constr = ['primary key', 'not null', 'not null', 'not null',
    '','not null', 'not null', 'not null'
    ]
exp_params_table_constr = (f'uniq_params unique ({exp_params_cols[1]}, {exp_params_cols[2]}, '
    f'{exp_params_cols[3]}, {exp_params_cols[4]}, {exp_params_cols[6]}, '
    f'{exp_params_cols[7]});'
    )

img_data_cols = ['image_id','frame_count','rows_count', 'col_count', 'data_type',
    'pixels'
    ]
img_data_dtypes = ['integer', 'integer', 'integer', 'integer', 'integer',
    'bytea'
    ]
img_data_col_constr = ['primary key', 'not null', 'not null', 'not null',
    'not null', 'not null'
    ]
img_data_table_constr = (f'fkey_data foreign key ({img_data_cols[0]}) '
    f'references {{ref_table}} '
    f'on update cascade on delete cascade'
    ) 

img_analysis_cols = ['image_id', 'num_center', 'num_left', 'num_right',
    'num_total', 'num_double'
    ]
img_analysis_dtypes = ['integer', 'real', 'real', 'real', 'real', 'real']
img_analysis_col_constr = ['primary key', 'not null', 'not null', 'not null',
    'not null', 'not null'
    ]
img_analysis_table_constr = (f'fkey_analysis foreign key ({img_analysis_cols[0]}) '
    f'references {{ref_table}} on update cascade on delete cascade'
    )

conn = open_connection()

create_schema(conn, schema)

new_table(conn, exp_params_table, exp_params_cols, exp_params_dtypes,
    exp_params_col_constr, schema=schema)

add_table_constraint(conn, exp_params_table, exp_params_table_constr,
    schema=schema)

new_table(conn, img_data_table, img_data_cols, img_data_dtypes,
    img_data_col_constr, schema=schema)

add_table_constraint(conn, img_data_table, img_data_table_constr,
    reference=exp_params_table, schema=schema)

new_table(conn, img_analysis_table, img_analysis_cols, img_analysis_dtypes,
    img_analysis_col_constr, schema=schema)

add_table_constraint(conn, img_analysis_table, img_analysis_table_constr,
    reference=exp_params_table, schema=schema)

get_table_names(conn, schema) 
close_connection(conn)
