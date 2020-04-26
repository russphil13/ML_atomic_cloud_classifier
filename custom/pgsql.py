from config import config
import psycopg2
from psycopg2.extras import execute_values
from psycopg2.sql import Identifier, SQL

def add_table_constraint(conn, table_name, constraint, reference=None,
        schema=None):
    """Add constraint to existing table affecting multiple columns.
    
    Args
        conn: Database connection.
        table_name: Name of table in database.
        constraint: String with pgsql syntax.
        reference: A table being referenced in the constraint.
        schema: Schema containing the table: table_name.
    """
    
    cmd_str = 'alter table if exists {table} add constraint ' + constraint

    if reference is None:
        cmd = SQL(cmd_str).format(table=Identifier(table_name))
    else:
        cmd = SQL(cmd_str).format(table=Identifier(table_name),
            ref_table=Identifier(reference)
            )

    try:
        with conn.cursor() as curs:
            if schema is not None:
                curs.execute(f'set search_path={schema};')

            curs.execute(cmd)
            print(f'Added constraint to table: {table_name} successfully.')
            
        conn.commit()
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)

def close_connection(conn):
    """Close connection to local server.
    
    Args
        conn: Database connection.
    """
    
    try:
        conn.close()
        print('Database connection closed.')
    except () as error:
        print(error)

def create_schema(conn, schema_name):
    """Create a new schema in the database.
    
    Args
        conn: Database connection.
        schema_name: Name of schema to create in database."""
    
    cmd = f'create schema if not exists {schema_name};'
    
    try:
        with conn.cursor() as curs:
            curs.execute(cmd)
            print(f'Created schema: {schema_name} successfully.')

        conn.commit()
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)

def create_table(conn, cmd, schema=None):
    """Create a table in the current schema.
    
    Args
        conn: Database connection.
        cmd: SQL string for creating the table.
        schema: Schema to create the table in.
    """
    
    try:
        with conn.cursor() as curs:
            if schema is not None:
                curs.execute(f'set search_path={schema}')
            curs.execute(cmd)

        conn.commit()
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)

def get_image_ids(conn, table_name, schema=None):
    """Retrieve the image_ids for all rows in the table.
    
    Args
        conn: Database connection.
        table_name: Table to search in.
        schema: Schema to search in.
        
    Returns: A list.
    """

    query = SQL(f'select image_id from {{table}};').format(
        table=Identifier(table_name))

    try:
        if schema is not None:
            with conn.cursor() as curs:
                curs.execute(f'set search_path={schema};')
        with conn.cursor(name='my_cursor', withhold=True) as curs:
            curs.execute(query)
            rows = curs.fetchall()
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)

    image_ids = [row[0] for row in rows]

    return image_ids

def get_row_by_id(conn, image_id, table_name, schema=None):
    """Get the row containg 'image_id' in the specified table.
    
    Args
        conn: Database connection.
        image_id: Image id of interest.
        table_name: Table to search in.
        schema: Schema to search in.
        
    Returns: Row from database.
    """

    query = SQL(f'select * from {{table}} '
            f'where image_id = %s').format(table=Identifier(table_name))

    try:
        if schema is not None:
            with conn.cursor() as curs:
                curs.execute(f'set search_path={schema}')
        with conn.cursor(name='my_cursor', withhold=True) as curs:
            curs.execute(query, (image_id,))
            row = curs.fetchone()
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)

    return row

def get_schemas(conn):
    """Get a list of the schemas in the database.
    
    Args
        conn: Database connection.
        
    Returns: A list.
    """

    try:
        with conn.cursor(name='my_cursor', withhold=True) as curs:
            curs.execute('select schema_name '
                    'from information_schema.schemata;')
            schema_names = curs.fetchall()

        for schema in schema_names:
            print(schema)
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)

    return schema_names

def get_table(conn, table_name, schema=None):
    """Query table names from database.
    
    Args
        conn: Database connection.
        schema: Name of schema to search in.
        
    Returns: A list.
    """
    
    query = SQL('select * from {table};').format(table=Identifier(table_name))
    
    try:
        if schema is not None:
            with conn.cursor() as curs:
                curs.execute(f'set search_path={schema}')

        with conn.cursor(name='my_cursor', withhold=True) as curs:
            curs.execute(query)
            rows = curs.fetchall()
    except (Exception, psycopg2.DatabaseError) as error:
            print(error)
    
    return rows

def get_table_names(conn, schema):
    """Query table names from database.
    
    Args
        conn: Database connection.
        schema: Name of schema to search in.
        
    Returns: A list.
    """
    
    table_names = None
    
    query = ('select table_name '
        'from information_schema.tables '
        'where table_schema=%s;')
    
    try:
        with conn.cursor(name='my_cursor', withhold=True) as curs:
            curs.execute(query, (schema,))
            table_names = curs.fetchall()
            print('The number of tables is', curs.rowcount)
        
            for table_name in table_names:
                print(table_name[0])
    except (Exception, psycopg2.DatabaseError) as error:
            print(error)
    
    return table_names

def insert(conn, cmd, values, schema=None):
    """Handling insert commands.
    
    Args
        conn: Database connection.
        cmd: SQL sting with insert command.
        values: Value(s) to insert into new rows.
        schema: Schema containing the table to insert into.
    """
    
    try:
        with conn.cursor() as curs:
            if schema is not None:
                curs.execute(f'set search_path={schema}')

            if type(values) is tuple:
                curs.execute(cmd, values)
            elif type(values) is list:
                execute_values(curs, cmd, values)
            else:
                curs.execute(cmd, (values,))
        
        conn.commit()
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)

def insert_analysis_data(conn, values, schema=None):
    """Insert analysis data into table.
    
    Args
        conn: Database connection.
        values: Tuple or list of tuples containing values for each
            of the below columns.
        schema: Schema to search in.
    """

    columns = ['image_id',
        'num_center',
        'num_left',
        'num_right',
        'num_total',
        'num_double'
        ]

    cmd_str = 'insert into image_analysis ({cols})'

    sql_columns = SQL(', ').join([Identifier(column) for column in columns])

    if type(values) is list:
        #for execute_values (see 'insert')
        cmd_str = cmd_str + 'values %s;'
    else:
        #for execute (see 'insert')
        cmd_str = cmd_str + 'values (%s %s %s %s %s %s);'

    cmd = SQL(cmd_str).format(cols=sql_columns)

    insert(conn, cmd, values, schema=schema)

def insert_exp_params(conn, values, columns=None, schema=None):
    """Insert experiment parameters of image into table.
    
    Args
        conn: Database connection.
        table: Name of table in the database.
        values: Tuple or list of tuples containing values for each
            column.
        columns: Column names to insert data into.
        schema: Schema table exists in."""
    
    if columns is None:
        columns = ['acquisition_date',
            'time_of_flight',
            'disorder_strength',
            'lattice_depth',
            'hold_time',
            'initial_dipole_power',
            'duplicate_count'
            ]
    
    sql_columns = SQL(', ').join([Identifier(col) for col in columns])

    cmd_str = 'insert into experiment_parameters ({cols}) '
     
    if type(values) is list:
        #for execute_values (see 'insert')
        cmd_str = cmd_str + 'values %s;'
    else:
        #for execute (see 'insert')
        cmd_str = cmd_str + 'values (%s %s %s %s %s %s %s);'

    cmd = SQL(cmd_str).format(cols=sql_columns)
    
    insert(conn, cmd, values, schema=schema)

def insert_image_data(conn, values, schema=None):
    """Insert image data into table.
    
    Args
        conn: Database connection.
        values: Tuple or list of tuples containing values for each
            of the below columns.
        schema: Schema to search in.
    """

    columns = ['image_id',
        'frame_count',
        'rows_count',
        'col_count',
        'data_type',
        'pixels'
        ]

    sql_columns = SQL(', ').join([Identifier(column) for column in columns])

    cmd_str = 'insert into image_data ({cols}) '
    
    if type(values) is list:
        values_nopix = [row[:-1] for row in values]
        pix2bin = [psycopg2.Binary(row[-1]) for row in values]

        values_pix2bin = [item[0] + (item[1],)
            for item in zip(values_nopix, pix2bin)]

        #for execute_values (see 'insert')
        cmd_str = cmd_str + 'values %s;'
    else:
        values_pix2bin = values[:-1] + tuple(psycopg2.Binary(values[-1]))

        #for execute (see 'insert')
        cmd_str = cmd_str + 'values (%s %s %s %s %s %s);'

    cmd = SQL(cmd_str).format(cols=sql_columns)

    insert(conn, cmd, values_pix2bin, schema=schema)

def find_image_id(conn, table_name, values, schema=None):
    """Get the image id based on the experiment parameters.
    
    Args
        conn: Database connection.
        values: Tuple of values for the columns below.
        schema: Schema to search in.
        
    Returns: An integer.
    """

    columns = ['acquisition_date',
        'time_of_flight',
        'disorder_strength',
        'lattice_depth',
        'hold_time',
        'initial_dipole_power',
        'duplicate_count'
        ]

    query_str = (f'select image_id '
        f'from {{table}} '
        f'where {{col1}} = %s '
        f'and {{col2}} = %s '
        f'and {{col3}} = %s '
        f'and {{col4}} = %s '
        f'and {{col5}} = %s '
        f'and {{col6}} = %s '
        f'and {{col7}} = %s;'
        )

    query = SQL(query_str).format(
        table=Identifier(table_name),
        col1=Identifier(columns[0]),
        col2=Identifier(columns[1]),
        col3=Identifier(columns[2]),
        col4=Identifier(columns[3]),
        col5=Identifier(columns[4]),
        col6=Identifier(columns[5]),
        col7=Identifier(columns[6])
        )

    try:
        if schema is not None:
            with conn.cursor() as curs:
                curs.execute(f'set search_path={schema};')
        with conn.cursor(name='my_cursor', withhold=True) as curs:
            curs.execute(query, values)
            image_id = curs.fetchone()[0]
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)

    return image_id

def new_table(conn, table_name, columns, data_types, constraints,
        schema=None):

    """Creates a new table.
    
    Prepare the pgsql statement dynamically.
    
    Args
        conn: Database connection.
        table_name: Name of the new table in the database.
        columns: Column names for table.
        data_types: Types for each column.
        constraints: Individual column constraints.
        schema: Schema to create table in.
    """

    col_args_list = [f'{arg[0]} {arg[1]} {arg[2]}'.rstrip()
        for arg in zip(columns, data_types, constraints)]

    col_args = ', '.join(col_args_list)

    cmd_str = f'create table if not exists {{table}} (%s);' % col_args

    cmd = SQL(cmd_str).format(table=Identifier(table_name))

    create_table(conn, cmd, schema=schema)

    print(f'Created table: {table_name} successfully.')
     
def open_connection():
    """ Open connection to local server."""
    
    conn = None
    
    try:
        params = config()
        
        conn = psycopg2.connect(**params)
        
#        with conn.cursor() as curs:
#            print('PostgreSQL database version.')
#            curs.execute("select version()")   
#            db_version = curs.fetchone()
#            print(db_version)
    except () as error:
        print(error)
    finally:
        if conn is not None:
            print('Database connection opened.')
    
    return conn

def print_table(conn, table_name, schema=None):
    """Display contents of the specified table.
    
    Args
        conn: Database connection.
        table_name: Name of table of interest.
        schema: Schema to search in.
    """

    query = SQL('select * from {table} order by image_id;').format(
        table=Identifier(table_name))

    try:
        if schema is not None:
            with conn.cursor() as curs:
                curs.execute(f'set search_path={schema};')
        with conn.cursor(name='my_cursor', withhold=True) as curs:
            curs.execute(query)
            row = curs.fetchone()
        
            while row is not None:
                print(row[:-1])
                row = curs.fetchone()

    except (Exception, psycopg2.DatabaseError) as error:
        print(error)

