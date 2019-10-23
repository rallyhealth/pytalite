from __future__ import absolute_import, division, print_function

from functools import reduce

import pyspark.sql.functions as F
from pyspark.ml.feature import VectorAssembler
from pyspark.sql.types import DoubleType

__all__ = ['preprocess', 'p1_proba', 'add_column_index']


def preprocess(df, y_column, model_input_col, columns_to_exclude, create_id=False):
    """"""
    df = df.fillna(-1)
    if create_id:
        assert '_id_' not in df.columns
        df = add_column_index(df)

    df.cache()
    df.count()
    columns_to_exclude = set(columns_to_exclude).union({y_column, '_id_'})
    assembler = VectorAssembler(inputCols=[c for c in df.columns if c not in columns_to_exclude],
                                outputCol=model_input_col)

    return df, assembler


@F.udf(returnType=DoubleType())
def p1_proba(x):
    return float(x[1])


def add_column_index(df):
    # Create new column names
    old_columns = df.schema.names
    new_columns = old_columns + ["_id_"]

    # Add Column index
    df_indexed = df.rdd.zipWithIndex().map(lambda x: x[0] + (x[1],)).toDF()
    default_columns = df_indexed.schema.names

    # Rename all the columns
    new_df = reduce(lambda data, idx: data.withColumnRenamed(default_columns[idx],
                                                             new_columns[idx]), range(len(default_columns)), df_indexed)
    return new_df
