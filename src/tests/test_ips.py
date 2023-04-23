import csv
import pandas as pd
import numpy as np

from numpy import genfromtxt

from src.tests.utils.ips_utils import compute_path

'''
In this scenario, we'll check if the transition from csv to pb, and reading into a pd.Dataframe
are resulting in correct columns
'''


def test_dataframe_columns(ips, expected_magnetics_columns, expected_positions_columns):

    for column in ips.positions.columns:
        assert column in expected_positions_columns
        print(f"Column {column} present in position dataframe columns")
    for column in ips.magnetics.columns:
        assert column in expected_magnetics_columns
        print(f"Column {column} present in magnetics dataframe columns")


'''
In this scenario we'll check if magnetics points recorded after the last ground truth point are
not present in the pd.Dataframe
'''


def test_magnetics_after_record_points(ips):
    assert ips.magnetics.iloc[-1]['t'] <= ips.positions.iloc[-1]['t']
    print(f"Last magnetic time: {ips.magnetics.iloc[-1]['t']}\nLast position time: {ips.positions.iloc[-1]['t']}")


'''
In this scenario, we'll test if the final dataframes contains all the values from the original csv file
'''


def test_dataframes_values(ips, expected_magnetics_columns, expected_positions_columns):
    with open('recordings_csv/10732/positions.csv') as csv_file:
        csv_reader = list(csv.reader(csv_file, delimiter=','))
        for index, row in enumerate(csv_reader[1:]):
            for item_index, item in enumerate(row):
                assert str(float(item)) == str(ips.positions.iloc[index][expected_positions_columns[item_index]])
                print(f"CSV item {item} found for column {expected_positions_columns[item_index]} in dataframe")

    with open('recordings_csv/10732/magnetics.csv') as csv_file:
        csv_reader = list(csv.reader(csv_file, delimiter=','))
        for index, row in enumerate(csv_reader[1:]):
            # magnetics.csv could have items registered after the last time of ground truth point
            # which are not present in the dataframe
            if float(ips.positions.iloc[-1]['t']) < float(row[0]):
                break
            for item_index, item in enumerate(row):
                assert str(float(item)) == str(ips.magnetics.iloc[index][expected_magnetics_columns[item_index]])
                print(f"CSV item {item} found for column {expected_magnetics_columns[item_index]} in dataframe")


'''
Test if the magnetics calculation with the interpolation are made correctly,
by comparing the algorithm output with one that has already x,y calculated
'''


def test_magnetics_calculation(ips):
    test_dataframe = pd.read_csv(compute_path('inputs/magnetics_interpolated.csv'))
    assert ips.magnetics.to_string() == test_dataframe.to_string()
    print(f"Interpolated point calculated\n {ips.magnetics.to_string()}\n Interpolated point expected\n {test_dataframe.to_string()}")


'''
In this scenario we'll check if the calculation for covering ground truth point
receive correct size for plotting
'''


def test_grid_cells(ips):
    cell_size = 5
    ips.set_rect_grid([cell_size, cell_size])
    x_points = []
    for i in range(63, 97, cell_size):
        x_points.append(i)
    y_points = []
    for i in range(61, 91, cell_size):
        y_points.append(i)
    assert len(ips.rect_grid) == len(x_points)+1
    print(f"Grid have {len(ips.rect_grid)} lines, expected: {len(x_points)+1}")
    assert len(ips.rect_grid[0]) == len(y_points)+1
    print(f"Grid have {len(ips.rect_grid)} columns, expected: {len(x_points) + 1}")


'''
In this scenario, we'll check if the correct number of magnetics position are determined for each cell.
This we'll be determined by diving final grid from algorithm, with an know grid that represent the sum of all
magnetics field amplitude for each cell
'''

def test_magnetic_count_cell(ips):
    cell_size = 5
    ips.set_rect_grid([cell_size, cell_size])
    expected_count = genfromtxt(compute_path('inputs/magnetics_count_cell.csv'), delimiter=',')
    raw_grid = genfromtxt(compute_path('inputs/rect_grid_raw.csv'), delimiter=',')
    ips.rect_grid [ ips.rect_grid==0] = np.NaN
    calculated_count = np.rint(np.divide(raw_grid,ips.rect_grid))
    # We'll replace np.NaN with 0 as np.array_equal returns False if np.Nan==np.Nan
    np.nan_to_num(calculated_count, copy=False)
    assert np.array_equal(expected_count,calculated_count)
    print(f"Grid for magnetics point\n {calculated_count}\n Expected grid for magnetics\n{expected_count}")

'''
Here we'll test if each cell has the the correct values for sum of all magnetics field amplitude for the cells
We'll check that by multiplying the output grid with the number of known magnetics points for each cell
'''


def test_magnetics_amplitude_compute(ips):
    cell_size = 5
    ips.set_rect_grid([cell_size, cell_size])
    expected_sum_amplitude = genfromtxt(compute_path('inputs/rect_grid_raw.csv'), delimiter=',')
    magnetics_count = genfromtxt(compute_path('inputs/magnetics_count_cell.csv'), delimiter=',')
    calculated_ampltitudes = np.multiply(magnetics_count,ips.rect_grid)
    # We'll replace np.NaN with 0 as np.array_equal returns False if np.Nan==np.Nan
    np.nan_to_num(calculated_ampltitudes, copy=False)
    # we'll choose allclose instead of array_equal as some multiplication are not exactly
    assert np.allclose(expected_sum_amplitude,calculated_ampltitudes)
    print(f"Grid for amplitude sum\n {calculated_ampltitudes}\n Expected grid for amplitude sum\n{expected_sum_amplitude}")

'''
In this scenario we'll test if the generated matrix for average field amplitude is correct
'''


def test_average_magnetics_amplitude(ips):
    cell_size = 5
    ips.set_rect_grid([cell_size, cell_size])
    expected_grid = genfromtxt(compute_path('inputs/rect_grid_final.csv'), delimiter=',')
    # We'll replace np.NaN with 0 as np.array_equal returns False if np.Nan==np.Nan
    np.nan_to_num(ips.rect_grid, copy=False)
    np.nan_to_num(expected_grid, copy=False)
    assert np.array_equal(ips.rect_grid, expected_grid)
    print(f"Grid for average amplitude\n {ips.rect_grid}\n Expected grid for average amplitude \n{expected_grid}")

